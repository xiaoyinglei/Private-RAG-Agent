"""Tests for the skill layer — Phase 1: inline local skills.

Covers: loader, catalog, context, policy, invocation, and integration.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from rag.agent.core.context import AgentRunConfig
from rag.agent.core.tool_schema import AgentMessageAssembler
from rag.agent.loop.state import create_loop_state
from rag.agent.service import AgentService
from rag.agent.skills.assets import (
    MaterializeSkillAssetInput,
    MaterializeSkillAssetOutput,
    MaterializeSkillAssetTool,
)
from rag.agent.skills.catalog import SkillCatalog
from rag.agent.skills.context import (
    SKILL_PROMPT_GUIDANCE,
    build_skills_prompt_section,
    render_active_loaded_skills,
    render_loaded_skill,
    render_skill_listing,
)
from rag.agent.skills.invocation import (
    INVOKE_SKILL_SPEC,
    InvokeSkillInput,
    InvokeSkillOutput,
    make_invoke_skill_runner,
)
from rag.agent.skills.loader import (
    SkillLoadError,
    load_skill_body,
    load_skill_from_file,
    scan_and_load_skills,
)
from rag.agent.skills.models import (
    SkillInvocation,
    SkillSource,
    SkillState,
    SkillSummary,
)
from rag.agent.skills.policy import SkillPolicy
from rag.agent.tools.registry import ToolExecutionContext
from rag.agent.workspace import open_workspace
from rag.schema.runtime import AccessPolicy

# ── Helpers ──────────────────────────────────────────────────────────


def _write_skill(
    root: Path,
    name: str,
    description: str = "A test skill",
    **extra: object,
) -> Path:
    """Write a minimal SKILL.md and return its path."""
    skill_dir = root / ".agents" / "skills" / name
    skill_dir.mkdir(parents=True)
    lines = [
        "---",
        f"name: {name}",
        f"description: {description}",
    ]
    for key, value in extra.items():
        if isinstance(value, list):
            lines.append(f"{key}:")
            for item in value:
                # Quote values that contain YAML-special chars (*, {, [, etc.)
                item_str = str(item)
                if set(item_str) & {"*", "{", "[", "]", "&", "!", "#", "?", ":", "@"}:
                    item_str = f'"{item_str}"'
                lines.append(f"  - {item_str}")
        elif isinstance(value, bool):
            lines.append(f"{key}: {str(value).lower()}")
        else:
            val_str = str(value)
            if set(val_str) & {"*", "{", "[", "]", "&", "!", "#", "?", ":", "@"}:
                val_str = f'"{val_str}"'
            lines.append(f"{key}: {val_str}")
    lines.extend(["---", "", f"# {name}", "", f"Body of {name}."])
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text("\n".join(lines))
    return skill_md


def _run_config() -> AgentRunConfig:
    return AgentRunConfig(
        run_id="run-test",
        thread_id="thread-test",
        agent_type="generic",
        max_depth=2,
        access_policy=AccessPolicy(),
    )


# ── Loader tests ─────────────────────────────────────────────────────


class TestLoader:
    def test_valid_skill_loads(self):
        """A well-formed SKILL.md should load into a SkillManifest."""
        with tempfile.TemporaryDirectory() as d:
            skill_md = _write_skill(Path(d), "test-skill", "Test description")
            manifest = load_skill_from_file(skill_md, SkillSource.PROJECT)

            assert manifest.name == "test-skill"
            assert manifest.skill_id == "project:test-skill"
            assert manifest.description == "Test description"
            assert manifest.source == SkillSource.PROJECT
            assert manifest.content_fingerprint != ""
            assert len(manifest.content_fingerprint) == 64  # SHA-256

    def test_missing_frontmatter(self):
        """A file without YAML frontmatter should raise SkillLoadError."""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "SKILL.md"
            p.write_text("Just markdown, no frontmatter")
            with pytest.raises(SkillLoadError, match="missing YAML frontmatter"):
                load_skill_from_file(p, SkillSource.PROJECT)

    def test_missing_required_field(self):
        """Missing 'description' should raise SkillLoadError."""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "SKILL.md"
            p.write_text("---\nname: only-name\n---\nbody")
            with pytest.raises(SkillLoadError, match="missing required field"):
                load_skill_from_file(p, SkillSource.PROJECT)

    def test_unknown_fields_stored_in_extra(self):
        """Unknown fields must NOT be rejected — they go into extra."""
        for field in ("context", "agent", "model", "effort", "hooks", "license"):
            with tempfile.TemporaryDirectory() as d:
                skill_dir = Path(d) / ".agents" / "skills" / "extra-test"
                skill_dir.mkdir(parents=True)
                sf = skill_dir / "SKILL.md"
                sf.write_text(
                    f"---\nname: extra-test\ndescription: d\n{field}: some-value\n---\nbody"
                )
                manifest = load_skill_from_file(sf, SkillSource.PROJECT)
                assert manifest.extra.get(field) == "some-value", (
                    f"Field '{field}' should be in extra"
                )

    def test_allowed_tools_parses(self):
        """allowed_tools should be parsed as a tuple."""
        with tempfile.TemporaryDirectory() as d:
            skill_md = _write_skill(
                Path(d), "t1", "desc",
                allowed_tools=["read_file", "search_knowledge"],
            )
            manifest = load_skill_from_file(skill_md, SkillSource.PROJECT)
            assert manifest.allowed_tools == ("read_file", "search_knowledge")

    def test_paths_parses(self):
        """paths should be parsed as a tuple."""
        with tempfile.TemporaryDirectory() as d:
            skill_md = _write_skill(
                Path(d), "t2", "desc",
                paths=["**/*.xlsx", "docs/**"],
            )
            manifest = load_skill_from_file(skill_md, SkillSource.PROJECT)
            assert manifest.path_patterns == ("**/*.xlsx", "docs/**")
            assert manifest.has_path_filter is True

    def test_disable_model_invocation(self):
        """disable_model_invocation should be parsed as bool."""
        with tempfile.TemporaryDirectory() as d:
            skill_md = _write_skill(
                Path(d), "t3", "desc",
                disable_model_invocation=True,
            )
            manifest = load_skill_from_file(skill_md, SkillSource.PROJECT)
            assert manifest.disable_model_invocation is True

    def test_invalid_allowed_tools_raises(self):
        """Non-list allowed_tools should raise."""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "SKILL.md"
            p.write_text("---\nname: x\ndescription: d\nallowed_tools: not_a_list\n---\nbody")
            with pytest.raises(SkillLoadError, match="must be a list"):
                load_skill_from_file(p, SkillSource.PROJECT)

    def test_scan_and_load_dedupes_by_path(self):
        """Duplicate paths should be skipped."""
        with tempfile.TemporaryDirectory() as d:
            _write_skill(Path(d), "dup-skill", "desc")
            manifests = scan_and_load_skills(Path(d), repo_root=Path(d))
            assert len(manifests) == 1
            # Second scan should not add duplicates
            manifests2 = scan_and_load_skills(Path(d), repo_root=Path(d))
            assert len(manifests2) == 1

    def test_load_skill_body(self):
        """load_skill_body should return only the markdown after frontmatter."""
        with tempfile.TemporaryDirectory() as d:
            skill_md = _write_skill(Path(d), "body-test", "desc")
            manifest = load_skill_from_file(skill_md, SkillSource.PROJECT)
            body = load_skill_body(manifest)
            assert "Body of body-test." in body
            assert "---" not in body  # frontmatter stripped


# ── Catalog tests ────────────────────────────────────────────────────


class TestCatalog:
    def test_empty_catalog(self):
        catalog = SkillCatalog()
        assert len(catalog) == 0
        assert catalog.listing_for_prompt() == ""

    def test_find(self):
        with tempfile.TemporaryDirectory() as d:
            _write_skill(Path(d), "find-me", "findable skill")
            manifests = scan_and_load_skills(Path(d), repo_root=Path(d))
            catalog = SkillCatalog(manifests)
            found = catalog.find("find-me")
            assert found is not None
            assert found.name == "find-me"
            assert found.skill_id == "project:find-me"
            assert catalog.find("nope") is None

    def test_listing_within_budget(self):
        with tempfile.TemporaryDirectory() as d:
            _write_skill(Path(d), "s1", "First skill for testing")
            _write_skill(Path(d), "s2", "Second skill for testing")
            manifests = scan_and_load_skills(Path(d), repo_root=Path(d))
            catalog = SkillCatalog(manifests)
            listing = catalog.listing_for_prompt(max_chars=5000)
            assert "project:s1" in listing
            assert "project:s2" in listing

    def test_listing_truncation(self):
        """When budget is tight, non-bundled descriptions should truncate."""
        with tempfile.TemporaryDirectory() as d:
            long_desc = "x" * 200
            _write_skill(Path(d), "long-skill", long_desc)
            _write_skill(Path(d), "short", "brief")
            manifests = scan_and_load_skills(Path(d), repo_root=Path(d))
            catalog = SkillCatalog(manifests)
            listing = catalog.listing_for_prompt(max_chars=100)
            # Should still include both skill names
            assert "long-skill" in listing
            assert "short" in listing

    def test_listing_excludes_active_skill_ids(self):
        with tempfile.TemporaryDirectory() as d:
            _write_skill(Path(d), "loaded-one", "Already loaded")
            _write_skill(Path(d), "available-one", "Still available")
            manifests = scan_and_load_skills(Path(d), repo_root=Path(d))
            catalog = SkillCatalog(manifests)

            listing = catalog.listing_for_prompt(
                max_chars=5000,
                exclude_skill_ids=frozenset({"project:loaded-one"}),
            )

            assert "project:loaded-one" not in listing
            assert "project:available-one" in listing

    def test_search(self):
        with tempfile.TemporaryDirectory() as d:
            _write_skill(Path(d), "pdf-reader", "Read PDF files", when_to_use="pdfs")
            _write_skill(Path(d), "code-review", "Review code")
            manifests = scan_and_load_skills(Path(d), repo_root=Path(d))
            catalog = SkillCatalog(manifests)
            results = catalog.search("pdf")
            assert len(results) == 1
            assert results[0].name == "pdf-reader"

    def test_load_returns_loaded_skill(self):
        with tempfile.TemporaryDirectory() as d:
            _write_skill(Path(d), "load-test", "Loading test")
            manifests = scan_and_load_skills(Path(d), repo_root=Path(d))
            catalog = SkillCatalog(manifests)
            loaded = catalog.load("load-test", iteration=5)
            assert loaded is not None
            assert loaded.manifest.name == "load-test"
            assert loaded.manifest.skill_id == "project:load-test"
            assert loaded.loaded_at_iteration == 5
            assert "Body of load-test." in loaded.content

    def test_duplicate_names_keep_distinct_skill_ids(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as ext:
            _write_skill(Path(d), "dup", "Project version")
            ext_skill = Path(ext) / "dup"
            ext_skill.mkdir()
            (ext_skill / "SKILL.md").write_text(
                "---\nname: dup\ndescription: External version\n---\nexternal"
            )
            manifests = scan_and_load_skills(
                Path(d),
                repo_root=Path(d),
                extra_dirs=[Path(ext)],
            )
            catalog = SkillCatalog(manifests)

            listing = catalog.listing_for_prompt(max_chars=5000)
            assert "project:dup" in listing
            assert "external:dup" in listing
            assert catalog.find("project:dup").source == SkillSource.PROJECT
            assert catalog.find("external:dup").source == SkillSource.EXTERNAL
            assert catalog.find("dup") is None


# ── Context tests ────────────────────────────────────────────────────


class TestContext:
    def test_guidance_has_blocking_requirement(self):
        assert "BLOCKING REQUIREMENT" in SKILL_PROMPT_GUIDANCE
        assert "invoke_skill" in SKILL_PROMPT_GUIDANCE

    def test_render_skill_listing(self):
        with tempfile.TemporaryDirectory() as d:
            _write_skill(Path(d), "ctx-test", "Context test")
            manifests = scan_and_load_skills(Path(d), repo_root=Path(d))
            catalog = SkillCatalog(manifests)
            listing = render_skill_listing(catalog)
            assert "<available_skills>" in listing
            assert "ctx-test" in listing

    def test_render_loaded_skill(self):
        with tempfile.TemporaryDirectory() as d:
            _write_skill(Path(d), "load-render", "Render test")
            manifests = scan_and_load_skills(Path(d), repo_root=Path(d))
            catalog = SkillCatalog(manifests)
            loaded = catalog.load("load-render", iteration=1)
            rendered = render_loaded_skill(loaded, args="hello")
            assert '<loaded_skill id="project:load-render" name="load-render"' in rendered
            assert "Base directory for this skill:" in rendered
            assert "Body of load-render." in rendered

    def test_args_substitution(self):
        """$ARGUMENTS should be replaced in the skill body."""
        with tempfile.TemporaryDirectory() as d:
            skill_dir = Path(d) / ".agents" / "skills" / "args-test"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                "---\nname: args-test\ndescription: Test args\n---\n"
                "Use $ARGUMENTS to process input."
            )
            manifests = scan_and_load_skills(Path(d), repo_root=Path(d))
            catalog = SkillCatalog(manifests)
            loaded = catalog.load("args-test", iteration=1)
            rendered = render_loaded_skill(loaded, args="my-input")
            assert "my-input" in rendered
            assert "$ARGUMENTS" not in rendered

    def test_skill_dir_substitution(self):
        """${SKILL_DIR} should be replaced with the skill's directory path."""
        with tempfile.TemporaryDirectory() as d:
            skill_dir = Path(d) / ".agents" / "skills" / "dir-test"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                "---\nname: dir-test\ndescription: Test dir\n---\n"
                "References are at ${SKILL_DIR}/references/."
            )
            manifests = scan_and_load_skills(Path(d), repo_root=Path(d))
            catalog = SkillCatalog(manifests)
            loaded = catalog.load("dir-test", iteration=1)
            rendered = render_loaded_skill(loaded)
            assert str(skill_dir.resolve()) in rendered
            assert "${SKILL_DIR}" not in rendered

    def test_skill_substitution_does_not_rewrite_prefixed_variables(self):
        with tempfile.TemporaryDirectory() as d:
            skill_dir = Path(d) / ".agents" / "skills" / "safe-vars"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                "---\nname: safe-vars\ndescription: Safe vars\n---\n"
                "Use $SKILL_DIR/references with $ARGUMENTS.\n"
                "Keep $SKILL_DIRECTORY and $ARGUMENTS_SUFFIX unchanged."
            )
            catalog = SkillCatalog(scan_and_load_skills(Path(d), repo_root=Path(d)))
            loaded = catalog.load("project:safe-vars", iteration=1)

            rendered = render_loaded_skill(loaded, args="input.xlsx")

            assert f"Use {skill_dir.resolve()}/references with input.xlsx." in rendered
            assert "$SKILL_DIRECTORY" in rendered
            assert "$ARGUMENTS_SUFFIX" in rendered

    def test_build_prompt_section(self):
        with tempfile.TemporaryDirectory() as d:
            _write_skill(Path(d), "prompt-test", "Prompt section test")
            manifests = scan_and_load_skills(Path(d), repo_root=Path(d))
            catalog = SkillCatalog(manifests)
            section = build_skills_prompt_section(catalog)
            assert SKILL_PROMPT_GUIDANCE in section
            assert "project:prompt-test" in section

    def test_build_prompt_section_empty_catalog_is_empty(self):
        assert build_skills_prompt_section(SkillCatalog()) == ""

    def test_build_prompt_section_excludes_loaded_skill_from_available_listing(self):
        with tempfile.TemporaryDirectory() as d:
            _write_skill(Path(d), "loaded-one", "Already loaded")
            _write_skill(Path(d), "available-one", "Still available")
            catalog = SkillCatalog(scan_and_load_skills(Path(d), repo_root=Path(d)))
            loaded = catalog.load("project:loaded-one", iteration=2)
            state = SkillState()
            state.active["project:loaded-one"] = loaded.to_ref()

            section = build_skills_prompt_section(catalog, skill_state=state)
            available = section.rsplit("<available_skills>", 1)[1].split("</available_skills>", 1)[0]

            assert "project:loaded-one" not in available
            assert "project:available-one" in available
            assert '<loaded_skill id="project:loaded-one"' in section

    def test_active_loaded_skills_render_from_state(self):
        with tempfile.TemporaryDirectory() as d:
            _write_skill(Path(d), "active-test", "Active render test")
            catalog = SkillCatalog(scan_and_load_skills(Path(d), repo_root=Path(d)))
            loaded = catalog.load("project:active-test", iteration=2)
            state = SkillState()
            state.active["project:active-test"] = loaded.to_ref(args="abc")

            rendered = render_active_loaded_skills(state)
            assert "<loaded_skills>" in rendered
            assert '<loaded_skill id="project:active-test"' in rendered
            assert "Body of active-test." in rendered
            assert "abc" in rendered

    def test_active_loaded_skill_warns_when_file_changed(self):
        with tempfile.TemporaryDirectory() as d:
            skill_md = _write_skill(Path(d), "changed-test", "Changed render test")
            catalog = SkillCatalog(scan_and_load_skills(Path(d), repo_root=Path(d)))
            loaded = catalog.load("project:changed-test", iteration=2)
            state = SkillState()
            state.active["project:changed-test"] = loaded.to_ref()

            skill_md.write_text(
                "---\nname: changed-test\ndescription: Changed render test\n---\n"
                "# changed-test\n\nChanged body.\n"
            )

            rendered = render_active_loaded_skills(state)

            assert 'code="skill_content_changed_on_resume"' in rendered
            assert "Changed body." in rendered

    def test_agent_message_assembler_injects_skill_context(self):
        state = create_loop_state(task="test", run_config=_run_config())

        def skill_context_provider(_state):
            return "<loaded_skills>loaded</loaded_skills>"

        from rag.agent.builtin.generic import GENERIC_AGENT

        msg = AgentMessageAssembler(
            skill_context_provider=skill_context_provider,
        ).build_system_message(
            definition=GENERIC_AGENT,
            state=state,
            visible_tool_names=["invoke_skill"],
        )
        assert "<loaded_skills>loaded</loaded_skills>" in msg.content


# ── Policy tests ─────────────────────────────────────────────────────


class TestPolicy:
    def test_default_policy_project_only(self):
        policy = SkillPolicy()
        assert policy.is_source_enabled(SkillSource.PROJECT) is True
        assert policy.is_source_enabled(SkillSource.USER) is False
        assert policy.is_source_enabled(SkillSource.BUNDLED) is False

    def test_disabled_skill(self):
        policy = SkillPolicy(disabled_skills=frozenset({"blocked-skill"}))
        with tempfile.TemporaryDirectory() as d:
            _write_skill(Path(d), "blocked-skill", "Should be blocked")
            manifests = scan_and_load_skills(Path(d), repo_root=Path(d))
            manifest = manifests[0]
            assert policy.is_skill_enabled(manifest) is False

    def test_can_autoload_project(self):
        policy = SkillPolicy()
        with tempfile.TemporaryDirectory() as d:
            _write_skill(Path(d), "auto", "Auto-load")
            manifests = scan_and_load_skills(Path(d), repo_root=Path(d))
            assert policy.can_autoload(manifests[0]) is True


# ── Invocation tests ─────────────────────────────────────────────────


class TestInvocation:
    def test_spec_registered(self):
        assert INVOKE_SKILL_SPEC.name == "invoke_skill"
        assert INVOKE_SKILL_SPEC.input_model is InvokeSkillInput
        assert INVOKE_SKILL_SPEC.output_model is InvokeSkillOutput

    @pytest.mark.anyio
    async def test_runner_returns_loaded_skill(self):
        with tempfile.TemporaryDirectory() as d:
            _write_skill(Path(d), "invoke-test", "Invocation test")
            manifests = scan_and_load_skills(Path(d), repo_root=Path(d))
            catalog = SkillCatalog(manifests)
            runner = make_invoke_skill_runner(catalog)

            result = await runner(InvokeSkillInput(name="invoke-test"), None)
            out = InvokeSkillOutput.model_validate(result)
            assert out.success is True
            assert out.name == "invoke-test"
            assert out.skill_id == "project:invoke-test"
            assert out.source == "project"
            assert len(out.fingerprint) == 16
            assert "Body of invoke-test." in out.loaded_content

    @pytest.mark.anyio
    async def test_runner_records_active_skill_ref_in_loop_state(self):
        from rag.agent.tools.registry import ToolExecutionContext

        with tempfile.TemporaryDirectory() as d:
            _write_skill(Path(d), "state-test", "State test")
            catalog = SkillCatalog(scan_and_load_skills(Path(d), repo_root=Path(d)))
            runner = make_invoke_skill_runner(catalog)
            run_config = _run_config()
            state = create_loop_state(task="test", run_config=run_config)

            result = await runner(
                InvokeSkillInput(name="project:state-test", args="input.xlsx"),
                ToolExecutionContext(run_config=run_config, state=state),
            )
            out = InvokeSkillOutput.model_validate(result)

            assert out.success is True
            assert "project:state-test" in state["skill_state"].active
            ref = state["skill_state"].active["project:state-test"]
            assert ref.skill_id == "project:state-test"
            assert ref.args == "input.xlsx"

    @pytest.mark.anyio
    async def test_runner_skill_not_found(self):
        catalog = SkillCatalog()
        runner = make_invoke_skill_runner(catalog)
        result = await runner(InvokeSkillInput(name="nope"), None)
        out = InvokeSkillOutput.model_validate(result)
        assert out.success is False
        assert out.error_code == "skill_not_found"
        assert "not found" in out.loaded_content

    @pytest.mark.anyio
    async def test_runner_ambiguous_bare_name(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as ext:
            _write_skill(Path(d), "dup", "Project version")
            ext_skill = Path(ext) / "dup"
            ext_skill.mkdir()
            (ext_skill / "SKILL.md").write_text(
                "---\nname: dup\ndescription: External version\n---\nexternal"
            )
            catalog = SkillCatalog(
                scan_and_load_skills(Path(d), repo_root=Path(d), extra_dirs=[Path(ext)])
            )
            runner = make_invoke_skill_runner(catalog)

            result = await runner(InvokeSkillInput(name="dup"), None)
            out = InvokeSkillOutput.model_validate(result)

            assert out.success is False
            assert out.error_code == "ambiguous_skill_name"
            assert "project:dup" in out.loaded_content
            assert "external:dup" in out.loaded_content

    @pytest.mark.anyio
    async def test_runner_disabled_skill(self):
        with tempfile.TemporaryDirectory() as d:
            _write_skill(Path(d), "disabled", "Disabled skill")
            manifests = scan_and_load_skills(Path(d), repo_root=Path(d))
            catalog = SkillCatalog(manifests)
            policy = SkillPolicy(disabled_skills=frozenset({"disabled"}))
            runner = make_invoke_skill_runner(catalog, policy)

            result = await runner(InvokeSkillInput(name="disabled"), None)
            out = InvokeSkillOutput.model_validate(result)
            assert out.success is False
            assert "disabled by the current policy" in out.loaded_content

    @pytest.mark.anyio
    async def test_runner_disable_model_invocation(self):
        with tempfile.TemporaryDirectory() as d:
            _write_skill(
                Path(d), "no-model", "No model invoke",
                disable_model_invocation=True,
            )
            manifests = scan_and_load_skills(Path(d), repo_root=Path(d))
            catalog = SkillCatalog(manifests)
            runner = make_invoke_skill_runner(catalog)

            result = await runner(InvokeSkillInput(name="no-model"), None)
            out = InvokeSkillOutput.model_validate(result)
            assert out.success is False
            assert "disable_model_invocation" in out.loaded_content

    @pytest.mark.anyio
    async def test_runner_returns_structured_error_when_skill_file_becomes_invalid(self):
        with tempfile.TemporaryDirectory() as d:
            skill_md = _write_skill(Path(d), "invalid-on-load", "Invalid on load")
            catalog = SkillCatalog(scan_and_load_skills(Path(d), repo_root=Path(d)))
            runner = make_invoke_skill_runner(catalog)
            skill_md.write_text("# frontmatter removed\n")

            result = await runner(InvokeSkillInput(name="project:invalid-on-load"), None)
            out = InvokeSkillOutput.model_validate(result)

            assert out.success is False
            assert out.skill_id == "project:invalid-on-load"
            assert out.error_code == "invalid_skill_manifest"
            assert "Failed to load skill" in out.loaded_content

    @pytest.mark.anyio
    async def test_runner_args_substitution(self):
        with tempfile.TemporaryDirectory() as d:
            skill_dir = Path(d) / ".agents" / "skills" / "args-invoke"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                "---\nname: args-invoke\ndescription: Test args\n---\n"
                "Process: $ARGUMENTS"
            )
            manifests = scan_and_load_skills(Path(d), repo_root=Path(d))
            catalog = SkillCatalog(manifests)
            runner = make_invoke_skill_runner(catalog)

            result = await runner(
                InvokeSkillInput(name="args-invoke", args="input.csv"), None
            )
            out = InvokeSkillOutput.model_validate(result)
            assert out.success is True
            assert "input.csv" in out.loaded_content
            assert "$ARGUMENTS" not in out.loaded_content


class TestMaterializeSkillAsset:
    @pytest.mark.anyio
    async def test_materializes_active_skill_script_into_scratch(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as ws_dir:
            skill_md = _write_skill(Path(d), "asset-test", "Asset test")
            script = skill_md.parent / "scripts" / "helper.py"
            script.parent.mkdir()
            script.write_text("print('ok')\n")
            catalog = SkillCatalog(scan_and_load_skills(Path(d), repo_root=Path(d)))
            loaded = catalog.load("project:asset-test", iteration=2)
            assert loaded is not None

            run_config = _run_config()
            state = create_loop_state(task="test", run_config=run_config)
            state["skill_state"].active["project:asset-test"] = loaded.to_ref()
            workspace = open_workspace(ws_dir)
            tool = MaterializeSkillAssetTool(workspace)

            result = await tool.execute(
                MaterializeSkillAssetInput(
                    skill_id="project:asset-test",
                    relative_path="scripts/helper.py",
                ),
                ToolExecutionContext(run_config=run_config, state=state),
            )
            out = MaterializeSkillAssetOutput.model_validate(result)

            assert out.workspace_path == "scratch/skills/project_asset-test/scripts/helper.py"
            materialized = workspace.root / out.workspace_path
            assert materialized.read_text() == "print('ok')\n"
            assert out.size_bytes == len("print('ok')\n")

    @pytest.mark.anyio
    async def test_rejects_non_active_or_unsafe_skill_assets(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as ws_dir:
            skill_md = _write_skill(Path(d), "guard-test", "Guard test")
            ref_file = skill_md.parent / "references" / "notes.md"
            ref_file.parent.mkdir()
            ref_file.write_text("notes\n")
            catalog = SkillCatalog(scan_and_load_skills(Path(d), repo_root=Path(d)))
            loaded = catalog.load("project:guard-test", iteration=1)
            assert loaded is not None

            run_config = _run_config()
            state = create_loop_state(task="test", run_config=run_config)
            workspace = open_workspace(ws_dir)
            tool = MaterializeSkillAssetTool(workspace)
            context = ToolExecutionContext(run_config=run_config, state=state)

            with pytest.raises(ValueError, match="not active"):
                await tool.execute(
                    MaterializeSkillAssetInput(
                        skill_id="project:guard-test",
                        relative_path="references/notes.md",
                    ),
                    context,
                )

            state["skill_state"].active["project:guard-test"] = loaded.to_ref()

            with pytest.raises(ValueError, match="scripts/ or references/"):
                await tool.execute(
                    MaterializeSkillAssetInput(
                        skill_id="project:guard-test",
                        relative_path="SKILL.md",
                    ),
                    context,
                )

            with pytest.raises(ValueError, match="must be relative"):
                await tool.execute(
                    MaterializeSkillAssetInput(
                        skill_id="project:guard-test",
                        relative_path="../SKILL.md",
                    ),
                    context,
                )


# ── Models tests ─────────────────────────────────────────────────────


class TestModels:
    def test_skill_state_default(self):
        state = SkillState()
        assert state.visible_skill_ids == ()
        assert state.visible_skill_names == ()
        assert state.invoked == ()
        assert state.active == {}
        assert state.loaded_skills == {}

    def test_skill_invocation(self):
        inv = SkillInvocation(
            name="test",
            source="project",
            skill_file="/some/path/SKILL.md",
            fingerprint="abc123",
            invoked_at_iteration=3,
            args="hello",
        )
        assert inv.name == "test"
        assert inv.invoked_at_iteration == 3

    def test_skill_summary_render(self):
        s = SkillSummary(
            name="my-skill",
            description="Does things",
            when_to_use="when things need doing",
        )
        rendered = s.render()
        assert "- my-skill:" in rendered
        assert "when things need doing" in rendered
        assert "—" in rendered  # em-dash separator

    def test_skill_summary_render_no_when_to_use(self):
        s = SkillSummary(name="simple", description="Simple skill")
        rendered = s.render()
        assert rendered == "- simple: Simple skill"

    def test_skill_source_values(self):
        assert SkillSource.PROJECT == "project"
        assert SkillSource.USER == "user"
        assert SkillSource.BUNDLED == "bundled"

    def test_skill_manifest_fingerprint_stable(self):
        """Same content should produce the same fingerprint."""
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            p1 = _write_skill(Path(d1), "stable", "Same desc")
            p2 = _write_skill(Path(d2), "stable", "Same desc")
            m1 = load_skill_from_file(p1, SkillSource.PROJECT)
            m2 = load_skill_from_file(p2, SkillSource.PROJECT)
            # Fingerprints should match (same name + desc) even if path differs
            assert m1.content_fingerprint == m2.content_fingerprint


# ── Forward compatibility tests ───────────────────────────────────────


class TestForwardCompatibility:
    def test_license_field_accepted(self):
        """Claude Code skills use 'license' — must be stored in extra."""
        with tempfile.TemporaryDirectory() as d:
            skill_dir = Path(d) / ".agents" / "skills" / "licensed"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                "---\nname: licensed\ndescription: d\nlicense: MIT\n---\nbody"
            )
            manifests = scan_and_load_skills(Path(d), repo_root=Path(d))
            assert len(manifests) == 1
            assert manifests[0].extra.get("license") == "MIT"

    def test_any_unknown_field_stored(self):
        """Arbitrary unknown fields must never cause rejection."""
        with tempfile.TemporaryDirectory() as d:
            skill_dir = Path(d) / ".agents" / "skills" / "future"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                "---\nname: future\ndescription: d\n"
                "foo: bar\nbaz: 42\nnested:\n  x: 1\n---\nbody"
            )
            manifests = scan_and_load_skills(Path(d), repo_root=Path(d))
            assert len(manifests) == 1
            assert manifests[0].extra == {
                "foo": "bar",
                "baz": 42,
                "nested": {"x": 1},
            }

    def test_claude_xlsx_skill_format(self):
        """Simulate loading a skill with Claude Code's format."""
        with tempfile.TemporaryDirectory() as d:
            skill_dir = Path(d) / ".agents" / "skills" / "xlsx"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                '---\n'
                'name: xlsx\n'
                'description: "Use this skill for spreadsheets."\n'
                'license: Proprietary. LICENSE.txt has complete terms\n'
                '---\n'
                '# XLSX Skill\n\nBody content.\n'
            )
            manifests = scan_and_load_skills(Path(d), repo_root=Path(d))
            assert len(manifests) == 1
            m = manifests[0]
            assert m.name == "xlsx"
            assert m.skill_id == "project:xlsx"
            assert m.extra.get("license") is not None


class TestNamespace:
    def test_namespaced_directory(self):
        """Skills in skills/<ns>/<name>/SKILL.md get namespace:name."""
        with tempfile.TemporaryDirectory() as d:
            ns_dir = Path(d) / ".agents" / "skills" / "acme" / "pdf-tool"
            ns_dir.mkdir(parents=True)
            (ns_dir / "SKILL.md").write_text(
                "---\nname: pdf-tool\ndescription: PDF converter\n---\nbody"
            )
            manifests = scan_and_load_skills(Path(d), repo_root=Path(d))
            assert len(manifests) == 1
            assert manifests[0].name == "acme:pdf-tool"
            assert manifests[0].namespace == "acme"
            assert manifests[0].basename == "pdf-tool"

    def test_find_by_basename(self):
        """find() should match by basename when exact name not found."""
        with tempfile.TemporaryDirectory() as d:
            ns_dir = Path(d) / ".agents" / "skills" / "acme" / "review"
            ns_dir.mkdir(parents=True)
            (ns_dir / "SKILL.md").write_text(
                "---\nname: review\ndescription: Review tool\n---\nbody"
            )
            manifests = scan_and_load_skills(Path(d), repo_root=Path(d))
            catalog = SkillCatalog(manifests)

            # Exact match
            assert catalog.find("project:acme:review") is not None
            # Basename match
            found = catalog.find("review")
            assert found is not None
            assert found.name == "acme:review"
            assert catalog.load("review") is not None


class TestExternalSkills:
    def test_skill_path_env(self):
        """SKILL_PATH env var should be scanned."""
        import os

        with tempfile.TemporaryDirectory() as d, \
             tempfile.TemporaryDirectory() as ext_dir:
            ext_skill = Path(ext_dir) / "ext-skill"
            ext_skill.mkdir()
            (ext_skill / "SKILL.md").write_text(
                "---\nname: ext-skill\ndescription: External skill\n---\nbody"
            )

            old_val = os.environ.get("SKILL_PATH", "")
            os.environ["SKILL_PATH"] = ext_dir
            try:
                manifests = scan_and_load_skills(Path(d), repo_root=Path(d))
                names = {m.name for m in manifests}
                assert "ext-skill" in names
                ext_m = next(m for m in manifests if m.name == "ext-skill")
                assert ext_m.source == SkillSource.EXTERNAL
            finally:
                if old_val:
                    os.environ["SKILL_PATH"] = old_val
                else:
                    os.environ.pop("SKILL_PATH", None)

    def test_extra_dirs_parameter(self):
        """extra_dirs parameter should inject additional skill dirs."""
        with tempfile.TemporaryDirectory() as d, \
             tempfile.TemporaryDirectory() as extra:
            extra_skill = Path(extra) / "extra-skill"
            extra_skill.mkdir()
            (extra_skill / "SKILL.md").write_text(
                "---\nname: extra-skill\ndescription: Extra\n---\nbody"
            )

            manifests = scan_and_load_skills(
                Path(d),
                repo_root=Path(d),
                extra_dirs=[Path(extra)],
            )
            names = {m.name for m in manifests}
            assert "extra-skill" in names

    def test_external_source_policy(self):
        """Policy should control EXTERNAL skill visibility."""
        with tempfile.TemporaryDirectory() as d:
            skill_dir = Path(d) / "ext-skill"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text(
                "---\nname: ext-skill\ndescription: External\n---\nbody"
            )
            m = load_skill_from_file(
                skill_dir / "SKILL.md", SkillSource.EXTERNAL,
            )

            # Default policy trusts external
            policy = SkillPolicy()
            assert policy.can_autoload(m) is True

            # Disable trust
            strict = SkillPolicy(trust_external_skills=False)
            assert strict.can_autoload(m) is False


# ── Integration: SkillState in LoopState ─────────────────────────────


class TestSkillStateIntegration:
    def test_skill_state_in_loop_state(self):
        """SkillState should be present in a newly created LoopState."""
        from rag.agent.core.context import AgentRunConfig
        from rag.agent.loop.state import create_loop_state
        from rag.schema.runtime import AccessPolicy

        run_config = AgentRunConfig(
            run_id="r1",
            thread_id="t1",
            agent_type="generic",
            max_depth=2,
            access_policy=AccessPolicy(),
        )
        state = create_loop_state(task="test", run_config=run_config)
        assert "skill_state" in state
        skill_state = state["skill_state"]
        assert isinstance(skill_state, SkillState)
        assert skill_state.visible_skill_ids == ()
        assert skill_state.visible_skill_names == ()
        assert skill_state.active == {}

    def test_checkpoint_serde_restores_skill_state(self):
        from rag.agent.core.checkpointing import agent_checkpoint_serde

        with tempfile.TemporaryDirectory() as d:
            _write_skill(Path(d), "checkpoint-test", "Checkpoint test")
            catalog = SkillCatalog(scan_and_load_skills(Path(d), repo_root=Path(d)))
            loaded = catalog.load("project:checkpoint-test", iteration=1)
            state = create_loop_state(task="test", run_config=_run_config())
            state["skill_state"].active["project:checkpoint-test"] = loaded.to_ref()

            serde = agent_checkpoint_serde()
            restored = serde.loads_typed(serde.dumps_typed(state))

            assert isinstance(restored["skill_state"], SkillState)
            assert "project:checkpoint-test" in restored["skill_state"].active


class TestServiceSkillIntegration:
    def test_agent_service_registers_skill_tools_when_catalog_present(self):
        from rag.agent.builtin.generic import GENERIC_AGENT
        from rag.agent.builtin_registry import create_builtin_tool_registry
        from rag.agent.tools.workspace_tools import create_workspace_tools

        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as ws_dir:
            _write_skill(Path(d), "service-skill", "Service skill")
            catalog = SkillCatalog(scan_and_load_skills(Path(d), repo_root=Path(d)))
            service = AgentService(
                definition=GENERIC_AGENT,
                tool_registry=create_builtin_tool_registry(),
                skill_catalog=catalog,
            )
            workspace = open_workspace(ws_dir)

            assert "invoke_skill" in service._policy.core_tool_names
            assert "materialize_skill_asset" in service._policy.core_tool_names

            registry = service._runtime_tool_registry(
                _run_config(),
                tools=create_workspace_tools(workspace),
            )

            assert registry.has_runner("invoke_skill")
            assert registry.has_runner("materialize_skill_asset")

from __future__ import annotations

import fnmatch
import os
import re
from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from rag.agent.tools.tool import (
    CancellationMode,
    InterruptBehavior,
    JsonValue,
    NormalizedToolOutput,
    ResolvedToolUse,
    Tool,
    ToolDefinition,
    ToolEffect,
    ToolTarget,
    ToolValidationError,
    json_schema_output,
    pydantic_input,
)
from rag.agent.workspace import WorkspaceRuntime

_INTERNAL_DIRECTORY = ".agent_memory"
_MAX_SEARCH_FILE_BYTES = 2_000_000
_MAX_RETURNED_LINE_CHARS = 500


class SearchTextInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pattern: str = Field(
        min_length=1,
        max_length=2000,
        description="Literal text or regular expression to find.",
    )
    path: str = Field(
        default=".",
        max_length=4096,
        description="Workspace-relative file or directory to search.",
    )
    glob: str | None = Field(
        default=None,
        max_length=512,
        description="Optional file glob such as '*.py' or 'src/**/*.ts'.",
    )
    regex: bool = Field(
        default=False,
        description="Interpret pattern as a Python regular expression.",
    )
    context_lines: int = Field(
        default=0,
        ge=0,
        le=5,
        description="Lines of context to return before and after each match.",
    )
    max_results: int = Field(
        default=40,
        ge=1,
        le=200,
        description="Maximum matching lines to return.",
    )


class SearchTextMatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file_path: str
    line_number: int = Field(ge=1)
    line_content: str
    match_start: int = Field(ge=0)
    match_end: int = Field(ge=0)
    context_before: list[str] = Field(default_factory=list)
    context_after: list[str] = Field(default_factory=list)


class SearchTextOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    matches: list[SearchTextMatch]
    total_matches: int = Field(ge=0)
    truncated: bool = False


_SEARCH_INPUT_SCHEMA, _validate_search_model = pydantic_input(SearchTextInput)
_SEARCH_OUTPUT_SCHEMA, _unused_search_output_validator = pydantic_input(
    SearchTextOutput
)


def create_search_text_tool(workspace: WorkspaceRuntime) -> Tool:
    return Tool(
        definition=ToolDefinition(
            name="search_text",
            description=(
                "Search current workspace files without an index. Supports literal or "
                "regex matching, one file or directory path, an optional glob, bounded "
                "context lines, and a result limit. Symlinks are not followed, so every "
                "call observes only current in-workspace file contents."
            ),
            input_schema=_SEARCH_INPUT_SCHEMA,
        ),
        validate_input=_validate_search_input,
        run=lambda arguments: _search_text(
            workspace,
            SearchTextInput.model_validate(arguments),
        ),
        normalize_output=_normalize_search_output,
        output_schema=_SEARCH_OUTPUT_SCHEMA,
        static_effects=frozenset({ToolEffect.READ_WORKSPACE}),
        resolve_use=lambda arguments: ResolvedToolUse(
            effects=frozenset({ToolEffect.READ_WORKSPACE}),
            targets=(
                ToolTarget(
                    kind="workspace_path",
                    value=str(
                        workspace.resolve_path(str(arguments["path"]) or ".").resolve()
                    ),
                ),
            ),
        ),
        execution_revision="builtin-search-text-v1",
        idempotent=True,
        concurrency_safe=True,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=15.0,
        max_model_output_bytes=2_000_000,
    )


def _validate_search_input(
    arguments: Mapping[str, JsonValue],
) -> Mapping[str, JsonValue]:
    canonical = _validate_search_model(arguments)
    if canonical["regex"]:
        try:
            re.compile(str(canonical["pattern"]))
        except re.error:
            raise ToolValidationError(
                path="$.pattern",
                message="invalid regular expression",
            ) from None
    return canonical


def _search_text(
    workspace: WorkspaceRuntime,
    request: SearchTextInput,
) -> SearchTextOutput:
    files = _searchable_files(workspace, request.path, request.glob)
    expression = re.compile(request.pattern) if request.regex else None
    matches: list[SearchTextMatch] = []
    truncated = False

    for path in files:
        raw = path.read_bytes()[: _MAX_SEARCH_FILE_BYTES + 1]
        if b"\x00" in raw:
            continue
        if len(raw) > _MAX_SEARCH_FILE_BYTES:
            raw = raw[:_MAX_SEARCH_FILE_BYTES]
            truncated = True
        lines = raw.decode("utf-8", errors="replace").splitlines()
        relative = path.relative_to(workspace.root).as_posix()
        for index, line in enumerate(lines):
            found = expression.search(line) if expression is not None else None
            if expression is None:
                start = line.find(request.pattern)
                if start < 0:
                    continue
                end = start + len(request.pattern)
            else:
                if found is None:
                    continue
                start, end = found.span()
            if len(matches) >= request.max_results:
                return SearchTextOutput(
                    matches=matches,
                    total_matches=len(matches),
                    truncated=True,
                )
            before_start = max(0, index - request.context_lines)
            after_end = min(len(lines), index + request.context_lines + 1)
            matches.append(
                SearchTextMatch(
                    file_path=relative,
                    line_number=index + 1,
                    line_content=_bounded_line(line),
                    match_start=start,
                    match_end=end,
                    context_before=[
                        _bounded_line(value) for value in lines[before_start:index]
                    ],
                    context_after=[
                        _bounded_line(value) for value in lines[index + 1 : after_end]
                    ],
                )
            )
    return SearchTextOutput(
        matches=matches,
        total_matches=len(matches),
        truncated=truncated,
    )


def _searchable_files(
    workspace: WorkspaceRuntime,
    value: str,
    glob: str | None,
) -> tuple[Path, ...]:
    lexical = workspace.resolve_path(value or ".")
    target = workspace.ensure_within_workspace(lexical)
    relative_target = target.relative_to(workspace.root.resolve())
    if relative_target.parts and relative_target.parts[0] == _INTERNAL_DIRECTORY:
        raise PermissionError("agent memory is not searchable")
    if lexical.is_symlink():
        return ()
    if target.is_file():
        return (target,) if _matches_glob(workspace, target, glob) else ()
    if not target.is_dir():
        return ()

    files: list[Path] = []
    for directory, directory_names, file_names in os.walk(
        target,
        followlinks=False,
    ):
        current = Path(directory)
        directory_names[:] = sorted(
            name
            for name in directory_names
            if name != _INTERNAL_DIRECTORY
            and not (current / name).is_symlink()
        )
        for name in sorted(file_names):
            path = current / name
            if path.is_symlink() or not path.is_file():
                continue
            if _matches_glob(workspace, path, glob):
                files.append(path)
    return tuple(files)


def _matches_glob(
    workspace: WorkspaceRuntime,
    path: Path,
    glob: str | None,
) -> bool:
    if glob is None:
        return True
    relative = path.relative_to(workspace.root).as_posix()
    return fnmatch.fnmatch(path.name, glob) or fnmatch.fnmatch(relative, glob)


def _bounded_line(value: str) -> str:
    return value[:_MAX_RETURNED_LINE_CHARS]


def _normalize_search_output(raw: object) -> NormalizedToolOutput:
    validated = SearchTextOutput.model_validate(raw)
    structured = json_schema_output(
        _SEARCH_OUTPUT_SCHEMA,
        validated.model_dump(mode="json"),
    )
    return NormalizedToolOutput(structured_content=structured)


__all__ = [
    "SearchTextInput",
    "SearchTextMatch",
    "SearchTextOutput",
    "create_search_text_tool",
]

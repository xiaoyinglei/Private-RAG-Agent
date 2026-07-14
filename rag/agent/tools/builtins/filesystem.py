from __future__ import annotations

import fnmatch
import os
import stat
import tempfile
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
    json_schema_output,
    pydantic_input,
)
from rag.agent.workspace import WorkspaceRuntime

_INTERNAL_DIRECTORY = ".agent_memory"


class ListFilesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(
        default=".",
        description="Workspace-relative directory to list.",
        max_length=4096,
    )
    glob: str | None = Field(
        default=None,
        description="Optional filename glob such as '*.py'.",
        max_length=512,
    )
    limit: int = Field(
        default=200,
        ge=1,
        le=200,
        description="Maximum number of sorted entries to return.",
    )


class FileEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    name: str
    size_bytes: int = Field(ge=0)
    is_directory: bool
    is_symlink: bool


class ListFilesOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entries: list[FileEntry]
    truncated: bool = False


class ReadFileInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(
        min_length=1,
        max_length=4096,
        description="Workspace-relative file path.",
    )
    encoding: str = Field(
        default="utf-8",
        min_length=1,
        max_length=64,
        description="Text encoding used when decoding non-binary bytes.",
    )
    offset: int = Field(default=0, ge=0, description="Byte offset to start reading.")
    max_bytes: int = Field(
        default=16_000,
        ge=1,
        le=1_000_000,
        description=(
            "Optional byte limit from 1 through 1,000,000; omit it to use "
            "the context-safe 16,000-byte default, then advance offset for "
            "another chunk."
        ),
    )


class ReadFileOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    content: str
    size_bytes: int = Field(ge=0)
    offset: int = Field(ge=0)
    truncated: bool
    is_binary: bool
    encoding: str


class ApplyPatchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file_path: str = Field(
        min_length=1,
        max_length=4096,
        description="Workspace-relative path of the existing file to edit.",
    )
    old_string: str = Field(
        min_length=1,
        max_length=1_000_000,
        description="Exact text that must already exist.",
    )
    new_string: str = Field(
        max_length=1_000_000,
        description="Replacement text; an empty value deletes the old text.",
    )
    replace_all: bool = Field(
        default=False,
        description="Replace every occurrence instead of requiring uniqueness.",
    )


class ApplyPatchOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file_path: str
    replaced: bool
    occurrences: int = Field(ge=0)
    message: str


_LIST_INPUT_SCHEMA, _validate_list_input = pydantic_input(ListFilesInput)
_LIST_OUTPUT_SCHEMA, _unused_list_output_validator = pydantic_input(ListFilesOutput)
_READ_INPUT_SCHEMA, _validate_read_input = pydantic_input(ReadFileInput)
_READ_OUTPUT_SCHEMA, _unused_read_output_validator = pydantic_input(ReadFileOutput)
_PATCH_INPUT_SCHEMA, _validate_patch_input = pydantic_input(ApplyPatchInput)
_PATCH_OUTPUT_SCHEMA, _unused_patch_output_validator = pydantic_input(ApplyPatchOutput)
_PATCH_ERROR_CODES = {
    "file not found": "file_not_found",
    "old_string not found": "old_string_not_found",
    "old_string is not unique; set replace_all=true": "old_string_not_unique",
}


def create_list_files_tool(workspace: WorkspaceRuntime) -> Tool:
    return Tool(
        definition=ToolDefinition(
            name="list_files",
            description=(
                "List one workspace directory in deterministic filename order. "
                "Use glob to narrow entries and limit to bound the result. This tool "
                "does not recurse; call it again for a returned directory."
            ),
            input_schema=_LIST_INPUT_SCHEMA,
        ),
        validate_input=_validate_list_input,
        run=lambda arguments: _list_files(
            workspace,
            ListFilesInput.model_validate(arguments),
        ),
        normalize_output=lambda raw: _normalize_model(
            raw,
            model=ListFilesOutput,
            schema=_LIST_OUTPUT_SCHEMA,
        ),
        output_schema=_LIST_OUTPUT_SCHEMA,
        static_effects=frozenset({ToolEffect.READ_WORKSPACE}),
        resolve_use=lambda arguments: _workspace_use(
            workspace,
            str(arguments["path"]),
            effects=frozenset({ToolEffect.READ_WORKSPACE}),
        ),
        execution_revision="builtin-list-files-v1",
        idempotent=True,
        concurrency_safe=True,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=5.0,
        max_model_output_bytes=300_000,
    )


def create_read_file_tool(workspace: WorkspaceRuntime) -> Tool:
    return Tool(
        definition=ToolDefinition(
            name="read_file",
            description=(
                "Read a bounded byte range from one workspace file. Binary files are "
                "reported without embedding their bytes. The 16,000-byte default "
                "bounds model context; use offset for later chunks, and never set "
                "max_bytes above 1,000,000."
            ),
            input_schema=_READ_INPUT_SCHEMA,
        ),
        validate_input=_validate_read_input,
        run=lambda arguments: _read_file(
            workspace,
            ReadFileInput.model_validate(arguments),
        ),
        normalize_output=lambda raw: _normalize_model(
            raw,
            model=ReadFileOutput,
            schema=_READ_OUTPUT_SCHEMA,
        ),
        output_schema=_READ_OUTPUT_SCHEMA,
        static_effects=frozenset({ToolEffect.READ_WORKSPACE}),
        resolve_use=lambda arguments: _workspace_use(
            workspace,
            str(arguments["path"]),
            effects=frozenset({ToolEffect.READ_WORKSPACE}),
        ),
        execution_revision="builtin-read-file-v1",
        idempotent=True,
        concurrency_safe=True,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=5.0,
        max_model_output_bytes=1_100_000,
    )


def create_apply_patch_tool(workspace: WorkspaceRuntime) -> Tool:
    return Tool(
        definition=ToolDefinition(
            name="apply_patch",
            description=(
                "Edit an existing UTF-8 workspace file by exact text replacement. "
                "Without replace_all, the old text must occur exactly once. The write "
                "is atomically installed and never creates a second write_file tool."
            ),
            input_schema=_PATCH_INPUT_SCHEMA,
        ),
        validate_input=_validate_patch_input,
        run=lambda arguments: _apply_patch(
            workspace,
            ApplyPatchInput.model_validate(arguments),
        ),
        normalize_output=_normalize_apply_patch,
        output_schema=_PATCH_OUTPUT_SCHEMA,
        static_effects=frozenset({
            ToolEffect.READ_WORKSPACE,
            ToolEffect.WRITE_WORKSPACE,
        }),
        resolve_use=lambda arguments: _workspace_use(
            workspace,
            str(arguments["file_path"]),
            effects=frozenset({
                ToolEffect.READ_WORKSPACE,
                ToolEffect.WRITE_WORKSPACE,
            }),
        ),
        execution_revision="builtin-apply-patch-v1",
        idempotent=True,
        concurrency_safe=True,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.FINISH_CURRENT,
        timeout_seconds=5.0,
        max_model_output_bytes=50_000,
    )


def _list_files(
    workspace: WorkspaceRuntime,
    request: ListFilesInput,
) -> ListFilesOutput:
    directory = _checked_path(workspace, request.path)
    if not directory.is_dir():
        raise NotADirectoryError(f"workspace directory not found: {request.path}")

    entries: list[FileEntry] = []
    truncated = False
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        if path.name == _INTERNAL_DIRECTORY:
            continue
        relative = path.relative_to(workspace.root).as_posix()
        if request.glob and not (
            fnmatch.fnmatch(path.name, request.glob)
            or fnmatch.fnmatch(relative, request.glob)
        ):
            continue
        if len(entries) >= request.limit:
            truncated = True
            break
        metadata = path.lstat()
        is_symlink = path.is_symlink()
        entries.append(
            FileEntry(
                path=relative,
                name=path.name,
                size_bytes=metadata.st_size,
                is_directory=not is_symlink and stat.S_ISDIR(metadata.st_mode),
                is_symlink=is_symlink,
            )
        )
    return ListFilesOutput(entries=entries, truncated=truncated)


def _read_file(
    workspace: WorkspaceRuntime,
    request: ReadFileInput,
) -> ReadFileOutput:
    target = _checked_path(workspace, request.path)
    if not target.is_file():
        raise FileNotFoundError(f"workspace file not found: {request.path}")

    size = target.stat().st_size
    with target.open("rb") as stream:
        stream.seek(request.offset)
        raw = stream.read(request.max_bytes + 1)
    truncated = len(raw) > request.max_bytes
    bounded = raw[: request.max_bytes]
    is_binary = b"\x00" in bounded
    content = "" if is_binary else bounded.decode(request.encoding, errors="replace")
    return ReadFileOutput(
        path=request.path,
        content=content,
        size_bytes=size,
        offset=request.offset,
        truncated=truncated,
        is_binary=is_binary,
        encoding=request.encoding,
    )


def _apply_patch(
    workspace: WorkspaceRuntime,
    request: ApplyPatchInput,
) -> ApplyPatchOutput:
    target = _checked_path(workspace, request.file_path)
    if not target.is_file():
        return ApplyPatchOutput(
            file_path=request.file_path,
            replaced=False,
            occurrences=0,
            message="file not found",
        )

    current = target.read_text(encoding="utf-8")
    occurrences = current.count(request.old_string)
    if occurrences == 0:
        return ApplyPatchOutput(
            file_path=request.file_path,
            replaced=False,
            occurrences=0,
            message="old_string not found",
        )
    if occurrences > 1 and not request.replace_all:
        return ApplyPatchOutput(
            file_path=request.file_path,
            replaced=False,
            occurrences=occurrences,
            message="old_string is not unique; set replace_all=true",
        )

    updated = (
        current.replace(request.old_string, request.new_string)
        if request.replace_all
        else current.replace(request.old_string, request.new_string, 1)
    )
    _atomic_write_text(target, updated)
    return ApplyPatchOutput(
        file_path=request.file_path,
        replaced=True,
        occurrences=occurrences if request.replace_all else 1,
        message="patch applied",
    )


def _atomic_write_text(path: Path, content: str) -> None:
    mode = stat.S_IMODE(path.stat().st_mode)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_name = stream.name
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary_name, mode)
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _workspace_use(
    workspace: WorkspaceRuntime,
    value: str,
    *,
    effects: frozenset[ToolEffect],
) -> ResolvedToolUse:
    target = workspace.resolve_path(value or ".").resolve()
    return ResolvedToolUse(
        effects=effects,
        targets=(ToolTarget(kind="workspace_path", value=str(target)),),
    )


def _checked_path(workspace: WorkspaceRuntime, value: str) -> Path:
    target = workspace.ensure_within_workspace(
        workspace.resolve_path(value or "."),
    )
    relative = target.relative_to(workspace.root.resolve())
    if relative.parts and relative.parts[0] == _INTERNAL_DIRECTORY:
        raise PermissionError("agent memory is not exposed as a workspace file")
    return target


def _normalize_model(
    raw: object,
    *,
    model: type[BaseModel],
    schema: Mapping[str, JsonValue],
) -> NormalizedToolOutput:
    validated = model.model_validate(raw)
    structured = json_schema_output(
        schema,
        validated.model_dump(mode="json"),
    )
    return NormalizedToolOutput(structured_content=structured)


def _normalize_apply_patch(raw: object) -> NormalizedToolOutput:
    validated = ApplyPatchOutput.model_validate(raw)
    structured = json_schema_output(
        _PATCH_OUTPUT_SCHEMA,
        validated.model_dump(mode="json"),
    )
    if validated.replaced:
        return NormalizedToolOutput(structured_content=structured)
    return NormalizedToolOutput(
        structured_content=structured,
        is_error=True,
        error_code=_PATCH_ERROR_CODES.get(
            validated.message,
            "patch_not_applied",
        ),
        error_message=validated.message,
        retryable=True,
    )


__all__ = [
    "ApplyPatchInput",
    "ApplyPatchOutput",
    "FileEntry",
    "ListFilesInput",
    "ListFilesOutput",
    "ReadFileInput",
    "ReadFileOutput",
    "create_apply_patch_tool",
    "create_list_files_tool",
    "create_read_file_tool",
]

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from inspect import isawaitable
from typing import TYPE_CHECKING, Protocol, cast

from pydantic import BaseModel, ValidationError

from rag.agent.tools.formatter import ToolOutputFormatter
from rag.agent.tools.spec import ToolSpec

if TYPE_CHECKING:
    from rag.agent.core.context import AgentRunConfig
    from rag.agent.core.definition import AgentDefinition
    from rag.agent.loop.state import LoopState


class ToolProgressCallback(Protocol):
    def __call__(
        self,
        progress: str,
        percent: float | None = None,
    ) -> Awaitable[None]: ...


@dataclass(frozen=True)
class ToolExecutionContext:
    run_config: AgentRunConfig
    operation_id: str | None = None
    tool_call_id: str | None = None
    state: LoopState | None = None
    definition: AgentDefinition | None = None
    # 进度回调：工具可以调用它来报告执行进度
    # callback(tool_call_id, progress_text, percent)
    progress_callback: ToolProgressCallback | None = None


ToolRunnerResult = BaseModel | dict[str, object]
ToolRunner = Callable[[BaseModel], ToolRunnerResult | Awaitable[ToolRunnerResult]]
ContextualToolRunner = Callable[
    [BaseModel, ToolExecutionContext],
    ToolRunnerResult | Awaitable[ToolRunnerResult],
]


class ToolRunnerMissingError(LookupError):
    pass


class ToolExecutionContextMissingError(RuntimeError):
    pass


class ToolInputValidationError(ValueError):
    def __init__(self, tool_name: str, validation_error: ValidationError) -> None:
        super().__init__(f"{tool_name} input validation failed")
        self.tool_name = tool_name
        self.validation_error = validation_error

    def errors(self) -> list[dict[str, object]]:
        return [cast(dict[str, object], dict(error)) for error in self.validation_error.errors()]


class ToolOutputValidationError(ValueError):
    def __init__(self, tool_name: str, validation_error: ValidationError) -> None:
        super().__init__(f"{tool_name} output validation failed")
        self.tool_name = tool_name
        self.validation_error = validation_error

    def errors(self) -> list[dict[str, object]]:
        return [cast(dict[str, object], dict(error)) for error in self.validation_error.errors()]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self._runners: dict[str, ToolRunner] = {}
        self._contextual_runners: dict[str, ContextualToolRunner] = {}
        self._formatters: dict[str, ToolOutputFormatter] = {}

    def register(self, spec: ToolSpec, *, runner: ToolRunner | None = None) -> None:
        self._tools[spec.name] = spec
        if runner is None:
            self._runners.pop(spec.name, None)
            self._contextual_runners.pop(spec.name, None)
        else:
            self._runners[spec.name] = runner
            self._contextual_runners.pop(spec.name, None)

    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found in registry")
        return self._tools[name]

    def register_runner(self, tool_name: str, runner: ToolRunner) -> None:
        self.get(tool_name)
        self._runners[tool_name] = runner
        self._contextual_runners.pop(tool_name, None)

    def register_contextual_runner(
        self,
        tool_name: str,
        runner: ContextualToolRunner,
    ) -> None:
        self.get(tool_name)
        self._contextual_runners[tool_name] = runner
        self._runners.pop(tool_name, None)

    def register_formatter(self, formatter: ToolOutputFormatter) -> None:
        """Register a per-tool output formatter for ContextBuilder."""
        self._formatters[formatter.tool_name] = formatter

    def get_formatter(self, tool_name: str) -> ToolOutputFormatter | None:
        """Get the formatter for a tool, or None."""
        return self._formatters.get(tool_name)

    def has_runner(self, tool_name: str) -> bool:
        return tool_name in self._runners or tool_name in self._contextual_runners

    async def run(
        self,
        tool_name: str,
        arguments: dict[str, object],
        *,
        execution_context: ToolExecutionContext | None = None,
    ) -> BaseModel:
        spec = self.get(tool_name)
        try:
            input_payload = spec.input_model.model_validate(arguments)
        except ValidationError as exc:
            raise ToolInputValidationError(tool_name, exc) from exc

        contextual_runner = self._contextual_runners.get(tool_name)
        runner = self._runners.get(tool_name)
        if contextual_runner is None and runner is None:
            raise ToolRunnerMissingError(f"{tool_name} has no registered callable runner")

        if contextual_runner is not None:
            if execution_context is None:
                raise ToolExecutionContextMissingError(f"{tool_name} requires a trusted execution context")
            raw_output = contextual_runner(input_payload, execution_context)
        else:
            if runner is None:
                raise ToolRunnerMissingError(f"{tool_name} has no registered callable runner")
            raw_output = runner(input_payload)
        if isawaitable(raw_output):
            raw_output = await raw_output

        try:
            return spec.output_model.model_validate(raw_output)
        except ValidationError as exc:
            raise ToolOutputValidationError(tool_name, exc) from exc

    def clone(self) -> ToolRegistry:
        """创建浅拷贝，用于 request-scoped runner 注入。

        返回的新 registry 共享 ToolSpec，但有独立的 runners dict，
        因此注入 AgentAsToolAdapter 不会污染原始 registry。
        """
        cloned = ToolRegistry()
        cloned._tools = dict(self._tools)
        cloned._runners = dict(self._runners)
        cloned._contextual_runners = dict(self._contextual_runners)
        cloned._formatters = dict(self._formatters)
        return cloned

    def list_all(self) -> list[ToolSpec]:
        return list(self._tools.values())

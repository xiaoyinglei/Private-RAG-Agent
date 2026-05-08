from __future__ import annotations

from collections.abc import Awaitable, Callable
from inspect import isawaitable

from pydantic import BaseModel, ValidationError

from rag.agent.tools.spec import ToolSpec


ToolRunnerResult = BaseModel | dict[str, object]
ToolRunner = Callable[[BaseModel], ToolRunnerResult | Awaitable[ToolRunnerResult]]


class ToolRunnerMissingError(LookupError):
    pass


class ToolInputValidationError(ValueError):
    def __init__(self, tool_name: str, validation_error: ValidationError) -> None:
        super().__init__(f"{tool_name} input validation failed")
        self.tool_name = tool_name
        self.validation_error = validation_error

    def errors(self) -> list[dict[str, object]]:
        return self.validation_error.errors()


class ToolOutputValidationError(ValueError):
    def __init__(self, tool_name: str, validation_error: ValidationError) -> None:
        super().__init__(f"{tool_name} output validation failed")
        self.tool_name = tool_name
        self.validation_error = validation_error

    def errors(self) -> list[dict[str, object]]:
        return self.validation_error.errors()


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self._runners: dict[str, ToolRunner] = {}

    def register(self, spec: ToolSpec, *, runner: ToolRunner | None = None) -> None:
        self._tools[spec.name] = spec
        if runner is None:
            self._runners.pop(spec.name, None)
        else:
            self._runners[spec.name] = runner

    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found in registry")
        return self._tools[name]

    def register_runner(self, tool_name: str, runner: ToolRunner) -> None:
        self.get(tool_name)
        self._runners[tool_name] = runner

    def has_runner(self, tool_name: str) -> bool:
        return tool_name in self._runners

    async def run(self, tool_name: str, arguments: dict[str, object]) -> BaseModel:
        spec = self.get(tool_name)
        try:
            input_payload = spec.input_model.model_validate(arguments)
        except ValidationError as exc:
            raise ToolInputValidationError(tool_name, exc) from exc

        runner = self._runners.get(tool_name)
        if runner is None:
            raise ToolRunnerMissingError(f"{tool_name} has no registered callable runner")

        raw_output = runner(input_payload)
        if isawaitable(raw_output):
            raw_output = await raw_output

        try:
            return spec.output_model.model_validate(raw_output)
        except ValidationError as exc:
            raise ToolOutputValidationError(tool_name, exc) from exc

    def list_all(self) -> list[ToolSpec]:
        return list(self._tools.values())

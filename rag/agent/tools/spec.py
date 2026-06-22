from __future__ import annotations

import importlib
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

from pydantic import (
    BaseModel,
    Field,
    SerializationInfo,
    field_serializer,
    field_validator,
    model_validator,
)

# ============================================================
# 行为枚举 —— 工具自描述的核心
# ============================================================

class ExecutionCategory(Enum):
    """工具的执行类别，影响权限决策和并行策略。"""
    READ = "read"           # 只读，可并行，自动 allow
    TRANSFORM = "transform" # 数据变换，可并行，自动 allow
    WRITE = "write"         # 写入，串行，需要 ASK
    MUTATE = "mutate"       # 不可逆变更，串行，需要 ASK
    EXECUTE = "execute"     # 代码执行，沙箱内自动 allow
    NETWORK = "network"     # 外部网络，需要 ASK
    SYSTEM = "system"       # 系统操作，需要 ASK


class RiskLevel(Enum):
    """工具的风险等级，影响 UI 展示和审计。"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class InterruptBehavior(Enum):
    """用户中断时工具应该怎么处理。"""
    CANCEL = "cancel"       # 立即取消（适合只读操作）
    BLOCK = "block"         # 等当前操作完成（适合写入操作）


# 哨兵值：表示 execution_category / risk_level 需要从 permissions 推导
_INFER: Any = object()

_AUTO_ALLOW_CATEGORIES = frozenset({
    ExecutionCategory.READ,
    ExecutionCategory.TRANSFORM,
})
_RISK_ORDER = {
    RiskLevel.LOW: 0,
    RiskLevel.MEDIUM: 1,
    RiskLevel.HIGH: 2,
}


@dataclass(frozen=True)
class ToolPermissions:
    read_db: bool = False
    write_db: bool = False
    read_object_store: bool = False
    embed: bool = False
    generate: bool = False
    external_network: bool = False
    kg_mutation: bool = False
    user_data: bool = False
    read_fs: bool = False
    write_fs: bool = False
    execute_code: bool = False


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    error_model: type[BaseModel]
    permissions: ToolPermissions
    timeout_seconds: float
    max_retries: int = 0
    idempotent: bool = False
    concurrency_safe: bool = False
    is_read_only: bool = False
    work_budget_cost: int = 0
    requires_confirmation: bool = False
    audit_log: bool = False

    # ---- 行为声明（新增）----
    execution_category: ExecutionCategory = _INFER
    risk_level: RiskLevel = _INFER
    interrupt_behavior: InterruptBehavior = InterruptBehavior.CANCEL
    max_result_size_chars: int = 64000
    prompt_template: str = ""

    def __post_init__(self) -> None:
        if self.execution_category is _INFER:
            category = self._infer_category()
        else:
            category = _coerce_execution_category(self.execution_category)
        object.__setattr__(self, "execution_category", category)

        object.__setattr__(
            self,
            "interrupt_behavior",
            _coerce_interrupt_behavior(self.interrupt_behavior),
        )
        self._validate_category_permissions(category)
        computed_read_only = (
            category in _AUTO_ALLOW_CATEGORIES
            and not self.permissions_require_approval
        )
        if self.is_read_only and not computed_read_only:
            raise ValueError(
                "is_read_only=True is incompatible with permissions/category "
                "that require approval"
            )
        object.__setattr__(self, "is_read_only", computed_read_only)

        minimum_risk = self._minimum_risk_level(category)
        if self.risk_level is _INFER:
            risk_level = minimum_risk
        else:
            risk_level = _coerce_risk_level(self.risk_level)
            if _RISK_ORDER[risk_level] < _RISK_ORDER[minimum_risk]:
                raise ValueError(
                    "risk_level cannot be lower than permissions/category risk: "
                    f"{risk_level.value} < {minimum_risk.value}"
                )
        object.__setattr__(self, "risk_level", risk_level)

    def _infer_category(self) -> ExecutionCategory:
        p = self.permissions
        if p.execute_code:
            return ExecutionCategory.EXECUTE
        if p.kg_mutation or p.user_data:
            return ExecutionCategory.MUTATE
        if p.write_db or p.write_fs:
            return ExecutionCategory.WRITE
        if p.external_network:
            return ExecutionCategory.NETWORK
        if p.generate:
            return ExecutionCategory.TRANSFORM
        return ExecutionCategory.READ

    def _infer_risk_level(self) -> RiskLevel:
        return self._minimum_risk_level(self.execution_category)

    def _minimum_risk_level(self, category: ExecutionCategory) -> RiskLevel:
        levels = [_category_risk_level(category)]
        p = self.permissions
        if p.kg_mutation or p.user_data:
            levels.append(RiskLevel.HIGH)
        elif p.write_db or p.write_fs or p.execute_code or p.external_network:
            levels.append(RiskLevel.MEDIUM)
        return max(levels, key=lambda risk: _RISK_ORDER[risk])

    def _validate_category_permissions(self, category: ExecutionCategory) -> None:
        if category in _AUTO_ALLOW_CATEGORIES and self.permissions_require_approval:
            raise ValueError(
                "execution_category cannot be read/transform for permissions "
                "that require approval"
            )

    @property
    def permissions_require_approval(self) -> bool:
        return _permissions_require_approval(self.permissions)

    @property
    def minimum_risk_level(self) -> RiskLevel:
        category = (
            self.execution_category
            if isinstance(self.execution_category, ExecutionCategory)
            else self._infer_category()
        )
        return self._minimum_risk_level(category)

    @property
    def is_destructive(self) -> bool:
        return self.execution_category == ExecutionCategory.MUTATE or (
            self.permissions.write_db
            or self.permissions.kg_mutation
            or self.permissions.user_data
        )

    # ---- 工厂方法 ----

    @classmethod
    def read_only(cls, **kwargs: object) -> ToolSpec:
        defaults: dict[str, object] = {
            "execution_category": ExecutionCategory.READ,
            "risk_level": RiskLevel.LOW,
            "interrupt_behavior": InterruptBehavior.CANCEL,
            "concurrency_safe": True,
            "requires_confirmation": False,
        }
        defaults.update(kwargs)
        return cls(**defaults)  # type: ignore[arg-type]

    @classmethod
    def write_tool(cls, **kwargs: object) -> ToolSpec:
        defaults: dict[str, object] = {
            "execution_category": ExecutionCategory.WRITE,
            "risk_level": RiskLevel.MEDIUM,
            "interrupt_behavior": InterruptBehavior.BLOCK,
            "concurrency_safe": False,
            "requires_confirmation": True,
        }
        defaults.update(kwargs)
        return cls(**defaults)  # type: ignore[arg-type]

    @classmethod
    def destructive(cls, **kwargs: object) -> ToolSpec:
        defaults: dict[str, object] = {
            "execution_category": ExecutionCategory.MUTATE,
            "risk_level": RiskLevel.HIGH,
            "interrupt_behavior": InterruptBehavior.BLOCK,
            "concurrency_safe": False,
            "requires_confirmation": True,
            "audit_log": True,
        }
        defaults.update(kwargs)
        return cls(**defaults)  # type: ignore[arg-type]

    @classmethod
    def sandboxed(cls, **kwargs: object) -> ToolSpec:
        defaults: dict[str, object] = {
            "execution_category": ExecutionCategory.EXECUTE,
            "risk_level": RiskLevel.MEDIUM,
            "interrupt_behavior": InterruptBehavior.CANCEL,
            "concurrency_safe": False,
            "requires_confirmation": False,
        }
        defaults.update(kwargs)
        return cls(**defaults)  # type: ignore[arg-type]


def _coerce_execution_category(value: object) -> ExecutionCategory:
    if isinstance(value, ExecutionCategory):
        return value
    if isinstance(value, str):
        try:
            return ExecutionCategory(value)
        except ValueError as exc:
            raise ValueError(f"invalid execution_category: {value!r}") from exc
    raise TypeError(f"execution_category must be ExecutionCategory, got {type(value).__name__}")


def _coerce_risk_level(value: object) -> RiskLevel:
    if isinstance(value, RiskLevel):
        return value
    if isinstance(value, str):
        try:
            return RiskLevel(value)
        except ValueError as exc:
            raise ValueError(f"invalid risk_level: {value!r}") from exc
    raise TypeError(f"risk_level must be RiskLevel, got {type(value).__name__}")


def _coerce_interrupt_behavior(value: object) -> InterruptBehavior:
    if isinstance(value, InterruptBehavior):
        return value
    if isinstance(value, str):
        try:
            return InterruptBehavior(value)
        except ValueError as exc:
            raise ValueError(f"invalid interrupt_behavior: {value!r}") from exc
    raise TypeError(
        "interrupt_behavior must be InterruptBehavior, "
        f"got {type(value).__name__}"
    )


def _category_risk_level(category: ExecutionCategory) -> RiskLevel:
    if category in (ExecutionCategory.MUTATE, ExecutionCategory.SYSTEM):
        return RiskLevel.HIGH
    if category in (
        ExecutionCategory.WRITE,
        ExecutionCategory.EXECUTE,
        ExecutionCategory.NETWORK,
    ):
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


def _permissions_require_approval(permissions: ToolPermissions) -> bool:
    return (
        permissions.write_db
        or permissions.kg_mutation
        or permissions.user_data
        or permissions.write_fs
        or permissions.execute_code
        or permissions.external_network
    )


class ToolError(BaseModel):
    code: str
    message: str
    retryable: bool
    detail: dict[str, object] = Field(default_factory=dict)


class ToolResult(BaseModel):
    tool_call_id: str
    tool_name: str
    status: Literal["ok", "error"]
    output: BaseModel | None = None
    error: ToolError | None = None
    latency_ms: float
    work_units_used: int = 0
    # Legacy checkpoint field. New tool execution records work_units_used instead.
    token_used: int = 0
    retry_count: int = 0

    @field_validator("output", mode="before")
    @classmethod
    def _restore_typed_output(cls, value: object) -> object:
        if not isinstance(value, dict) or value.get("__rag_model_payload__") is not True:
            return value
        module = value.get("module")
        name = value.get("name")
        data = value.get("data")
        if not isinstance(module, str) or not isinstance(name, str):
            raise ValueError("typed tool output payload is missing module/name")
        model_cls = getattr(importlib.import_module(module), name)
        if not isinstance(model_cls, type) or not issubclass(model_cls, BaseModel):
            raise ValueError(f"typed tool output is not a Pydantic model: {module}.{name}")
        return model_cls.model_validate(data)

    @field_serializer("output")
    def _serialize_typed_output(
        self,
        output: BaseModel | None,
        info: SerializationInfo,
    ) -> object:
        if output is None:
            return None
        if info.mode == "json":
            return output.model_dump(mode="json")
        return {
            "__rag_model_payload__": True,
            "module": output.__class__.__module__,
            "name": output.__class__.__name__,
            "data": output.model_dump(mode="json"),
        }

    @model_validator(mode="after")
    def _check_exclusivity(self) -> ToolResult:
        if self.status == "ok":
            if self.output is None:
                raise ValueError("output is required when status='ok'")
            if self.error is not None:
                raise ValueError("error must be None when status='ok'")
        if self.status == "error":
            if self.error is None:
                raise ValueError("error is required when status='error'")
            if self.output is not None:
                raise ValueError("output must be None when status='error'")
        return self

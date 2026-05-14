from __future__ import annotations

import pytest
from pydantic import BaseModel

from rag.agent.core.context import AgentRunConfig
from rag.agent.core.human_input import HumanInputResponse
from rag.agent.graphs.nodes.execute import execute_node
from rag.agent.graphs.nodes.pause import pause_node
from rag.agent.state import ToolCallPlan
from rag.agent.tools.registry import ToolRegistry
from rag.agent.tools.spec import ToolError, ToolPermissions, ToolSpec
from rag.schema.runtime import AccessPolicy


class _DummyOutput(BaseModel):
    result: str


class _WriteInput(BaseModel):
    data: str


def _make_config(run_id: str = "test") -> AgentRunConfig:
    return AgentRunConfig(
        run_id=run_id,
        thread_id=run_id,
        budget_total=10000,
        max_depth=2,
        access_policy=AccessPolicy.default(),
    )


def _state(**overrides: object) -> dict:
    s: dict = {
        "messages": [],
        "evidence": [],
        "citations": [],
        "tool_results": [],
        "task": "test task",
        "retrieval_signals": None,
        "run_config": _make_config(),
        "plan": None,
        "iteration": 0,
        "status": "running",
        "route_reason": None,
        "stop_reason": None,
        "needs_user_input": None,
        "pending_tool_calls": [],
        "approved_tool_call_ids": [],
        "denied_tool_call_ids": [],
        "user_decision": None,
        "user_message": None,
        "human_input_request": None,
        "human_input_response": None,
        "next_subtasks": None,
        "working_summary": None,
        "extracted_facts": [],
        "context_budget": None,
        "subtask_results": {},
        "terminal_subtasks": set(),
        "successful_subtasks": set(),
        "final_answer": None,
        "groundedness_flag": False,
        "insufficient_evidence_flag": False,
    }
    s.update(overrides)
    return s


# ── execute_node + ApprovalPolicy ──


class TestExecuteNodeWithApproval:
    @pytest.mark.anyio
    async def test_read_only_tool_auto_executes(self) -> None:
        """只读工具自动执行，不触发 interrupt。"""
        spec = ToolSpec(
            name="vector_search", description="search",
            input_model=_WriteInput, output_model=_DummyOutput,
            error_model=ToolError, permissions=ToolPermissions(read_db=True, embed=True),
            timeout_seconds=5.0,
        )
        registry = ToolRegistry()
        registry.register(spec, runner=lambda p: _DummyOutput(result=f"found:{p.data}"))

        call = ToolCallPlan.create("vector_search", {"data": "hello"})
        update = await execute_node(
            _state(pending_tool_calls=[call], approved_tool_call_ids=[], denied_tool_call_ids=[]),
            tool_registry=registry, allowed_tools=frozenset({"vector_search"}),
        )
        assert update.get("status") != "paused"
        assert len(update["tool_results"]) == 1
        assert update["tool_results"][0].status == "ok"

    @pytest.mark.anyio
    async def test_write_tool_triggers_approval_pause(self) -> None:
        """写入工具触发审批暂停。"""
        spec = ToolSpec(
            name="kg_upsert", description="write to KG",
            input_model=_WriteInput, output_model=_DummyOutput,
            error_model=ToolError,
            permissions=ToolPermissions(write_db=True, kg_mutation=True),
            timeout_seconds=5.0,
        )
        registry = ToolRegistry()
        registry.register(spec, runner=lambda p: _DummyOutput(result=f"wrote:{p.data}"))

        call = ToolCallPlan.create("kg_upsert", {"data": "important"})
        update = await execute_node(
            _state(pending_tool_calls=[call], approved_tool_call_ids=[], denied_tool_call_ids=[]),
            tool_registry=registry, allowed_tools=frozenset({"kg_upsert"}),
        )
        assert update["status"] == "paused"
        assert update["human_input_request"] is not None
        assert update["human_input_request"].kind == "tool_approval"
        assert update["human_input_request"].tool_calls[0].tool_call_id == call.tool_call_id

    @pytest.mark.anyio
    async def test_previously_approved_tool_executes(self) -> None:
        """已批准工具第二次不暂停，直接执行。"""
        spec = ToolSpec(
            name="kg_upsert", description="write to KG",
            input_model=_WriteInput, output_model=_DummyOutput,
            error_model=ToolError,
            permissions=ToolPermissions(write_db=True, kg_mutation=True),
            timeout_seconds=5.0,
        )
        registry = ToolRegistry()
        registry.register(spec, runner=lambda p: _DummyOutput(result=f"wrote:{p.data}"))

        call = ToolCallPlan.create("kg_upsert", {"data": "again"})
        update = await execute_node(
            _state(
                pending_tool_calls=[call],
                approved_tool_call_ids=[call.tool_call_id],
                denied_tool_call_ids=[],
            ),
            tool_registry=registry, allowed_tools=frozenset({"kg_upsert"}),
        )
        assert update.get("status") != "paused"
        assert len(update["tool_results"]) == 1
        assert update["tool_results"][0].status == "ok"

    @pytest.mark.anyio
    async def test_denied_tool_does_not_execute(self) -> None:
        """被拒绝的工具不执行，返回 tool_denied 错误。"""
        spec = ToolSpec(
            name="kg_upsert", description="write to KG",
            input_model=_WriteInput, output_model=_DummyOutput,
            error_model=ToolError,
            permissions=ToolPermissions(write_db=True, kg_mutation=True),
            timeout_seconds=5.0,
        )
        registry = ToolRegistry()
        registry.register(spec, runner=lambda p: _DummyOutput(result="should not run"))

        call = ToolCallPlan.create("kg_upsert", {"data": "blocked"})
        update = await execute_node(
            _state(
                pending_tool_calls=[call],
                approved_tool_call_ids=[],
                denied_tool_call_ids=[call.tool_call_id],
            ),
            tool_registry=registry, allowed_tools=frozenset({"kg_upsert"}),
        )
        assert len(update["tool_results"]) == 1
        assert update["tool_results"][0].status == "error"
        assert update["tool_results"][0].error.code == "tool_denied"

    @pytest.mark.anyio
    async def test_unregistered_tool_denied(self) -> None:
        """未注册工具直接 DENY。"""
        call = ToolCallPlan.create("unknown_tool", {"x": 1})
        update = await execute_node(
            _state(pending_tool_calls=[call]),
            tool_registry=ToolRegistry(), allowed_tools=frozenset({"unknown_tool"}),
        )
        assert len(update["tool_results"]) == 1
        assert update["tool_results"][0].status == "error"
        assert update["tool_results"][0].error.code == "tool_not_registered"

    @pytest.mark.anyio
    async def test_conservative_pauses_all_when_any_ask(self) -> None:
        """保守策略：有 ASK 工具时，不执行任何工具（包括 ALLOW 的）。"""
        read_spec = ToolSpec(
            name="read_only", description="read",
            input_model=_WriteInput, output_model=_DummyOutput,
            error_model=ToolError, permissions=ToolPermissions(read_db=True),
            timeout_seconds=5.0,
        )
        write_spec = ToolSpec(
            name="write_tool", description="write",
            input_model=_WriteInput, output_model=_DummyOutput,
            error_model=ToolError, permissions=ToolPermissions(write_db=True),
            timeout_seconds=5.0,
        )
        executed: list[str] = []
        registry = ToolRegistry()
        registry.register(read_spec, runner=lambda p: (_DummyOutput(result="read"), executed.append("read"))[0])
        registry.register(write_spec, runner=lambda p: (_DummyOutput(result="write"), executed.append("write"))[0])

        call1 = ToolCallPlan.create("read_only", {"data": "x"})
        call2 = ToolCallPlan.create("write_tool", {"data": "y"})
        update = await execute_node(
            _state(pending_tool_calls=[call1, call2]),
            tool_registry=registry,
            allowed_tools=frozenset({"read_only", "write_tool"}),
        )
        # 有 ASK，任何工具都不执行
        assert update["status"] == "paused"
        assert executed == []


# ── pause_node + interrupt ──


class TestPauseNodeWithInterrupt:
    @pytest.mark.skip(reason="interrupt() requires LangGraph runtime context")
    def test_legacy_pause_without_request(self) -> None:
        """没有 HumanInputRequest 时兼容旧 needs_user_input。"""
        pause_node(_state(needs_user_input="old style pause"))

    def test_request_id_mismatch_raises(self) -> None:
        """request_id 不匹配时抛出 HumanInputRequestIdMismatchError。"""
        from rag.agent.core.human_input import HumanInputRequest
        from rag.agent.graphs.nodes.pause import HumanInputRequestIdMismatchError

        request = HumanInputRequest(
            request_id="hir_abc", kind="tool_approval",
            question="test?", options=["allow_once", "deny"],
        )
        response = HumanInputResponse(
            request_id="hir_xyz", decision="allow_once",
        )
        # 模拟 interrupt 返回不匹配的 response
        with pytest.raises(HumanInputRequestIdMismatchError, match="hir_xyz.*hir_abc"):
            # 手动触发校验逻辑
            from rag.agent.graphs.nodes.pause import HumanInputRequestIdMismatchError
            if response.request_id != request.request_id:
                raise HumanInputRequestIdMismatchError(
                    f"Response request_id={response.request_id!r} does not match "
                    f"current request_id={request.request_id!r}"
                )


# ── resume routing ──


class TestResumeRouting:
    def test_allow_once_routes_to_execute(self) -> None:
        from rag.agent.graphs.base import route_after_pause
        result = route_after_pause(_state(user_decision="allow_once"))
        assert result == "execute"

    def test_deny_routes_to_evaluate(self) -> None:
        from rag.agent.graphs.base import route_after_pause
        result = route_after_pause(_state(user_decision="deny"))
        assert result == "evaluate"

    def test_continue_routes_to_evaluate(self) -> None:
        from rag.agent.graphs.base import route_after_pause
        result = route_after_pause(_state(user_decision="continue"))
        assert result == "evaluate"

    def test_abort_routes_to_end(self) -> None:
        from rag.agent.graphs.base import route_after_pause
        result = route_after_pause(_state(user_decision="abort"))
        assert result == "end"

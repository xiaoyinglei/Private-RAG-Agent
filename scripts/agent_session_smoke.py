#!/usr/bin/env python
"""Real Session/Turn smoke with a process boundary and one input-file Turn."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import cast
from uuid import UUID

from agent_runtime import Agent
from rag.agent.sessions import SessionStore, TurnStatus

DEFAULT_MODEL = "qwen3_5_9b_mlx_4bit"


def _chat(
    agent: Agent,
    message: str,
    *,
    session_id: str | None,
    files: tuple[str, ...] = (),
) -> dict[str, object]:
    result = agent.chat(
        message,
        session_id=session_id,
        files=files,
        max_tokens_total=16_000,
    )
    if result.status != "done":
        raise RuntimeError(
            f"Turn {result.turn_id} ended with status={result.status!r}: "
            f"{result.answer!r}"
        )
    return {
        "session_id": result.session_id,
        "turn_id": result.turn_id,
        "answer": result.answer or "",
        "tools": list(result.tool_calls),
        "workspace_path": getattr(result.raw, "workspace_path", None),
    }


def _require_tokens(answer: str, *tokens: str) -> None:
    folded = answer.casefold()
    missing = [token for token in tokens if token.casefold() not in folded]
    if missing:
        raise AssertionError(
            f"answer is missing {missing!r}: {answer!r}"
        )


def _seed(root: Path, model: str) -> dict[str, object]:
    database = root / "agent.sqlite"
    workspace = root / "workspace"
    fixture = root / "session-facts.txt"
    fixture.write_text(
        "The durable file secret is FILE-CEDAR-902.\n",
        encoding="utf-8",
    )
    agent = Agent(
        model=model,
        checkpoint_db=database,
        workspace_path=workspace,
    )
    turns: list[dict[str, object]] = []
    turns.append(
        _chat(
            agent,
            "Remember these durable conversation facts: codename ORCHID-731, "
            "owner Lin, and budget 4287. Do not use tools. Reply ACK-1.",
            session_id=None,
        )
    )
    session_id = str(turns[0]["session_id"])
    turns.append(
        _chat(
            agent,
            "Add region Hangzhou and deadline Friday to the facts. Do not use "
            "tools. Reply ACK-2.",
            session_id=session_id,
        )
    )
    file_turn = _chat(
        agent,
        "Use the file tools to read the attached file, then reply with exactly "
        "its durable file secret.",
        session_id=session_id,
        files=(str(fixture),),
    )
    turns.append(file_turn)
    _require_tokens(str(file_turn["answer"]), "FILE-CEDAR-902")
    if "read_file" not in cast(list[str], file_turn["tools"]):
        raise AssertionError(f"file Turn did not call read_file: {file_turn!r}")
    codename_turn = _chat(
        agent,
        "What is the codename from this Session? Do not use tools. Reply only "
        "with the codename.",
        session_id=session_id,
    )
    turns.append(codename_turn)
    _require_tokens(str(codename_turn["answer"]), "ORCHID-731")
    budget_turn = _chat(
        agent,
        "Using the remembered budget, what is 4287 plus 13? Do not use tools. "
        "Reply only with the number.",
        session_id=session_id,
    )
    turns.append(budget_turn)
    _require_tokens(str(budget_turn["answer"]), "4300")
    payload: dict[str, object] = {
        "model": model,
        "session_id": session_id,
        "turns": turns,
    }
    (root / "seed.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def _continue(root: Path) -> dict[str, object]:
    state = json.loads((root / "seed.json").read_text(encoding="utf-8"))
    session_id = str(state["session_id"])
    # Deliberately wrong caller settings prove persisted Session metadata wins
    # before workspace/model runtime assembly in this new process.
    agent = Agent(
        model="groq_gpt_oss_120b",
        checkpoint_db=root / "agent.sqlite",
        workspace_path=root / "wrong-workspace",
    )
    turns = list(state["turns"])
    owner_turn = _chat(
        agent,
        "Who is the remembered owner? Do not use tools. Reply only with the "
        "owner.",
        session_id=session_id,
    )
    turns.append(owner_turn)
    _require_tokens(str(owner_turn["answer"]), "Lin")
    region_turn = _chat(
        agent,
        "Give the remembered region and deadline. Do not use tools.",
        session_id=session_id,
    )
    turns.append(region_turn)
    _require_tokens(str(region_turn["answer"]), "Hangzhou", "Friday")
    file_turn = _chat(
        agent,
        "Recall the durable file secret from the earlier file Turn without "
        "calling tools.",
        session_id=session_id,
    )
    turns.append(file_turn)
    _require_tokens(str(file_turn["answer"]), "FILE-CEDAR-902")
    turns.append(
        _chat(
            agent,
            "Update the conversation fact: the deadline is now Tuesday, not "
            "Friday. Do not use tools. Reply ACK-9.",
            session_id=session_id,
        )
    )
    final_turn = _chat(
        agent,
        "Return the final codename, owner, budget, region, deadline, and file "
        "secret. Do not use tools.",
        session_id=session_id,
    )
    turns.append(final_turn)
    _require_tokens(
        str(final_turn["answer"]),
        "ORCHID-731",
        "Lin",
        "4287",
        "Hangzhou",
        "Tuesday",
        "FILE-CEDAR-902",
    )
    payload: dict[str, object] = {
        "model": state["model"],
        "session_id": session_id,
        "turns": turns,
    }
    (root / "complete.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def _verify(root: Path) -> dict[str, object]:
    state = json.loads((root / "complete.json").read_text(encoding="utf-8"))
    session_id = str(state["session_id"])
    turns = list(state["turns"])
    UUID(session_id)
    turn_ids = [str(item["turn_id"]) for item in turns]
    if len(turn_ids) != 10 or len(set(turn_ids)) != 10:
        raise AssertionError(f"expected ten distinct Turns: {turn_ids!r}")
    for turn_id in turn_ids:
        UUID(turn_id)
    if session_id in turn_ids:
        raise AssertionError("Session ID was reused as a Turn ID")

    store = SessionStore(root / "agent.sqlite")
    session = store.get_session(session_id)
    if session.active_turn_id is not None:
        raise AssertionError(f"Session still has active Turn {session.active_turn_id}")
    if session.runtime.model_alias != state["model"]:
        raise AssertionError(f"persisted model binding drifted: {session.runtime}")
    expected_workspace = str((root / "workspace").resolve())
    if session.runtime.workspace_path != expected_workspace:
        raise AssertionError(f"persisted workspace binding drifted: {session.runtime}")
    for turn_id in turn_ids:
        if store.get_turn(turn_id).status is not TurnStatus.COMPLETED:
            raise AssertionError(f"Turn is not completed: {turn_id}")
    history = store.history(session_id)
    if len(history) < 20:
        raise AssertionError(f"canonical history is unexpectedly short: {len(history)}")
    if history[0].role != "user" or history[-1].role != "assistant":
        raise AssertionError("canonical history endpoints are invalid")
    store.close()
    return {
        "passed": True,
        "root": str(root),
        "session_id": session_id,
        "turn_ids": turn_ids,
        "canonical_messages": len(history),
        "file_tool_calls": turns[2]["tools"],
        "final_answer": turns[-1]["answer"],
    }


def _run_child(phase: str, root: Path, model: str) -> None:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--phase",
        phase,
        "--root",
        str(root),
        "--model",
        model,
    ]
    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
        timeout=900,
    )
    if completed.returncode:
        raise RuntimeError(
            f"{phase} process failed\nstdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=("all", "seed", "continue", "verify"),
        default="all",
    )
    parser.add_argument("--root", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    root = (
        args.root.expanduser().resolve()
        if args.root is not None
        else Path(tempfile.mkdtemp(prefix="agent-session-smoke-"))
    )
    root.mkdir(parents=True, exist_ok=True)
    if args.phase == "seed":
        print(json.dumps(_seed(root, args.model), ensure_ascii=False))
        return
    if args.phase == "continue":
        print(json.dumps(_continue(root), ensure_ascii=False))
        return
    if args.phase == "verify":
        print(json.dumps(_verify(root), ensure_ascii=False))
        return
    _run_child("seed", root, args.model)
    _run_child("continue", root, args.model)
    print(json.dumps(_verify(root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
from pathlib import Path

from tests.agent.parity.scenarios import run_legacy_scenarios

EXPECTED_SOURCE_COMMIT = "dbd746b9"
BASELINE_PATH = (
    Path(__file__).parent
    / "baselines"
    / "legacy_graph_v1.json"
)


def _source_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "--short=8", "HEAD"],
        text=True,
    ).strip()


async def _capture() -> dict[str, object]:
    source_commit = _source_commit()
    if source_commit != EXPECTED_SOURCE_COMMIT:
        raise SystemExit(
            "legacy baseline may only be captured from "
            f"{EXPECTED_SOURCE_COMMIT}; current HEAD is {source_commit}"
        )
    return {
        "schema_version": 1,
        "source_commit": source_commit,
        "scenarios": await run_legacy_scenarios(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--update",
        action="store_true",
        help="Explicitly replace the reviewed legacy baseline.",
    )
    args = parser.parse_args()
    if not args.update:
        raise SystemExit("refusing to rewrite baseline without --update")

    payload = asyncio.run(_capture())
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

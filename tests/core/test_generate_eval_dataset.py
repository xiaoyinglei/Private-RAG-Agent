from __future__ import annotations

from generate_eval_dataset import SectionWindow, build_parser, build_prompt, split_section_to_windows


class _WhitespaceTokenAccounting:
    def count(self, text: str) -> int:
        return len(text.split())

    def _offset_spans(self, text: str) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        cursor = 0
        for token in text.split():
            start = text.index(token, cursor)
            end = start + len(token)
            spans.append((start, end))
            cursor = end
        return spans


def test_eval_dataset_windows_split_by_tokens_not_characters() -> None:
    text = " ".join(f"token{i}" for i in range(12))

    windows = split_section_to_windows(
        doc_id="1",
        section_id="2",
        text=text,
        max_window_tokens=5,
        window_overlap_tokens=1,
        min_window_tokens=1,
        token_accounting=_WhitespaceTokenAccounting(),  # type: ignore[arg-type]
    )

    assert [window.window_text for window in windows] == [
        "token0 token1 token2 token3 token4",
        "token4 token5 token6 token7 token8",
        "token8 token9 token10 token11",
    ]
    assert [window.token_range_start for window in windows] == [0, 4, 8]
    assert [window.token_range_end for window in windows] == [5, 9, 12]


def test_eval_dataset_parser_accepts_small_local_model_options() -> None:
    args = build_parser().parse_args(
        [
            "--input",
            "sections.jsonl",
            "--output",
            "golden.jsonl",
            "--model",
            "/models/Qwen3-8B-MLX-4bit",
            "--base-url",
            "http://127.0.0.1:8080/v1",
            "--api-key",
            "not-needed",
            "--max-window-tokens",
            "700",
            "--window-overlap-tokens",
            "80",
        ]
    )

    assert args.input == "sections.jsonl"
    assert args.output == "golden.jsonl"
    assert args.model == "/models/Qwen3-8B-MLX-4bit"
    assert args.base_url == "http://127.0.0.1:8080/v1"
    assert args.api_key == "not-needed"
    assert args.max_window_tokens == 700
    assert args.window_overlap_tokens == 80


def test_eval_dataset_prompt_rejects_context_free_generic_questions() -> None:
    prompt = build_prompt(
        SectionWindow(
            doc_id="1",
            section_id="2",
            text="售假制假造成损失时，按照产品市场价计算赔偿。",
            window_text="售假制假造成损失时，按照产品市场价计算赔偿。",
            window_index=0,
            char_range_start=0,
            char_range_end=20,
            token_range_start=0,
            token_range_end=10,
            title="假一罚十承诺书",
            toc_path=["假一罚十承诺书"],
        ),
        "single_section_fact",
    )

    assert "不能是脱离上下文的泛问" in prompt
    assert "针对【具体流程/业务场景】，某种情况如何处理？" in prompt
    assert "《假一罚十承诺书》中" not in prompt

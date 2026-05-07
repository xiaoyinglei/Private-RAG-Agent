from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Literal, Protocol

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, ValidationError

from rag.assembly import TokenAccountingService, TokenizerContract

QuestionType = Literal[
    "single_section_fact",
    "rule_condition",
    "process_step",
]


class QAGeneration(BaseModel):
    question: str = Field(description="基于制度片段生成的问题。")
    answer: str = Field(description="基于制度片段生成的标准答案。")


class SectionWindow(BaseModel):
    doc_id: str
    section_id: str
    text: str
    window_text: str
    window_index: int
    char_range_start: int
    char_range_end: int
    token_range_start: int
    token_range_end: int
    title: str | None = None
    toc_path: list[str] = Field(default_factory=list)
    source_id: str | int | None = None


class TokenAccountingLike(Protocol):
    def count(self, text: str) -> int: ...
    def _offset_spans(self, text: str) -> list[tuple[int, int]] | None: ...


QUESTION_TYPE_INSTRUCTIONS: dict[QuestionType, str] = {
    "single_section_fact": """
题型：single_section_fact，单 section 事实题。

生成要求：
1. 问题只考察当前制度片段中的一个明确事实。
2. 答案必须能从片段中直接找到依据。
3. 不要跨 section 推理。
4. 不要问太泛的问题。
5. 适合测试 RAG 是否能从单个 section 中找到明确答案。

好问题示例：
- 员工申请事假需要提前多久提交申请？
- 哪些材料需要随报销单一起提交？
""".strip(),
    "rule_condition": """
题型：rule_condition，条件/门槛/例外题。

生成要求：
1. 问题必须包含条件、门槛、范围、例外、限制中的至少一种。
2. 适合考察制度中的“如果……则……”“超过……需要……”“除……外……”这类规则。
3. 答案要明确说明条件和对应处理结果。
4. 不要编造片段中没有出现的条件。
5. 如果片段没有明显条件或门槛，不要硬编，返回空 question 和空 answer。

好问题示例：
- 如果报销金额超过规定额度，需要经过哪些额外审批？
- 哪些情况下员工不能享受该项福利？
""".strip(),
    "process_step": """
题型：process_step，流程步骤题。

生成要求：
1. 问题必须围绕流程、步骤、审批链路、办理顺序生成。
2. 答案要按顺序说明步骤。
3. 适合考察制度中的“先……再……最后……”。
4. 如果片段中没有流程，不要硬编，返回空 question 和空 answer。
5. 不要把单个事实包装成流程题。

好问题示例：
- 员工申请调岗需要经过哪些流程？
- 合同审批从提交到归档需要经过哪些步骤？
""".strip(),
}


INPUT_FILE = Path("data/company_policy_sections.jsonl")
OUTPUT_FILE = Path("data/eval_private/golden_eval_dataset.jsonl")
FAILED_FILE = Path("data/eval_private/golden_eval_failed.jsonl")

MODEL_NAME = "Qwen/Qwen3-8B-MLX-4bit"
BASE_URL = "http://127.0.0.1:8081/v1"
DUMMY_API_KEY = "local-mlx-dummy-key"

MAX_WINDOW_TOKENS = 700
WINDOW_OVERLAP_TOKENS = 80
MIN_WINDOW_TOKENS = 120

MAX_RETRIES = 2
RETRY_SLEEP_SECONDS = 1.5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a private golden retrieval eval dataset from SectionRecord JSONL."
    )
    parser.add_argument("--input", default=str(INPUT_FILE))
    parser.add_argument("--output", default=str(OUTPUT_FILE))
    parser.add_argument("--failed-output", default=str(FAILED_FILE))
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--api-key", default=DUMMY_API_KEY)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument("--max-window-tokens", type=int, default=MAX_WINDOW_TOKENS)
    parser.add_argument("--window-overlap-tokens", type=int, default=WINDOW_OVERLAP_TOKENS)
    parser.add_argument("--min-window-tokens", type=int, default=MIN_WINDOW_TOKENS)
    parser.add_argument(
        "--question-types",
        default="single_section_fact,rule_condition,process_step",
        help="Comma-separated question types.",
    )
    parser.add_argument("--limit-windows", type=int, default=None)
    parser.add_argument("--limit-tasks", type=int, default=None)
    return parser


def build_llm(
    *,
    model: str = MODEL_NAME,
    base_url: str = BASE_URL,
    api_key: str = DUMMY_API_KEY,
    temperature: float = 0.1,
    max_output_tokens: int = 1024,
) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_output_tokens,
    )


def stable_hash(text: str, length: int = 10) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def build_generation_key(
    doc_id: str,
    section_id: str,
    window_index: int,
    question_type: QuestionType,
) -> str:
    raw = f"{doc_id}::{section_id}::{window_index}::{question_type}"
    return stable_hash(raw, length=16)


def build_query_id(
    doc_id: str,
    section_id: str,
    window_index: int,
    question_type: QuestionType,
    question: str,
) -> str:
    raw = f"{doc_id}::{section_id}::{window_index}::{question_type}::{question}"
    return f"policy_{stable_hash(raw, length=16)}"


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def default_token_accounting(*, max_window_tokens: int, window_overlap_tokens: int) -> TokenAccountingService:
    return TokenAccountingService(
        TokenizerContract(
            embedding_model_name="eval-generation",
            tokenizer_model_name="eval-generation",
            chunking_tokenizer_model_name="eval-generation",
            tokenizer_backend="simple",
            chunk_token_size=max(max_window_tokens, 1),
            chunk_overlap_tokens=max(window_overlap_tokens, 0),
            local_files_only=True,
        )
    )


def offset_spans(text: str, token_accounting: TokenAccountingLike | None) -> list[tuple[int, int]]:
    if token_accounting is not None:
        spans_method = getattr(token_accounting, "_offset_spans", None)
        if callable(spans_method):
            try:
                spans = spans_method(text)
                if spans:
                    return list(spans)
            except Exception:
                pass
    return [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]


def normalize_toc_path(toc_path: list[Any] | str | int | None) -> list[str]:
    if toc_path is None:
        return []

    if isinstance(toc_path, list):
        return [str(item).strip() for item in toc_path if str(item).strip()]

    if isinstance(toc_path, str):
        text = toc_path.strip()
        if not text:
            return []
        return [text]

    return [str(toc_path).strip()]


def split_section_to_windows(
    *,
    doc_id: str,
    section_id: str,
    text: str,
    max_window_tokens: int = MAX_WINDOW_TOKENS,
    window_overlap_tokens: int = WINDOW_OVERLAP_TOKENS,
    min_window_tokens: int = MIN_WINDOW_TOKENS,
    token_accounting: TokenAccountingLike | None = None,
    title: str | None = None,
    toc_path: list[Any] | str | int | None = None,
    source_id: str | int | None = None,
) -> list[SectionWindow]:
    normalized = normalize_text(text)
    normalized_toc_path = normalize_toc_path(toc_path)
    spans = offset_spans(normalized, token_accounting=token_accounting)
    token_count = len(spans)

    if not normalized or not spans:
        return []

    if token_count <= max_window_tokens:
        return [
            SectionWindow(
                doc_id=doc_id,
                section_id=section_id,
                text=normalized,
                window_text=normalized,
                window_index=0,
                char_range_start=0,
                char_range_end=len(normalized),
                token_range_start=0,
                token_range_end=token_count,
                title=title,
                toc_path=normalized_toc_path,
                source_id=source_id,
            )
        ]

    windows: list[SectionWindow] = []
    token_start = 0
    window_index = 0
    window_size = max(max_window_tokens, 1)
    overlap = min(max(window_overlap_tokens, 0), max(window_size - 1, 0))
    step = max(window_size - overlap, 1)

    while token_start < token_count:
        token_end = min(token_start + window_size, token_count)
        char_start = spans[token_start][0]
        char_end = spans[token_end - 1][1]
        window_text = normalized[char_start:char_end].strip()

        if token_end - token_start >= max(min_window_tokens, 1):
            windows.append(
                SectionWindow(
                    doc_id=doc_id,
                    section_id=section_id,
                    text=normalized,
                    window_text=window_text,
                    window_index=window_index,
                    char_range_start=char_start,
                    char_range_end=char_end,
                    token_range_start=token_start,
                    token_range_end=token_end,
                    title=title,
                    toc_path=normalized_toc_path,
                    source_id=source_id,
                )
            )
            window_index += 1

        if token_end >= token_count:
            break

        token_start += step

    return windows


def load_existing_generation_keys(output_file: Path) -> set[str]:
    existing_keys: set[str] = set()

    if not output_file.exists():
        return existing_keys

    with output_file.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            generation_key = item.get("generation_key")

            if generation_key:
                existing_keys.add(str(generation_key))

    return existing_keys


def extract_json_object(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()

    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"没有找到 JSON 对象：{text[:300]}")

    json_text = text[start : end + 1]

    return json.loads(json_text)


def build_prompt(window: SectionWindow, question_type: QuestionType) -> str:
    question_type_instruction = QUESTION_TYPE_INSTRUCTIONS[question_type]
    toc_path_text = " > ".join(window.toc_path)

    return f"""
你是一个资深的企业合规审计员，正在为 RAG 系统生成本地黄金评测集。

你的任务：
根据【制度片段】生成 1 个指定题型的问答对。

总要求：
1. 只能根据【制度片段】生成问题和答案。
2. 不允许使用制度片段之外的常识、经验或猜测。
3. 不允许补充片段中没有出现的审批人、金额、天数、流程、例外。
4. 如果该制度片段不适合生成指定题型，返回空字符串：
   {{"question": "", "answer": ""}}
5. 问题要像普通员工、业务人员、人事或财务人员真实会问的问题。
6. 答案必须准确、克制、可追溯，不要扩写。
7. 禁止输出推理过程、思考过程、分析过程。
8. 禁止输出 Markdown。
9. 只能输出一个 JSON 对象。
10. 如果制度片段内容不足以支撑高质量问题，请返回空 question 和空 answer。
11. answer 必须尽量引用制度片段中的原始表述，不要改写成制度外解释。
12. question 不能是脱离上下文的泛问，必须包含至少一个来自 title、toc_path 或制度片段的业务场景关键词。
13. 避免只问“如何处理/如何计算/需要哪些条件/有哪些步骤”这类泛问题；要写清楚制度对象或业务场景。

泛问题形式（禁止）：
- 某个标准如何计算？
- 某种情况如何处理？
- 某个流程需要哪些条件？

合格问题形式：
- 在【具体制度/业务对象】中，某个标准如何计算？
- 针对【具体流程/业务场景】，某种情况如何处理？
- 办理【具体事项】时，需要满足哪些条件？

【目标题型】
{question_type}

【题型要求】
{question_type_instruction}

【制度片段元信息】
doc_id: {window.doc_id}
section_id: {window.section_id}
window_index: {window.window_index}
char_range_start: {window.char_range_start}
char_range_end: {window.char_range_end}
token_range_start: {window.token_range_start}
token_range_end: {window.token_range_end}
title: {window.title or ""}
toc_path: {toc_path_text}
source_id: {window.source_id or ""}

【制度片段】
{window.window_text}

请严格输出如下 JSON 格式：
{{
  "question": "这里填写问题",
  "answer": "这里填写标准答案"
}}
""".strip()


def generate_one_qa(
    *,
    llm: ChatOpenAI,
    window: SectionWindow,
    question_type: QuestionType,
) -> QAGeneration:
    prompt = build_prompt(window, question_type)

    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 2):
        try:
            response = llm.invoke(prompt)
            raw_content = str(response.content).strip()
            parsed = extract_json_object(raw_content)
            return QAGeneration.model_validate(parsed)
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            last_error = exc

            if attempt <= MAX_RETRIES:
                time.sleep(RETRY_SLEEP_SECONDS)
                continue

            raise exc

    raise RuntimeError(f"生成失败：{last_error}")


def append_jsonl(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False))
        f.write("\n")
        f.flush()


def load_section_windows(
    input_file: Path,
    *,
    failed_file: Path = FAILED_FILE,
    max_window_tokens: int = MAX_WINDOW_TOKENS,
    window_overlap_tokens: int = WINDOW_OVERLAP_TOKENS,
    min_window_tokens: int = MIN_WINDOW_TOKENS,
    token_accounting: TokenAccountingLike | None = None,
) -> list[SectionWindow]:
    windows: list[SectionWindow] = []

    with input_file.open(encoding="utf-8") as infile:
        for line_number, line in enumerate(infile, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                chunk_data = json.loads(line)
            except json.JSONDecodeError as exc:
                append_jsonl(
                    failed_file,
                    {
                        "line_number": line_number,
                        "error": f"JSONDecodeError: {exc}",
                        "raw_line": line[:500],
                    },
                )
                continue

            doc_id = chunk_data.get("doc_id")
            section_id = chunk_data.get("section_id")
            text_content = chunk_data.get("text", chunk_data.get("content", ""))

            if doc_id is None or section_id is None or not text_content:
                append_jsonl(
                    failed_file,
                    {
                        "line_number": line_number,
                        "error": "缺少必要字段 doc_id / section_id / text",
                        "raw_item": chunk_data,
                    },
                )
                continue

            section_windows = split_section_to_windows(
                doc_id=str(doc_id),
                section_id=str(section_id),
                text=str(text_content),
                max_window_tokens=max_window_tokens,
                window_overlap_tokens=window_overlap_tokens,
                min_window_tokens=min_window_tokens,
                token_accounting=token_accounting,
                title=chunk_data.get("title"),
                toc_path=chunk_data.get("toc_path"),
                source_id=chunk_data.get("source_id"),
            )

            windows.extend(section_windows)

    return windows


def build_eval_item(
    *,
    window: SectionWindow,
    question_type: QuestionType,
    qa: QAGeneration,
    generation_key: str,
    generation_model: str,
) -> dict[str, Any]:
    query_id = build_query_id(
        doc_id=window.doc_id,
        section_id=window.section_id,
        window_index=window.window_index,
        question_type=question_type,
        question=qa.question,
    )

    return {
        "query_id": query_id,
        "generation_key": generation_key,
        "question": qa.question,
        "answer": qa.answer,
        "doc_id": window.doc_id,
        "section_id": window.section_id,
        "title": window.title,
        "toc_path": window.toc_path,
        "source_id": window.source_id,
        "question_type": question_type,
        "answerable": True,
        "evidence": [
            {
                "doc_id": window.doc_id,
                "section_id": window.section_id,
                "title": window.title,
                "toc_path": window.toc_path,
                "source_id": window.source_id,
                "char_range_start": window.char_range_start,
                "char_range_end": window.char_range_end,
                "token_range_start": window.token_range_start,
                "token_range_end": window.token_range_end,
                "window_index": window.window_index,
                "text": window.window_text,
            }
        ],
        "generation_backend": "mlx",
        "generation_model": generation_model,
    }


def parse_question_types(raw_value: str) -> list[QuestionType]:
    values = [item.strip() for item in raw_value.split(",") if item.strip()]
    allowed = set(QUESTION_TYPE_INSTRUCTIONS)
    invalid = [item for item in values if item not in allowed]
    if invalid:
        raise ValueError(f"unsupported question types: {', '.join(invalid)}")
    return [item for item in values if item in allowed]  # type: ignore[return-value]


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    input_file = Path(args.input)
    output_file = Path(args.output)
    failed_file = Path(args.failed_output)
    if not input_file.is_file():
        raise FileNotFoundError(f"找不到输入 SectionRecord JSONL 文件，或路径不是文件：{input_file}")

    llm = build_llm(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
    )

    question_types = parse_question_types(args.question_types)

    existing_generation_keys = load_existing_generation_keys(output_file)
    token_accounting = default_token_accounting(
        max_window_tokens=args.max_window_tokens,
        window_overlap_tokens=args.window_overlap_tokens,
    )
    windows = load_section_windows(
        input_file,
        failed_file=failed_file,
        max_window_tokens=args.max_window_tokens,
        window_overlap_tokens=args.window_overlap_tokens,
        min_window_tokens=args.min_window_tokens,
        token_accounting=token_accounting,
    )

    if args.limit_windows is not None:
        windows = windows[: max(args.limit_windows, 0)]

    total_tasks = len(windows) * len(question_types)
    if args.limit_tasks is not None:
        total_tasks = min(total_tasks, max(args.limit_tasks, 0))

    print(f"🚀 输入文件：{input_file}")
    print(f"💾 输出文件：{output_file}")
    print(f"🧩 section 窗口数：{len(windows)}")
    print(f"🧪 计划任务数：{total_tasks}")
    print(f"⏭️ 已存在任务数：{len(existing_generation_keys)}")
    print(f"🤖 本地模型：{args.model}")
    print(f"🔌 本地服务：{args.base_url}")
    print(f"🧮 窗口 token：max={args.max_window_tokens}, overlap={args.window_overlap_tokens}")

    finished_count = 0
    skipped_count = 0
    failed_count = 0
    attempted_count = 0

    for window in windows:
        for question_type in question_types:
            if args.limit_tasks is not None and attempted_count >= max(args.limit_tasks, 0):
                break
            generation_key = build_generation_key(
                doc_id=window.doc_id,
                section_id=window.section_id,
                window_index=window.window_index,
                question_type=question_type,
            )

            if generation_key in existing_generation_keys:
                skipped_count += 1
                continue

            attempted_count += 1
            print(
                "⏳ 生成中 | "
                f"doc={window.doc_id} | "
                f"section={window.section_id} | "
                f"window={window.window_index} | "
                f"type={question_type}"
            )

            try:
                qa = generate_one_qa(
                    llm=llm,
                    window=window,
                    question_type=question_type,
                )

                question = qa.question.strip()
                answer = qa.answer.strip()

                if not question or not answer:
                    print(f"⚠️ 不适合生成该题型，跳过：{question_type}")
                    existing_generation_keys.add(generation_key)
                    skipped_count += 1
                    continue

                eval_item = build_eval_item(
                    window=window,
                    question_type=question_type,
                    qa=qa,
                    generation_key=generation_key,
                    generation_model=args.model,
                )

                append_jsonl(output_file, eval_item)
                existing_generation_keys.add(generation_key)
                finished_count += 1

            except Exception as exc:
                failed_count += 1

                append_jsonl(
                    failed_file,
                    {
                        "generation_key": generation_key,
                        "doc_id": window.doc_id,
                        "section_id": window.section_id,
                        "window_index": window.window_index,
                        "question_type": question_type,
                        "char_range_start": window.char_range_start,
                        "char_range_end": window.char_range_end,
                        "error": repr(exc),
                    },
                )

                print(f"❌ 失败：{repr(exc)}")

    print("\n🎉 生成流程结束")
    print(f"✅ 新增成功：{finished_count}")
    print(f"⏭️ 跳过任务：{skipped_count}")
    print(f"❌ 失败任务：{failed_count}")
    print(f"📄 输出文件：{output_file}")
    if failed_count:
        print(f"🧾 失败日志：{failed_file}")
    else:
        print("🧾 失败日志：未生成（无失败记录）")


if __name__ == "__main__":
    main()

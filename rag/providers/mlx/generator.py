from __future__ import annotations

import json
import re
from typing import Any, TypeVar

from pydantic import BaseModel

from rag.schema.model_protocols import Generator

T = TypeVar("T", bound=BaseModel)


class MLXGenerator(Generator):
    """
    Apple Silicon / MLX 本地生成专员。
    """

    def __init__(
        self,
        model_name_or_path: str,
        *,
        tokenizer_config: dict[str, Any] | None = None,
    ) -> None:
        self._model_name_or_path = model_name_or_path
        self._tokenizer_config = tokenizer_config or {}

        try:
            from mlx_lm import load
        except ImportError as exc:
            raise RuntimeError(
                "mlx_lm is not installed. Please install mlx-lm before using MLXGenerator."
            ) from exc

        try:
            self._model, self._tokenizer = load(
                model_name_or_path,
                tokenizer_config=self._tokenizer_config,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load MLX model '{model_name_or_path}': {exc}"
            ) from exc

    def generate_text(self, *, prompt: str, **kwargs: Any) -> str:
        max_tokens = int(kwargs.pop("max_tokens", 512))
        temperature = float(kwargs.pop("temperature", 0.7))
        top_p = float(kwargs.pop("top_p", 0.95))

        try:
            from mlx_lm import generate  # noqa: F811
            from mlx_lm.sample_utils import make_sampler
        except ImportError as exc:
            raise RuntimeError(
                "mlx_lm is not installed. Please install mlx-lm before using MLXGenerator."
            ) from exc

        formatted_prompt = self._render_chat_prompt(self._tokenizer, prompt)
        sampler = make_sampler(temp=temperature, top_p=top_p)

        try:
            result = generate(
                self._model,
                self._tokenizer,
                prompt=formatted_prompt,
                max_tokens=max_tokens,
                sampler=sampler,
                verbose=False,
            )
        except Exception as exc:
            raise RuntimeError(f"MLX text generation failed: {exc}") from exc

        if not isinstance(result, str):
            result = str(result)

        return self._normalize_chat_text(result)

    def generate_structured(self, *, prompt: str, schema: type[T], **kwargs: Any) -> T:
        schema_json = schema.model_json_schema()

        structured_prompt = f"""
Return ONLY valid JSON matching this schema.
Do not include markdown fences, explanations, or extra text.

JSON schema:
{json.dumps(schema_json, ensure_ascii=False)}

User task:
{prompt}
""".strip()

        raw = self.generate_text(prompt=structured_prompt, **kwargs)
        cleaned = self._extract_json_object(raw)

        try:
            return schema.model_validate_json(cleaned)
        except Exception as exc:
            raise RuntimeError(
                f"MLX structured generation failed to parse into {schema.__name__}: {exc}"
            ) from exc

    @property
    def model_name_or_path(self) -> str:
        return self._model_name_or_path

    def close(self) -> None:
        pass

    @staticmethod
    def _render_chat_prompt(tokenizer: object, prompt: str) -> str:
        apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
        if callable(apply_chat_template):
            preferred_calls = (
                {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False},
                {"tokenize": False, "add_generation_prompt": True},
            )
            for kwargs in preferred_calls:
                try:
                    rendered = apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        **kwargs,
                    )
                except Exception:
                    continue
                if isinstance(rendered, str) and rendered.strip():
                    return rendered
        return prompt

    @staticmethod
    def _normalize_chat_text(text: str) -> str:
        normalized = re.sub(
            r"<think>\s*.*?\s*</think>\s*",
            "",
            text,
            flags=re.DOTALL,
        ).strip()
        return normalized or text.strip()

    @staticmethod
    def _extract_json_object(text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            return stripped

        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end != -1 and start < end:
            return stripped[start : end + 1]

        return stripped
from __future__ import annotations

import json
from typing import Any, TypeVar
import re
import torch
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from rag.schema.model_protocols import Generator

T = TypeVar("T", bound=BaseModel)


class HuggingFaceGenerator(Generator):
    """
    本地 Hugging Face 文本生成专员。

    适用于：
    - AutoModelForCausalLM 类模型
    - 本地或 Hugging Face 缓存中的生成模型
    """

    def __init__(
        self,
        model_name_or_path: str,
        *,
        device: str | None = None,
        torch_dtype: str | None = None,
        trust_remote_code: bool = False,
        local_files_only: bool = False,
    ) -> None:
        self._model_name_or_path = model_name_or_path

        tokenizer_kwargs: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "local_files_only": local_files_only,
        }
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "local_files_only": local_files_only,
        }

        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = self._resolve_torch_dtype(torch_dtype)

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            **tokenizer_kwargs,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            **model_kwargs,
        )

        self._device = self._resolve_device(device)
        self._model.to(self._device)
        self._model.eval()

        if self._tokenizer.pad_token is None and self._tokenizer.eos_token is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    def generate_text(self, *, prompt: str, **kwargs: Any) -> str:
        max_new_tokens = int(kwargs.pop("max_tokens", 512))
        temperature = float(kwargs.pop("temperature", 0.7))
        do_sample = bool(kwargs.pop("do_sample", temperature > 0))
        top_p = float(kwargs.pop("top_p", 0.95))
        repetition_penalty = float(kwargs.pop("repetition_penalty", 1.0))
        formatted_prompt = self._render_chat_prompt(self._tokenizer, prompt)
        inputs = self._tokenizer(
            formatted_prompt, # 使用包装后的 prompt
            return_tensors="pt",
            truncation=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature if do_sample else None,
            "top_p": top_p if do_sample else None,
            "repetition_penalty": repetition_penalty,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }

        # 去掉 transformers 不接受的 None
        generate_kwargs = {k: v for k, v in generate_kwargs.items() if v is not None}

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                **generate_kwargs,
            )

        input_length = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][input_length:]
        raw_text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)

        # 🚨 2. 在返回之前，剥离思考过程！
        return self._normalize_chat_text(raw_text)
    
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
        normalized = re.sub(r"<think>\s*.*?\s*</think>\s*", "", text, flags=re.DOTALL).strip()
        return normalized or text.strip()

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

        raw = self.generate_text(
            prompt=structured_prompt,
            **kwargs,
        )

        cleaned = self._extract_json_object(raw)

        try:
            return schema.model_validate_json(cleaned)
        except Exception as exc:
            raise RuntimeError(
                f"HuggingFace structured generation failed to parse into {schema.__name__}: {exc}"
            ) from exc

    @property
    def model_name_or_path(self) -> str:
        return self._model_name_or_path

    def close(self) -> None:
        # transformers 本地模型一般不需要像 http client 那样 close
        # 这里保留接口一致性，方便上层统一处理
        pass

    @staticmethod
    def _resolve_device(device: str | None) -> str:
        if device is not None:
            return device

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _resolve_torch_dtype(dtype_name: str) -> torch.dtype:
        mapping: dict[str, torch.dtype] = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if dtype_name not in mapping:
            raise ValueError(f"Unsupported torch_dtype: {dtype_name}")
        return mapping[dtype_name]

    @staticmethod
    def _extract_json_object(text: str) -> str:
        """
        最低配 JSON 清洗：
        - 优先直接返回
        - 否则尝试截取第一个 {...} 块
        """
        stripped = text.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            return stripped

        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end != -1 and start < end:
            return stripped[start : end + 1]

        return stripped
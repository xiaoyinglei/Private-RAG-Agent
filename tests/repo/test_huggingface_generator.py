from __future__ import annotations

from typing import Any

import torch
from pytest import MonkeyPatch

from rag.providers.huggingface import generator as generator_module
from rag.providers.huggingface.generator import HuggingFaceGenerator


def test_huggingface_generator_passes_dtype_to_transformers(monkeypatch: MonkeyPatch) -> None:
    model_kwargs_seen: dict[str, Any] = {}

    class FakeTokenizer:
        pad_token: str | None = None
        eos_token = "</s>"

    class FakeModel:
        def to(self, device: str) -> None:
            assert device == "cpu"

        def eval(self) -> None:
            pass

    def fake_tokenizer_from_pretrained(_model_name_or_path: str, **_kwargs: Any) -> FakeTokenizer:
        return FakeTokenizer()

    def fake_model_from_pretrained(_model_name_or_path: str, **kwargs: Any) -> FakeModel:
        model_kwargs_seen.update(kwargs)
        return FakeModel()

    monkeypatch.setattr(
        generator_module.AutoTokenizer,
        "from_pretrained",
        fake_tokenizer_from_pretrained,
    )
    monkeypatch.setattr(
        generator_module.AutoModelForCausalLM,
        "from_pretrained",
        fake_model_from_pretrained,
    )

    HuggingFaceGenerator(
        "fake-model",
        device="cpu",
        torch_dtype="float16",
    )

    assert model_kwargs_seen["dtype"] is torch.float16
    assert "torch_dtype" not in model_kwargs_seen

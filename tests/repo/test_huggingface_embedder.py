from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from pytest import MonkeyPatch

from rag.providers.huggingface import embedder as embedder_module
from rag.providers.huggingface.embedder import BgeM3Embedder


def test_bge_m3_embedder_accepts_numpy_dense_payload(monkeypatch: MonkeyPatch) -> None:
    class FakeBGEM3FlagModel:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        @staticmethod
        def encode(*_args: object, **_kwargs: object) -> dict[str, np.ndarray]:
            return {"dense_vecs": np.array([[0.1, 0.2], [0.3, 0.4]])}

    monkeypatch.setattr(
        embedder_module,
        "_load_flagembedding_module",
        lambda: SimpleNamespace(BGEM3FlagModel=FakeBGEM3FlagModel),
    )
    monkeypatch.setattr(
        embedder_module,
        "suppress_backend_fast_tokenizer_padding_warning",
        lambda backend: backend,
    )

    provider = BgeM3Embedder(model_name_or_path="fake-bge-m3", device="cpu")

    assert provider.embed(["alpha", "beta"]) == [[0.1, 0.2], [0.3, 0.4]]

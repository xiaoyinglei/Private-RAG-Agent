from __future__ import annotations

import importlib
import logging
from collections.abc import MutableMapping, Sequence

from pathlib import Path
from typing import cast

_FAST_TOKENIZER_PADDING_WARNING = "Asking-to-pad-a-fast-tokenizer"
_LOGGER = logging.getLogger(__name__)
_DECODER_ONLY_RERANKER_MARKERS = ("qwen", "gemma", "minicpm", "llm-reranker")

def suppress_backend_fast_tokenizer_padding_warning(backend: object) -> object:
    tokenizer = getattr(backend, "tokenizer", None)
    if tokenizer is None:
        return backend
    if not _looks_like_fast_tokenizer(tokenizer):
        return backend

    deprecation_warnings = getattr(tokenizer, "deprecation_warnings", None)
    if isinstance(deprecation_warnings, MutableMapping):
        deprecation_warnings[_FAST_TOKENIZER_PADDING_WARNING] = True
    return backend


def _looks_like_fast_tokenizer(tokenizer: object) -> bool:
    if bool(getattr(tokenizer, "is_fast", False)):
        return True
    return tokenizer.__class__.__name__.endswith("Fast")


def expand_optional_path(raw: str | Path | None) -> Path | None:
    if raw is None:
        return None
    if isinstance(raw, Path):
        return raw.expanduser()
    normalized = raw.strip()
    if not normalized:
        return None
    return Path(normalized).expanduser()


def resolve_local_model_reference(model_name: str, model_path: str | Path | None) -> str:
    expanded = expand_optional_path(model_path)
    if expanded is None:
        return model_name
    return str(resolve_huggingface_snapshot_path(expanded))


def resolve_huggingface_snapshot_path(model_root: str | Path) -> Path:
    path = Path(model_root).expanduser()
    if _looks_like_model_dir(path):
        return path

    main_ref = path / "refs" / "main"
    if main_ref.exists():
        revision = main_ref.read_text(encoding="utf-8").strip()
        snapshot = path / "snapshots" / revision
        if _looks_like_model_dir(snapshot):
            return snapshot

    snapshots_root = path / "snapshots"
    if snapshots_root.exists():
        candidates = sorted(
            candidate
            for candidate in snapshots_root.iterdir()
            if candidate.is_dir() and _looks_like_model_dir(candidate)
        )
        if len(candidates) == 1:
            return candidates[0]

    return path


def _looks_like_model_dir(path: Path) -> bool:
    return (path / "config.json").exists() or (path / "tokenizer_config.json").exists()


def _patch_transformers_import_utils_for_flagembedding() -> None:
    try:
        import transformers.utils.import_utils as import_utils
    except Exception:
        return
    if hasattr(import_utils, "is_torch_fx_available"):
        return
    import_utils.is_torch_fx_available = import_utils.is_torch_available  # type: ignore[attr-defined]


def _prepare_for_model_fallback(
    tokenizer: object,
    ids: Sequence[int],
    pair_ids: Sequence[int] | None = None,
    *,
    add_special_tokens: bool = True,
    truncation: str | bool | None = None,
    max_length: int | None = None,
    padding: bool | str = False,
    return_attention_mask: bool = True,
    return_token_type_ids: bool | None = None,
    **_: object,
) -> dict[str, list[int]]:
    del padding
    first = list(ids)
    second = list(pair_ids) if pair_ids is not None else None

    def special_token_count(has_pair: bool) -> int:
        counter = getattr(tokenizer, "num_special_tokens_to_add", None)
        if callable(counter) and add_special_tokens:
            try:
                return int(counter(pair=has_pair))
            except Exception:
                return 0
        return 0

    def apply_truncation() -> tuple[list[int], list[int] | None]:
        local_first = list(first)
        local_second = list(second) if second is not None else None
        if max_length is None:
            return local_first, local_second

        budget = max(0, int(max_length) - special_token_count(local_second is not None))
        if local_second is None:
            return local_first[:budget], None

        strategy = truncation
        if strategy in (None, False):
            if len(local_first) + len(local_second) <= budget:
                return local_first, local_second
            strategy = "only_second"

        if strategy == "only_first":
            keep = max(0, budget - len(local_second))
            return local_first[:keep], local_second
        if strategy == "only_second":
            keep = max(0, budget - len(local_first))
            return local_first, local_second[:keep]

        # Fall back to longest-first style trimming.
        while len(local_first) + len(local_second) > budget and (local_first or local_second):
            if len(local_second) >= len(local_first) and local_second:
                local_second.pop()
                continue
            if local_first:
                local_first.pop()
                continue
            break
        return local_first, local_second

    def build_input_ids(local_first: list[int], local_second: list[int] | None) -> list[int]:
        if not add_special_tokens:
            return local_first + (local_second or [])

        cls_id = getattr(tokenizer, "cls_token_id", None)
        sep_id = getattr(tokenizer, "sep_token_id", None)
        bos_id = getattr(tokenizer, "bos_token_id", None)
        eos_id = getattr(tokenizer, "eos_token_id", None)
        class_name = tokenizer.__class__.__name__.lower()

        if cls_id is not None or sep_id is not None:
            prefix = [int(cls_id)] if cls_id is not None else []
            sep = [int(sep_id)] if sep_id is not None else ([int(eos_id)] if eos_id is not None else [])
            if local_second is None:
                return prefix + local_first + sep
            middle = sep + sep if "roberta" in class_name and sep else sep
            return prefix + local_first + middle + local_second + sep

        prefix = [int(bos_id)] if bos_id is not None else []
        suffix = [int(eos_id)] if eos_id is not None else []
        if local_second is None:
            return prefix + local_first + suffix
        return prefix + local_first + suffix + local_second + suffix

    truncated_first, truncated_second = apply_truncation()
    input_ids = build_input_ids(truncated_first, truncated_second)
    encoded: dict[str, list[int]] = {"input_ids": input_ids}
    if return_attention_mask:
        encoded["attention_mask"] = [1] * len(input_ids)
    if return_token_type_ids:
        if truncated_second is None:
            encoded["token_type_ids"] = [0] * len(input_ids)
        else:
            encoded["token_type_ids"] = [0] * len(input_ids)
    return encoded


def _patch_transformers_tokenizer_prepare_for_model() -> None:
    try:
        from transformers.tokenization_utils_tokenizers import TokenizersBackend
    except Exception:
        return
    if hasattr(TokenizersBackend, "prepare_for_model"):
        return

    def prepare_for_model(
        self: object,
        ids: Sequence[int],
        pair_ids: Sequence[int] | None = None,
        **kwargs: object,
    ) -> dict[str, list[int]]:
        return _prepare_for_model_fallback(self, ids, pair_ids, **kwargs)

    TokenizersBackend.prepare_for_model = prepare_for_model  # type: ignore[attr-defined]


def _load_flagembedding_module() -> object:
    _patch_transformers_import_utils_for_flagembedding()
    _patch_transformers_tokenizer_prepare_for_model()
    return cast(object, importlib.import_module("FlagEmbedding"))


def _infer_flagembedding_reranker_model_class(model_ref: str) -> str:
    normalized = model_ref.strip().lower()
    if any(marker in normalized for marker in _DECODER_ONLY_RERANKER_MARKERS):
        return "decoder-only-base"
    return "encoder-only-base"

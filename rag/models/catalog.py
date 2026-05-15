from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from rag.models.config import GenerationConfig, GenerationTaskConfig, ModelCapability, ModelSpec

_DEFAULT_CATALOG_PATH = "configs/models.yaml"


class ModelCatalog:
    """Loads model definitions from configs/models.yaml.

    This is the ONLY module that reads models.yaml.
    Business code MUST NOT construct a ModelCatalog directly —
    use resolve_runtime_config() from rag.models.runtime instead.
    """

    def __init__(
        self,
        models: dict[str, ModelSpec],
        defaults: dict[str, str],
        generation: GenerationConfig,
    ) -> None:
        self._models = models
        self._defaults = defaults
        self._generation = generation

    @classmethod
    def from_yaml(cls, path: str = _DEFAULT_CATALOG_PATH) -> ModelCatalog:
        resolved = Path(path)
        if not resolved.exists():
            raise FileNotFoundError(f"Model catalog not found: {resolved}")
        with resolved.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        if not isinstance(data, dict):
            raise ValueError(f"Invalid model catalog: expected a dict, got {type(data).__name__}")
        return cls._parse(data)

    @classmethod
    def _parse(cls, data: dict[str, Any]) -> ModelCatalog:
        raw_models = data.get("models")
        if not isinstance(raw_models, dict):
            raise ValueError("Model catalog must contain a 'models' section")
        raw_defaults = data.get("defaults")
        if not isinstance(raw_defaults, dict):
            raise ValueError("Model catalog must contain a 'defaults' section")

        models: dict[str, ModelSpec] = {}
        for alias, entry in raw_models.items():
            if not isinstance(entry, dict):
                raise ValueError(f"Model entry {alias!r} must be a dict, got {type(entry).__name__}")
            models[alias] = ModelSpec(
                alias=alias,
                capability=ModelCapability(entry["capability"]),
                provider=entry["provider"],
                model=entry["model"],
                base_url=entry.get("base_url"),
                api_key_env=entry.get("api_key_env"),
                embedding_space=entry.get("embedding_space"),
            )

        defaults = {
            "primary_model": raw_defaults.get("primary_model", ""),
            "embedding_model": raw_defaults.get("embedding_model", ""),
            "reranker_model": raw_defaults.get("reranker_model", ""),
        }

        generation = cls._parse_generation(data.get("generation"), defaults)
        return cls(models=models, defaults=defaults, generation=generation)

    @staticmethod
    def _parse_generation(raw: object, defaults: dict[str, str]) -> GenerationConfig:
        if not isinstance(raw, dict):
            return GenerationConfig()

        def _parse_task(name: str) -> GenerationTaskConfig:
            entry = raw.get(name)
            if not isinstance(entry, dict):
                return GenerationTaskConfig()
            return GenerationTaskConfig(
                model=entry.get("model"),
                max_tokens=int(entry["max_tokens"]) if "max_tokens" in entry else None,
                temperature=float(entry["temperature"]) if "temperature" in entry else None,
            )

        return GenerationConfig(
            summary=_parse_task("summary"),
            answer=_parse_task("answer"),
            planner=_parse_task("planner"),
            synthesize=_parse_task("synthesize"),
            factcheck=_parse_task("factcheck"),
        )

    @property
    def generation(self) -> GenerationConfig:
        return self._generation

    def get_model(self, alias: str) -> ModelSpec:
        spec = self._models.get(alias)
        if spec is None:
            available = ", ".join(sorted(self._models))
            raise KeyError(f"Unknown model alias: {alias!r}. Available: {available}")
        return spec

    def get_default_primary(self) -> ModelSpec:
        alias = self._defaults["primary_model"]
        if not alias:
            raise ValueError("No default primary_model configured")
        return self.get_model(alias)

    def get_default_embedding(self) -> ModelSpec:
        alias = self._defaults["embedding_model"]
        if not alias:
            raise ValueError("No default embedding_model configured")
        return self.get_model(alias)

    def get_default_reranker(self) -> ModelSpec | None:
        alias = self._defaults["reranker_model"]
        if not alias:
            return None
        return self.get_model(alias)

    def list_models(self, capability: ModelCapability | None = None) -> list[ModelSpec]:
        models = list(self._models.values())
        if capability is not None:
            models = [m for m in models if m.capability == capability]
        return sorted(models, key=lambda m: m.alias)

"""Provider adapters and model-facing services."""

from rag.providers.citation_formatter import CitationFormatter, FormattedAnswer
from rag.providers.fallback import FallbackEmbeddingRepo
from rag.providers.generation import (
    AnswerGenerationResult,
    AnswerGenerationService,
    AnswerGenerator,
    AnswerSectionPayload,
    GeneratorBinding,
    StructuredAnswerPayload,
)
from rag.providers.telemetry import TelemetryEmbedder, TelemetryGenerator, TelemetryReranker

__all__ = [
    "AnswerGenerationResult",
    "AnswerGenerationService",
    "AnswerGenerator",
    "AnswerSectionPayload",
    "CitationFormatter",
    "FallbackEmbeddingRepo",
    "FormattedAnswer",
    "GeneratorBinding",
    "StructuredAnswerPayload",
    "TelemetryEmbedder",
    "TelemetryGenerator",
    "TelemetryReranker",
]

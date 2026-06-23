"""Data models for persistent cross-session memory."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

import yaml

# ── Constants ──

MEMORY_TYPES = Literal["user", "feedback", "project", "reference"]

_INDEX_LINE_RE = re.compile(
    r"^- \[(?P<name>[^\]]+)\]\((?P<file>[^)]+)\) — (?P<desc>.+)$"
)

_FRONTMATTER_RE = re.compile(
    r"^---\n(?P<frontmatter>.*?)\n---\n(?P<body>.*)$",
    re.DOTALL,
)


# ── Memory file ──


@dataclass
class MemoryFile:
    """A single persistent memory file with YAML frontmatter."""

    name: str
    description: str
    memory_type: str  # MEMORY_TYPES literal
    content: str  # body text (without frontmatter)
    metadata: dict[str, Any] = field(default_factory=dict)

    _VALID_TYPES = frozenset({"user", "feedback", "project", "reference"})
    _MAX_NAME_LEN = 64
    _MAX_CONTENT_LEN = 10_000
    _MAX_DESC_LEN = 200

    def __post_init__(self) -> None:
        # Sanitize name: kebab-case alphanumeric
        if not self.name:
            raise ValueError("Memory name must not be empty")
        self.name = re.sub(r"[^a-z0-9_-]", "-", self.name.lower()).strip("-")[:self._MAX_NAME_LEN]
        if not self.name:
            raise ValueError("Memory name is invalid after sanitization")

        # Validate type
        if self.memory_type not in self._VALID_TYPES:
            self.memory_type = "reference"

        # Truncate description
        if len(self.description) > self._MAX_DESC_LEN:
            self.description = self.description[:self._MAX_DESC_LEN] + "..."

        # Truncate content
        if len(self.content) > self._MAX_CONTENT_LEN:
            self.content = self.content[:self._MAX_CONTENT_LEN] + "\n[truncated]"

    def to_markdown(self) -> str:
        """Serialize to Markdown with YAML frontmatter.

        Uses PyYAML's safe_dump for proper escaping of special characters.
        """
        now = datetime.now(UTC).isoformat()
        meta = {
            "name": self.name,
            "description": self.description,
            "metadata": {
                "type": self.memory_type,
                "created": self.metadata.get("created", now),
                "updated": now,
                **{k: v for k, v in self.metadata.items() if k not in ("type", "created", "updated")},
            },
        }
        frontmatter = yaml.safe_dump(meta, default_flow_style=False, allow_unicode=True)
        return f"---\n{frontmatter}---\n\n{self.content}"

    def index_line(self) -> str:
        """One-line summary for MEMORY.md index."""
        return f"- [{self.name}]({self.name}.md) — {self.description}"

    @classmethod
    def from_markdown(cls, text: str) -> MemoryFile:
        """Parse a Markdown file with YAML frontmatter."""
        match = _FRONTMATTER_RE.match(text)
        if not match:
            return cls(
                name="unnamed",
                description="",
                memory_type="reference",
                content=text.strip(),
            )

        frontmatter_text = match.group("frontmatter")
        body = match.group("body").strip()

        try:
            meta = yaml.safe_load(frontmatter_text)
            if not isinstance(meta, dict):
                meta = {}
        except yaml.YAMLError:
            meta = {}

        raw_meta = meta.get("metadata", {})
        if not isinstance(raw_meta, dict):
            raw_meta = {}

        return cls(
            name=str(meta.get("name", "unnamed")),
            description=str(meta.get("description", "")),
            memory_type=str(raw_meta.get("type", "reference")),
            content=body,
            metadata=raw_meta,
        )


# ── Memory file metadata (for listing) ──


@dataclass
class MemoryFileMeta:
    """Lightweight metadata for a memory file (no content loaded)."""

    name: str
    description: str
    memory_type: str
    created: str
    updated: str
    path: str  # relative path


# ── Index entry ──


@dataclass
class IndexEntry:
    """A single entry parsed from MEMORY.md."""

    name: str
    file: str  # filename, e.g. "user_coding_style.md"
    description: str

    @classmethod
    def parse_line(cls, line: str) -> IndexEntry | None:
        match = _INDEX_LINE_RE.match(line.strip())
        if not match:
            return None
        return cls(
            name=match.group("name"),
            file=match.group("file"),
            description=match.group("desc"),
        )


# ── Consolidation result ──


@dataclass
class ConsolidationResult:
    """Result of a memory consolidation operation."""

    action: str  # "skipped" | "consolidated"
    before_count: int = 0
    after_count: int = 0
    merged: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)


__all__ = [
    "ConsolidationResult",
    "IndexEntry",
    "MEMORY_TYPES",
    "MemoryFile",
    "MemoryFileMeta",
]

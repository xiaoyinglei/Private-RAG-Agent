"""Backward-compatible import for the built-in tool registry composition root."""

from rag.agent.builtin_registry import create_builtin_tool_registry

__all__ = ["create_builtin_tool_registry"]

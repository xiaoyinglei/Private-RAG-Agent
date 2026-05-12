from __future__ import annotations


class EmbeddingSpaceMismatchError(RuntimeError):
    """Raised when query embedding_space differs from the index embedding_space.

    Silent retrieval would produce meaningless results.
    """

    def __init__(self, query_space: str, index_space: str) -> None:
        self.query_space = query_space
        self.index_space = index_space
        super().__init__(
            f"Embedding space mismatch: query uses {query_space!r} but index was built with "
            f"{index_space!r}. Re-ingest documents with the current embedding model, "
            f"or switch back to an embedding model compatible with {index_space!r}."
        )


def assert_embedding_space_compatible(query_space: str, index_space: str) -> None:
    """Raise EmbeddingSpaceMismatchError if the two embedding spaces differ."""
    if query_space != index_space:
        raise EmbeddingSpaceMismatchError(query_space=query_space, index_space=index_space)

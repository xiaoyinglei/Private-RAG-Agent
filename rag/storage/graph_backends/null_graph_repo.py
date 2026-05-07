from __future__ import annotations

from rag.schema.graph import GraphEdge, GraphNode


class NullGraphRepo:

    def save_node(self, node: GraphNode) -> None:
        del node

    def merge_node_evidence(self, node_id: str, evidence_ids: tuple[str, ...] | list[str]) -> None:
        del node_id, evidence_ids

    def get_node(self, node_id: str) -> GraphNode | None:
        del node_id
        return None

    def list_nodes(self, *, node_type: str | None = None) -> list[GraphNode]:
        del node_type
        return []

    def list_nodes_by_alias(self, alias: str, *, node_type: str | None = None) -> list[GraphNode]:
        del alias, node_type
        return []

    def list_node_evidence_ids(self, node_id: str) -> list[str]:
        del node_id
        return []

    def save_candidate_edge(self, edge: GraphEdge) -> None:
        del edge

    def save_edge(self, edge: GraphEdge) -> None:
        del edge

    def bind_node_evidence(self, node_id: str, evidence_ids: tuple[str, ...] | list[str]) -> None:
        del node_id, evidence_ids

    def promote_candidate_edge(self, edge_id: str) -> None:
        del edge_id

    def get_edge(self, edge_id: str, *, include_candidates: bool = False) -> GraphEdge | None:
        del edge_id, include_candidates
        return None

    def list_candidate_edges(self) -> list[GraphEdge]:
        return []

    def list_edges(self) -> list[GraphEdge]:
        return []

    def delete_node(self, node_id: str) -> None:
        del node_id

    def delete_edge(self, edge_id: str, *, include_candidates: bool = True) -> None:
        del edge_id, include_candidates

    def list_edges_for_node(self, node_id: str, *, include_candidates: bool = False) -> list[GraphEdge]:
        del node_id, include_candidates
        return []

    def list_edges_for_evidence(self, evidence_id: str, *, include_candidates: bool = False) -> list[GraphEdge]:
        del evidence_id, include_candidates
        return []

    def delete_by_evidence_ids(self, evidence_ids: tuple[str, ...] | list[str]) -> tuple[list[str], list[str]]:
        del evidence_ids
        return ([], [])

    def close(self) -> None:
        return None

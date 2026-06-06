"""Alias-Graph Retrieval benchmark construction."""

from src.alias_graph.builder import build_alias_graph
from src.alias_graph.wiki_quality import check_wiki_name_quality

__all__ = ["build_alias_graph", "check_wiki_name_quality"]

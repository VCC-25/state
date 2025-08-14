"""
Graph construction module for STATE perturbation encoding.
Replaces one-hot encoding with graph-based perturbation representations.
"""

from .graph_construction import StateGraphBuilder
from .graph_types import GraphType

__all__ = ["StateGraphBuilder", "GraphType"] 
"""
Graph types for STATE perturbation encoding.
Defines the different types of graphs that can be used for perturbation encoding.
"""

from enum import Enum
from typing import List

class GraphType(Enum):
    """Available graph types for perturbation encoding."""
    
    GO = "go"  # Gene Ontology relationships
    STRING = "string"  # STRING database protein interactions
    SCGPT_DERIVED = "scgpt_derived"  # Experimental similarity from scGPT embeddings
    REACTOME = "reactome"  # Reactome pathway relationships
    DENSE = "dense"  # Fully connected graph
    
    @classmethod
    def get_all_types(cls) -> List[str]:
        """Get all available graph types as strings."""
        return [graph_type.value for graph_type in cls]
    
    @classmethod
    def is_valid(cls, graph_type: str) -> bool:
        """Check if a graph type is valid."""
        return graph_type in cls.get_all_types() 
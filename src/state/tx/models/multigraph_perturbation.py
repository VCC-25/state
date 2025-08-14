"""
Multigraph perturbation encoder for STATE.
Replaces one-hot encoding with graph-based perturbation representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)

class StateMultigraphPerturbationModel(nn.Module):
    """Multigraph perturbation encoder for STATE."""
    
    def __init__(
        self,
        graph_builder: 'StateGraphBuilder',
        input_dim: int = 64,
        hidden_dim: int = 128,
        output_dim: int = 64,
        num_layers: int = 4,
        num_heads: int = 2,
        dropout: float = 0.1,
        device: str = "cpu",
        graph_config: Dict[str, Any] = None,
    ):
        super().__init__()
        
        self.graph_builder = graph_builder
        self.device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Create multiple graphs
        self.graphs = {}
        if graph_config:
            for graph_name, graph_cfg in graph_config.items():
                graph_type = graph_cfg["type"]
                graph_args = graph_cfg.get("args", {})
                try:
                    self.graphs[graph_name] = self.graph_builder.create_graph(graph_type, graph_args)
                    logger.info(f"Created graph '{graph_name}' of type '{graph_type}'")
                except Exception as e:
                    logger.warning(f"Failed to create graph '{graph_name}': {e}")
        
        # Perturbation embeddings (replaces one-hot encoding)
        self.pert_embeddings = nn.Embedding(
            num_embeddings=len(graph_builder.pert2id),
            embedding_dim=input_dim,
            device=device
        )
        
        # GNN layers for each graph
        self.gnn_layers = nn.ModuleDict()
        for graph_name in self.graphs.keys():
            self.gnn_layers[graph_name] = self._create_gnn_layer()
        
        # Graph combination layer
        if self.graphs:
            self.graph_combiner = nn.Linear(
                len(self.graphs) * hidden_dim,
                output_dim
            )
        else:
            # Fallback to simple embedding if no graphs
            self.graph_combiner = nn.Linear(input_dim, output_dim)
        
        # Final perturbation encoder
        self.pert_encoder = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        logger.info(f"Initialized StateMultigraphPerturbationModel with {len(self.graphs)} graphs")
    
    def _create_gnn_layer(self) -> nn.Module:
        """Create GNN layer for graph processing."""
        # Simple GAT-like layer for demonstration
        # In practice, you might want to use more sophisticated GNN architectures
        return SimpleGNNLayer(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=self.dropout
        )
    
    def forward(self, pert_indices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for perturbation encoding.
        
        Args:
            pert_indices: Tensor of perturbation indices (not one-hot)
            
        Returns:
            Tensor: Graph-based perturbation representations
        """
        batch_size = pert_indices.size(0)
        
        # Get perturbation embeddings
        pert_emb = self.pert_embeddings(pert_indices)  # [batch_size, input_dim]
        
        if not self.graphs:
            # Fallback: simple embedding without graph processing
            return self.pert_encoder(pert_emb)  # [batch_size, output_dim]
        
        # Process through each graph
        graph_outputs = []
        for graph_name, (edge_index, edge_weight, num_nodes) in self.graphs.items():
            gnn_layer = self.gnn_layers[graph_name]
            
            # Process through GNN
            # Note: This is a simplified version. In practice, you'd need proper
            # graph neural network message passing
            graph_out = self._process_graph(pert_emb, edge_index, edge_weight, gnn_layer)
            graph_outputs.append(graph_out)
        
        # Combine graph outputs
        combined = torch.cat(graph_outputs, dim=-1)
        combined = self.graph_combiner(combined)
        
        # Final perturbation encoding
        pert_encoding = self.pert_encoder(combined)
        
        return pert_encoding
    
    def _process_graph(
        self, 
        pert_emb: torch.Tensor, 
        edge_index: torch.Tensor, 
        edge_weight: torch.Tensor,
        gnn_layer: nn.Module
    ) -> torch.Tensor:
        """
        Process perturbation embeddings through a graph.
        
        Args:
            pert_emb: Perturbation embeddings [batch_size, input_dim]
            edge_index: Graph edge indices [2, num_edges]
            edge_weight: Graph edge weights [num_edges]
            gnn_layer: GNN layer to process the graph
            
        Returns:
            Tensor: Graph-processed embeddings
        """
        # For now, we'll use a simplified approach
        # In practice, you'd implement proper graph neural network message passing
        
        # Simple aggregation: average embeddings of connected perturbations
        batch_size, input_dim = pert_emb.shape
        
        # Create adjacency matrix
        num_nodes = max(edge_index.max().item() + 1, len(self.graph_builder.pert2id))
        adj_matrix = torch.zeros(num_nodes, num_nodes, device=self.device)
        adj_matrix[edge_index[0], edge_index[1]] = edge_weight
        
        # Normalize adjacency matrix
        adj_matrix = F.softmax(adj_matrix, dim=-1)
        
        # Apply graph convolution (simplified)
        # In practice, you'd use proper GNN message passing
        # For now, just process the embeddings directly
        graph_out = gnn_layer(pert_emb)
        
        return graph_out
    
    def get_graph_info(self) -> Dict[str, Any]:
        """Get information about the graphs and model."""
        return {
            "num_graphs": len(self.graphs),
            "graph_names": list(self.graphs.keys()),
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "num_perturbations": len(self.graph_builder.pert2id),
            "device": self.device
        }


class SimpleGNNLayer(nn.Module):
    """Simple GNN layer for demonstration."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int = 1, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Linear transformation
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.dropout_layer = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Simple linear transformation with activation
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout_layer(x)
        return x 
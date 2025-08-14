"""
State Transition Model with Graph-based Perturbation Encoding.
Replaces one-hot based perturbation models with graph-based approach.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple
from .base import PerturbationModel
from .decoders import FinetuneVCICountsDecoder

from .utils import build_mlp, get_activation_class, get_transformer_backbone
from ..graphs.graph_construction import StateGraphBuilder

from ..graphs.graphmodule import GSPGraph
from ..pert_models.basic_gnn import GNN
from ..pert_models.exphormer import ExphormerModel
from ..pert_models.multi_graph import MultiGraph

import logging
import numpy as np
from geomloss import SamplesLoss
from .decoders_nb import NBDecoder, nb_nll

logger = logging.getLogger(__name__)

class CombinedLoss(nn.Module):
    """
    Combined Sinkhorn + Energy loss
    """
    def __init__(self, sinkhorn_weight=0.001, energy_weight=1.0, blur=0.05):
        super().__init__()
        self.sinkhorn_weight = sinkhorn_weight
        self.energy_weight = energy_weight
        self.sinkhorn_loss = SamplesLoss(loss="sinkhorn", blur=blur)
        self.energy_loss = SamplesLoss(loss="energy", blur=blur)
    
    def forward(self, pred, target):
        sinkhorn_val = self.sinkhorn_loss(pred, target)
        energy_val = self.energy_loss(pred, target)
        return self.sinkhorn_weight * sinkhorn_val + self.energy_weight * energy_val

class ConfidenceToken(nn.Module):
    """
    Learnable confidence token that gets appended to the input sequence
    and learns to predict the expected loss value.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        # Learnable confidence token embedding
        self.confidence_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # Projection head to map confidence token output to scalar loss prediction
        self.confidence_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
            nn.ReLU(),  # Ensure positive loss prediction
        )

    def append_confidence_token(self, seq_input: torch.Tensor) -> torch.Tensor:
        """
        Append confidence token to the sequence input.

        Args:
            seq_input: Input tensor of shape [B, S, E]

        Returns:
            Extended tensor of shape [B, S+1, E]
        """
        batch_size = seq_input.size(0)
        # Expand confidence token to batch size
        confidence_tokens = self.confidence_token.expand(batch_size, -1, -1)
        # Concatenate along sequence dimension
        return torch.cat([seq_input, confidence_tokens], dim=1)

    def extract_confidence_prediction(self, transformer_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract main output and confidence prediction from transformer output.

        Args:
            transformer_output: Output tensor of shape [B, S+1, E]

        Returns:
            main_output: Tensor of shape [B, S, E]
            confidence_pred: Tensor of shape [B, 1]
        """
        # Split the output
        main_output = transformer_output[:, :-1, :]  # [B, S, E]
        confidence_output = transformer_output[:, -1:, :]  # [B, 1, E]

        # Project confidence token output to scalar
        confidence_pred = self.confidence_projection(confidence_output).squeeze(-1)  # [B, 1]

        return main_output, confidence_pred

class SimpleGNNLayer(nn.Module):
    """Simple GNN layer for fallback when no GNN backend is specified."""
    
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

class GraphEncoder(nn.Module):
    """Graph perturbation encoder for STATE."""
    
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
        gnn_backend: str = "basic_gnn",  # "basic_gnn", "exphormer", "multi_graph", or None for fallback
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
        self.gnn_backend = gnn_backend
        
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
        
        # Initialize GNN backend if specified
        if self.gnn_backend:
            self.gsp_graph = self._build_gsp_graph(self.graphs, device)
            
            # Debug: Check GSPGraph structure
            if hasattr(self.gsp_graph, 'graph_dict'):
                logger.info(f"GSPGraph graph_dict keys: {list(self.gsp_graph.graph_dict.keys())}")
                logger.info(f"GSPGraph graph_dict values: {list(self.gsp_graph.graph_dict.values())}")
            else:
                logger.warning("GSPGraph has no graph_dict attribute")
            
            self.gnn = self._build_gnn_backend(self.gsp_graph, device)
            logger.info(f"Initialized {self.gnn_backend} backend")
        else:
            # Fallback to simple embeddings and GNN layers
            self.gsp_graph = None
            self.gnn = None
        
        # Perturbation embeddings (replaces one-hot encoding)
        self.pert_embeddings = nn.Embedding(
            num_embeddings=len(graph_builder.pert2id),
            embedding_dim=input_dim,
            device=device
        )
        
        # Only create fallback components if no GNN backend is specified
        if not self.gnn_backend:
            # GNN layers for each graph (fallback)
            self.gnn_layers = nn.ModuleDict()
            for graph_name in self.graphs.keys():
                self.gnn_layers[graph_name] = self._create_gnn_layer()
            
            # Graph combination layer (fallback)
            if self.graphs:
                self.graph_combiner = nn.Linear(
                    len(self.graphs) * hidden_dim,
                    output_dim
                )
            else:
                # Fallback to simple embedding if no graphs
                self.graph_combiner = nn.Linear(input_dim, output_dim)
        else:
            # When using GNN backend, these components are not needed
            self.gnn_layers = None
            self.graph_combiner = None
        
        # Final perturbation encoder
        self.pert_encoder = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        logger.info(f"Initialized StateMultigraphPerturbationModel with {len(self.graphs)} graphs")
    
    def _build_gsp_graph(self, graphs_dict, device):
        """Build GSPGraph from raw data files."""
        # Create graph configuration based on available graphs
        graph_cfg = {}
        
        # Map graph names to their corresponding raw data types
        graph_type_mapping = {
            "go_graph": "go",
            "string_graph": "string", 
            "reactome_graph": "reactome",
            "experimental_graph": "scgpt_derived"  # For experimental data
        }
        
        # Build graph configuration from the graphs_dict
        for graph_name in graphs_dict.keys():
            if graph_name in graph_type_mapping:
                graph_type = graph_type_mapping[graph_name]
                graph_cfg[graph_name] = {
                    "graph_type": graph_type,
                    "reduce2perts": True,
                    "norm_weights": False,
                    "mode": "top_20"
                }
        
        # If no graphs were configured, use a default
        if not graph_cfg:
            graph_cfg = {"go": {"graph_type": "go"}}
        
        # Debug: Check pert2id and graph_cfg
        logger.info(f"Building GSPGraph with pert2id size: {len(self.graph_builder.pert2id)}")
        logger.info(f"Graph config: {graph_cfg}")
        
        # Create gene2id mapping from pert2id for compatibility
        gene2id = self.graph_builder.pert2id.copy()
        
        gsp_graph = GSPGraph(
            pert2id=self.graph_builder.pert2id, 
            gene2id=gene2id, 
            graph_cfg=graph_cfg,
            cache_dir="graphs"
        )
        
        # Debug: Check what GSPGraph created
        if hasattr(gsp_graph, 'graph_dict'):
            logger.info(f"GSPGraph created with graph_dict keys: {list(gsp_graph.graph_dict.keys())}")
            for key, value in gsp_graph.graph_dict.items():
                logger.info(f"  {key}: {type(value)}, length: {len(value) if hasattr(value, '__len__') else 'N/A'}")
                if isinstance(value, tuple) and len(value) >= 3:
                    edge_index, edge_weight, num_nodes = value
                    logger.info(f"    Edge index shape: {edge_index.shape}, Edge weight shape: {edge_weight.shape}, Num nodes: {num_nodes}")
        else:
            logger.warning("GSPGraph has no graph_dict attribute")
        
        return gsp_graph
    
    def _build_gnn_backend(self, gsp_graph: GSPGraph, device: str):
        """Build GNN backend based on specified type."""
        if self.gnn_backend == "basic_gnn":
            # Debug: Check what's being passed to GNN
            logger.info(f"Building GNN with gsp_graph type: {type(gsp_graph)}")
            if hasattr(gsp_graph, 'graph_dict'):
                logger.info(f"GNN input graph_dict: {gsp_graph.graph_dict}")
            else:
                logger.warning("GNN input has no graph_dict")
            
            return GNN(
                graph=gsp_graph,
                layer_type="gat_v2",
                num_layers=self.num_layers,
                hidden_dim=self.hidden_dim,
                out_dim=self.output_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
                use_edge_weight=True,
                device=device,
            )
        elif self.gnn_backend == "exphormer":
            return ExphormerModel(
                graph=gsp_graph,
                layer_type="exphormer_w_mpnn",
                num_layers=self.num_layers,
                hidden_dim=self.hidden_dim,
                out_dim=self.output_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
                add_self_loops=True,
                use_edge_weight=True,
                expander_degree=3,
                device=device,
            )
        elif self.gnn_backend == "multi_graph":
            return MultiGraph(
                graph=gsp_graph,
                dropout=self.dropout,
                device=device,
                no_struct=False,
                input_dim=self.hidden_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim,
                num_hidden=self.num_layers,
                n_heads=self.num_heads,
                activation="relu",
                psi=1.0,
                edge_dim=32,
                node_dim=64,
                phi_dim=64,
            )
        else:
            raise ValueError(f"Unknown gnn_backend: {self.gnn_backend}")
    
    def _create_supra_adj(self, graphs):
        """Create supra-adjacency matrix for MultiGraph."""
        offset = 0
        rows, cols, wts = [], [], []
        for (ei, ew, n) in graphs:
            rows.append(ei[0] + offset)
            cols.append(ei[1] + offset)
            wts.append(ew)
            offset += n
        edge_index = torch.cat([torch.cat(rows), torch.cat(cols)]).view(2, -1)
        edge_weight = torch.cat(wts)
        return edge_index, edge_weight, offset
    
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
        
        # Use GNN backend if available
        if self.gnn_backend and self.gnn is not None:
            if self.gnn_backend == "basic_gnn":
                # Pick first graph or aggregate over all graphs
                name, (ei, ew, _) = next(iter(self.graphs.items()))
                # Ensure tensors are on the correct device
                device = torch.device(self.device) if isinstance(self.device, str) else self.device
                ei = ei.to(device)
                ew = ew.to(device)
                
                # Debug: Check GNN output size
                node_emb = self.gnn(ei, ew)  # [num_perts, output_dim]
                logger.info(f"GNN output shape: {node_emb.shape}, pert_indices shape: {pert_indices.shape}")
                
                # Check if we have enough embeddings
                if node_emb.shape[0] == 0:
                    logger.error(f"GNN produced 0 embeddings! num_perts: {self.gnn.num_perts}")
                    # Fallback to simple embeddings
                    pert_emb = self.pert_embeddings(pert_indices)
                    return self.pert_encoder(pert_emb)
            elif self.gnn_backend == "exphormer":
                node_emb = self.gnn()  # [num_perts, output_dim]
            elif self.gnn_backend == "multi_graph":
                # Prepare inputs for MultiGraph
                graphs = [(ei.to(self.device), ew.to(self.device), n) for (_, (ei, ew, n)) in self.graphs.items()]
                p = [[j for j in range(len(graphs)) if j != i] for i in range(len(graphs))]  # simple layer mix
                # Create supra adjacency
                supra_ei, supra_ew, total_nodes = self._create_supra_adj(graphs)
                node_emb = self.gnn(graphs, p, (supra_ei.to(self.device), supra_ew.to(self.device), total_nodes))  # [num_perts, output_dim]
            else:
                raise ValueError(f"Unknown gnn_backend: {self.gnn_backend}")
            
            # Gather rows for this batch
            return node_emb[pert_indices]  # [batch_size, output_dim]
        
        # Fallback to simple embedding approach (only when no GNN backend)
        if self.gnn_layers is not None and self.graph_combiner is not None:
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
        else:
            # No fallback available when GNN backend is specified but failed
            raise RuntimeError("GNN backend specified but failed to initialize, and no fallback available")
    
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

class GraphPerturbationModel(PerturbationModel):
    """Graph-based perturbation model for STATE.
    This model:
      1) Projects basal expression and graph perturbation encodings into a shared latent space.
      2) Uses an OT-based distributional loss (energy, sinkhorn, etc.) from geomloss.
      3) Enables cells to attend to one another, learning a set-to-set function rather than
      a sample-to-sample single-cell map.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        pert_dim: int,
        batch_dim: int = None,
        predict_residual: bool = True,
        distributional_loss: str = "energy",
        transformer_backbone_key: str = "GPT2",
        transformer_backbone_kwargs: dict = None,
        graph_builder: 'StateGraphBuilder' = None,
        graph_config: Dict[str, Any] = None,
        gnn_backend: str = "basic_gnn",  # "basic_gnn", "exphormer", "multi_graph", or None for fallback
        output_space: str = "gene",
        gene_dim: Optional[int] = None,
        device: str = "mps",
        **kwargs,
    ):
        """
        Args:
            input_dim: dimension of the input expression (e.g. number of genes or embedding dimension).
            hidden_dim: not necessarily used, but required by PerturbationModel signature.
            output_dim: dimension of the output space (genes or latent).
            pert_dim: dimension of perturbation embedding.
            graph_builder: StateGraphBuilder object.
            graph_config: dictionary of graph configurations.
            gnn_backend: backend for graph neural network.
            output_space: space of the output (genes or latent).
            gene_dim: dimension of the gene space.
            **kwargs: anything else to pass up to PerturbationModel or not used.
        """
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            pert_dim=pert_dim,
            batch_dim=batch_dim,
            output_space=output_space,
            **kwargs,
        )
        
        # Store graph-related parameters
        self.graph_builder = graph_builder
        self.graph_config = graph_config
        self.gnn_backend = gnn_backend
        self._device = device  # Use _device to avoid conflict with nn.Module.device property
        
        # Save or store relevant hyperparams
        self.predict_residual = predict_residual
        self.output_space = output_space
        self.n_encoder_layers = kwargs.get("n_encoder_layers", 2)
        self.n_decoder_layers = kwargs.get("n_decoder_layers", 2)
        self.activation_class = get_activation_class(kwargs.get("activation", "gelu"))
        self.cell_sentence_len = kwargs.get("cell_set_len", 256)
        self.decoder_loss_weight = kwargs.get("decoder_weight", 1.0)
        self.regularization = kwargs.get("regularization", 0.0)
        self.detach_decoder = kwargs.get("detach_decoder", False)

        self.transformer_backbone_key = transformer_backbone_key
        self.transformer_backbone_kwargs = transformer_backbone_kwargs
        self.transformer_backbone_kwargs["n_positions"] = self.cell_sentence_len + kwargs.get("extra_tokens", 0)

        self.distributional_loss = distributional_loss
        self.gene_dim = gene_dim
        
        # basal_projection attribute
        self.use_basal_projection = kwargs.get("use_basal_projection", True)
        
        # Build the distributional loss from geomloss
        blur = kwargs.get("blur", 0.05)
        loss_name = kwargs.get("loss", "energy")
        if loss_name == "energy":
            self.loss_fn = SamplesLoss(loss=self.distributional_loss, blur=blur)
        elif loss_name == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss_name == "se":
            sinkhorn_weight = kwargs.get("sinkhorn_weight", 0.01)  # 1/100 = 0.01
            energy_weight = kwargs.get("energy_weight", 1.0)
            self.loss_fn = CombinedLoss(sinkhorn_weight=sinkhorn_weight, energy_weight=energy_weight, blur=blur)
        elif loss_name == "sinkhorn":
            self.loss_fn = SamplesLoss(loss="sinkhorn", blur=blur)
        else:
            raise ValueError(f"Unknown loss function: {loss_name}")

        # Build the underlying neural OT network
        self._build_networks()

        # Add an optional encoder that introduces a batch variable
        self.batch_encoder = None
        self.batch_dim = None
        self.predict_mean = kwargs.get("predict_mean", False)
        
        if kwargs.get("batch_encoder", False) and batch_dim is not None:
            self.batch_encoder = nn.Embedding(
                num_embeddings=batch_dim,
                embedding_dim=hidden_dim,
            )
            self.batch_dim = batch_dim

        # if the model is outputting to counts space, apply relu
        # otherwise its in embedding space and we don't want to
        is_gene_space = kwargs["embed_key"] == "X_hvg" or kwargs["embed_key"] is None
        if is_gene_space or self.gene_decoder is None:
            self.relu = torch.nn.ReLU()

        # initialize a confidence token
        self.confidence_token = None
        self.confidence_loss_fn = None
        if kwargs.get("confidence_token", False):
            self.confidence_token = ConfidenceToken(hidden_dim=self.hidden_dim, dropout=self.dropout)
            self.confidence_loss_fn = nn.MSELoss()

        self.freeze_pert_backbone = kwargs.get("freeze_pert_backbone", False)
        if self.freeze_pert_backbone:
            modules_to_freeze = [
                self.transformer_backbone,
                self.project_out,
            ]
            for module in modules_to_freeze:
                for param in module.parameters():
                    param.requires_grad = False

        if kwargs.get("nb_decoder", False):
            self.gene_decoder = NBDecoder(
                latent_dim=self.output_dim + (self.batch_dim or 0),
                gene_dim=gene_dim,
                hidden_dims=[512, 512, 512],
                dropout=self.dropout,
            )

        control_pert = kwargs.get("control_pert", "non-targeting")
        if kwargs.get("finetune_vci_decoder", False):  # TODO: This will go very soon
            gene_names = []

            if output_space == "gene":
                # hvg's but for which dataset?
                if "DMSO_TF" in control_pert:
                    gene_names = np.load(
                        "/large_storage/ctc/userspace/aadduri/datasets/tahoe_19k_to_2k_names.npy", allow_pickle=True
                    )
                elif "non-targeting" in control_pert:
                    temp = ad.read_h5ad("/large_storage/ctc/userspace/aadduri/datasets/hvg/replogle/jurkat.h5")
                    # gene_names = temp.var.index.values
            else:
                assert output_space == "all"
                if "DMSO_TF" in control_pert:
                    gene_names = np.load(
                        "/large_storage/ctc/userspace/aadduri/datasets/tahoe_19k_names.npy", allow_pickle=True
                    )
                elif "non-targeting" in control_pert:
                    # temp = ad.read_h5ad('/scratch/ctc/ML/vci/paper_replogle/jurkat.h5')
                    # gene_names = temp.var.index.values
                    temp = ad.read_h5ad("/large_storage/ctc/userspace/aadduri/cross_dataset/replogle/jurkat.h5")
                    gene_names = temp.var.index.values

            self.gene_decoder = FinetuneVCICountsDecoder(
                genes=gene_names,
                # latent_dim=self.output_dim + (self.batch_dim or 0),
            )

        print(self)
        
        logger.info(f"Initialized GraphPerturbationModel with {len(graph_config)} graph types, {gnn_backend} backend, and {loss_name} loss")
        
        # Move model to device
        if device != "cpu":
            self.to(device)
            logger.info(f"Moved model to device: {device}")
        
        # Store device for internal use
        self._device = device

    def _build_networks(self):
        """Build the core neural network components."""
        # Initialize graph encoder for perturbation encoding
        device = self._device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        # New addition:
        self.graph_encoder = GraphEncoder(
            graph_builder=self.graph_builder,
            input_dim=self.pert_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            graph_config=self.graph_config,
            gnn_backend=self.gnn_backend,
            device=device,
        )
        
        
        self.basal_encoder = build_mlp(
            in_dim=self.input_dim,
            out_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_encoder_layers,
            dropout=self.dropout,
            activation=self.activation_class,
        )

        # Simple linear layer that maintains the input dimension
        if self.use_basal_projection:
            self.basal_encoder = build_mlp(
                in_dim=self.input_dim,
                out_dim=self.hidden_dim,
                hidden_dim=self.hidden_dim,
                n_layers=self.n_encoder_layers,
                dropout=self.dropout,
                activation=self.activation_class,
            )
        else:
            self.basal_encoder = nn.Linear(self.input_dim, self.hidden_dim)

        self.transformer_backbone, self.transformer_model_dim = get_transformer_backbone(
            self.transformer_backbone_key,
            self.transformer_backbone_kwargs,
        )

        # Project from input_dim to hidden_dim for transformer input
        # self.project_to_hidden = nn.Linear(self.input_dim, self.hidden_dim)
        
 

        # Project from input_dim to hidden_dim for transformer input
        self.project_out = build_mlp(
            in_dim=self.hidden_dim,
            out_dim=self.output_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_decoder_layers,
            dropout=self.dropout,
            activation=self.activation_class,
        )
        
        # Add final_down_then_up layer for output_space='all' (identical to state_transition.py)
        if self.output_space == 'all':
            self.final_down_then_up = nn.Sequential(
                nn.Linear(self.output_dim, self.output_dim // 8),
                nn.GELU(),
                nn.Linear(self.output_dim // 8, self.output_dim),
            )
    
        
        logger.info("Built graph perturbation model networks")

    def encode_perturbation(self, pert_indices: torch.Tensor) -> torch.Tensor:
        """Encode perturbations using graph-based approach."""
        return self.graph_encoder(pert_indices)
    
    def encode_basal_expression(self, expr: torch.Tensor) -> torch.Tensor:
        """Encode basal expression."""
        return self.basal_encoder(expr)
    
    # concatenate the basal and perturbation encodings
    def perturb(self, pert_encoding: torch.Tensor, basal_encoding: torch.Tensor) -> torch.Tensor:
        """Predict perturbation effect by combining basal and perturbation encodings."""
        # Concatenate basal and perturbation encodings
        combined = torch.cat([basal_encoding, pert_encoding], dim=-1)
        
        # Predict perturbation effect
        perturbation_effect = self.perturbation_predictor(combined)
        
        return perturbation_effect
    
    def forward(self, batch: Dict[str, torch.Tensor], padded=True) -> torch.Tensor:
        """
        The main forward call. Identical to state_transition.py except for perturbation encoding.
        
        Expects input tensors of shape (B, S, N) where:
        B = batch size
        S = sequence length (cell_sentence_len)
        N = feature dimension
        The `padded` argument here is set to True if the batch is padded. Otherwise, we
        expect a single batch, so that sentences can vary in length across batches.
        """
        if padded:
            # Handle both pert_idx (indices) and pert_emb (embeddings) formats
            if "pert_idx" in batch:
                pert_indices = batch["pert_idx"].reshape(-1, self.cell_sentence_len)
            elif "pert_emb" in batch:
                pert_emb = batch["pert_emb"]
                if pert_emb.dim() == 2 and pert_emb.shape[1] > 1:
                    pert_indices = torch.argmax(pert_emb, dim=1)
                else:
                    pert_indices = pert_emb.squeeze()
                pert_indices = pert_indices.reshape(-1, self.cell_sentence_len)
            else:
                raise KeyError("Neither 'pert_idx' nor 'pert_emb' found in batch")
            
            basal = batch["ctrl_cell_emb"].reshape(-1, self.cell_sentence_len, self.input_dim)
        else:
            # we are inferencing on a single batch, so accept variable length sentences
            if "pert_idx" in batch:
                pert_indices = batch["pert_idx"].reshape(1, -1)
            elif "pert_emb" in batch:
                pert_emb = batch["pert_emb"]
                if pert_emb.dim() == 2 and pert_emb.shape[1] > 1:
                    pert_indices = torch.argmax(pert_emb, dim=1)
                else:
                    pert_indices = pert_emb.squeeze()
                pert_indices = pert_indices.reshape(1, -1)
            else:
                raise KeyError("Neither 'pert_idx' nor 'pert_emb' found in batch")
            
            basal = batch["ctrl_cell_emb"].reshape(1, -1, self.input_dim)
        
        # Encode perturbations using graph-based approach (ONLY DIFFERENCE)
        pert_embedding = self.encode_perturbation(pert_indices) # indices -> graph embedding
        control_cells = self.encode_basal_expression(basal)
        
        # Add encodings in input_dim space, then project to hidden_dim (identical to state_transition.py)
        combined_input = pert_embedding + control_cells  # Shape: [B, S, hidden_dim]
        seq_input = combined_input  # Shape: [B, S, hidden_dim]
        
        # Add batch encoding if available (identical to state_transition.py)
        if self.batch_encoder is not None:
            # Extract batch indices (assume they are integers or convert from one-hot)
            batch_indices = batch["batch"]

            # Handle one-hot encoded batch indices
            if batch_indices.dim() > 1 and batch_indices.size(-1) == self.batch_dim:
                batch_indices = batch_indices.argmax(-1)

            # Reshape batch indices to match sequence structure
            if padded:
                batch_indices = batch_indices.reshape(-1, self.cell_sentence_len)
            else:
                batch_indices = batch_indices.reshape(1, -1)

            # Get batch embeddings and add to sequence input
            batch_embeddings = self.batch_encoder(batch_indices.long())  # Shape: [B, S, hidden_dim]
            seq_input = seq_input + batch_embeddings

        confidence_pred = None
        if self.confidence_token is not None:
            # Append confidence token: [B, S, E] -> [B, S+1, E]
            seq_input = self.confidence_token.append_confidence_token(seq_input)

        # forward pass + extract CLS last hidden state (identical to state_transition.py)
        if self.hparams.get("mask_attn", False):
            batch_size, seq_length, _ = seq_input.shape
            device = seq_input.device

            self.transformer_backbone._attn_implementation = "eager"

            # create a [1,1,S,S] mask (now S+1 if confidence token is used)
            base = torch.eye(seq_length, device=device).view(1, seq_length, seq_length)

            # repeat out to [B,H,S,S]
            attn_mask = base.repeat(batch_size, 1, 1)

            outputs = self.transformer_backbone(inputs_embeds=seq_input, attention_mask=attn_mask)
            transformer_output = outputs.last_hidden_state
        else:
            transformer_output = self.transformer_backbone(inputs_embeds=seq_input).last_hidden_state

        # Extract confidence prediction if confidence token was used
        if self.confidence_token is not None:
            res_pred, confidence_pred = self.confidence_token.extract_confidence_prediction(transformer_output)
        else:
            res_pred = transformer_output

        # add to basal if predicting residual (identical to state_transition.py)
        if self.predict_residual and self.output_space == "all":
            # Project control_cells to hidden_dim space to match res_pred
            # control_cells_hidden = self.project_to_hidden(control_cells)
            # treat the actual prediction as a residual sum to basal
            out_pred = self.project_out(res_pred) + basal
            out_pred = self.final_down_then_up(out_pred)
        elif self.predict_residual:
            out_pred = self.project_out(res_pred + control_cells)
        else:
            out_pred = self.project_out(res_pred)

        # apply relu if specified and we output to HVG space (identical to state_transition.py)
        is_gene_space = self.hparams["embed_key"] == "X_hvg" or self.hparams["embed_key"] is None
        if is_gene_space or self.gene_decoder is None:
            out_pred = self.relu(out_pred)

        output = out_pred.reshape(-1, self.output_dim)

        if confidence_pred is not None:
            return output, confidence_pred
        else:
            return output
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, padded=True) -> torch.Tensor:
        """Training step for graph-based perturbation model with advanced loss features."""
        # Forward pass
        confidence_pred = None
        if self.confidence_token is not None:
            pred, confidence_pred = self.forward(batch, padded)
        else:
            pred = self.forward(batch, padded)
        
        # Get target
        target = batch.get("pert_cell_emb", batch.get("target", None))
        if target is None:
            raise ValueError("No target found in batch")
        
        # Calculate main loss
        main_loss = self.loss_fn(pred, target)
        if hasattr(main_loss, 'nanmean'):
            main_loss = main_loss.nanmean()
        else:
            main_loss = main_loss.mean()
        
        # Log main loss
        self.log("train_loss", main_loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Log individual loss components if using combined loss
        if hasattr(self.loss_fn, 'sinkhorn_loss') and hasattr(self.loss_fn, 'energy_loss'):
            sinkhorn_component = self.loss_fn.sinkhorn_loss(pred, target)
            energy_component = self.loss_fn.energy_loss(pred, target)
            if hasattr(sinkhorn_component, 'nanmean'):
                sinkhorn_component = sinkhorn_component.nanmean()
                energy_component = energy_component.nanmean()
            else:
                sinkhorn_component = sinkhorn_component.mean()
                energy_component = energy_component.mean()
            self.log("train/sinkhorn_loss", sinkhorn_component)
            self.log("train/energy_loss", energy_component)
        
         # Process decoder if available
        decoder_loss = None
        total_loss = main_loss
        
        if self.gene_decoder is not None and "pert_cell_counts" in batch:
            gene_targets = batch["pert_cell_counts"]
            # Train decoder to map latent predictions to gene space

            if self.detach_decoder:
                # with some random change, use the true targets
                if np.random.rand() < 0.1:
                    latent_preds = target.reshape_as(pred).detach()
                else:
                    latent_preds = pred.detach()
            else:
                latent_preds = pred

            if isinstance(self.gene_decoder, NBDecoder):
                mu, theta = self.gene_decoder(latent_preds)
                gene_targets = batch["pert_cell_counts"].reshape_as(mu)
                decoder_loss = nb_nll(gene_targets, mu, theta)
            else:
                pert_cell_counts_preds = self.gene_decoder(latent_preds)
                if padded:
                    gene_targets = gene_targets.reshape(-1, self.cell_sentence_len, self.gene_decoder.gene_dim())
                else:
                    gene_targets = gene_targets.reshape(1, -1, self.gene_decoder.gene_dim())

                decoder_loss = self.loss_fn(pert_cell_counts_preds, gene_targets).mean()

            # Log decoder loss
            self.log("decoder_loss", decoder_loss)

            total_loss = total_loss + self.decoder_loss_weight * decoder_loss
        
        # Handle confidence token loss
        if confidence_pred is not None:
            # Detach main loss to prevent gradients flowing through it
            loss_target = total_loss.detach().clone().unsqueeze(0) * 10
            
            # Ensure proper shapes for confidence loss computation
            if confidence_pred.dim() == 2:  # [B, 1]
                loss_target = loss_target.unsqueeze(0).expand(confidence_pred.size(0), 1)
            else:  # confidence_pred is [B,]
                loss_target = loss_target.unsqueeze(0).expand(confidence_pred.size(0))
            
            # Compute confidence loss
            confidence_loss = self.confidence_loss_fn(confidence_pred.squeeze(), loss_target.squeeze())
            self.log("train/confidence_loss", confidence_loss)
            self.log("train/actual_loss", loss_target.mean())
            
            # Add to total loss with weighting
            confidence_weight = 0.1  # You can make this configurable
            total_loss = total_loss + confidence_weight * confidence_loss
        
        # Handle regularization
        if self.regularization > 0.0:
            ctrl_cell_emb = batch["ctrl_cell_emb"]
            delta = pred - ctrl_cell_emb
            
            # Compute l1 loss
            l1_loss = torch.abs(delta).mean()
            
            # Log the regularization loss
            self.log("train/l1_regularization", l1_loss)
            
            # Add regularization to total loss
            total_loss = total_loss + self.regularization * l1_loss
        
        
        return total_loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Validation step logic."""
        if self.confidence_token is None:
            pred, confidence_pred = self.forward(batch), None
        else:
            pred, confidence_pred = self.forward(batch)

        pred = pred.reshape(-1, self.cell_sentence_len, self.output_dim)
        target = batch["pert_cell_emb"]
        target = target.reshape(-1, self.cell_sentence_len, self.output_dim)

        loss = self.loss_fn(pred, target).mean()
        self.log("val_loss", loss)
        
        # Log individual loss components if using combined loss
        if hasattr(self.loss_fn, 'sinkhorn_loss') and hasattr(self.loss_fn, 'energy_loss'):
            sinkhorn_component = self.loss_fn.sinkhorn_loss(pred, target).mean()
            energy_component = self.loss_fn.energy_loss(pred, target).mean()
            self.log("val/sinkhorn_loss", sinkhorn_component)
            self.log("val/energy_loss", energy_component)

        if self.gene_decoder is not None and "pert_cell_counts" in batch:
            gene_targets = batch["pert_cell_counts"]

            # Get model predictions from validation step
            latent_preds = pred

            # Train decoder to map latent predictions to gene space
            if isinstance(self.gene_decoder, NBDecoder):
                mu, theta = self.gene_decoder(latent_preds)
                gene_targets = batch["pert_cell_counts"].reshape_as(mu)
                decoder_loss = nb_nll(gene_targets, mu, theta)
            else:
                # Get decoder predictions
                pert_cell_counts_preds = self.gene_decoder(latent_preds).reshape(
                    -1, self.cell_sentence_len, self.gene_decoder.gene_dim()
                )
                gene_targets = gene_targets.reshape(-1, self.cell_sentence_len, self.gene_decoder.gene_dim())
                decoder_loss = self.loss_fn(pert_cell_counts_preds, gene_targets).mean()

            # Log the validation metric
            self.log("val/decoder_loss", decoder_loss)
            loss = loss + self.decoder_loss_weight * decoder_loss

        if confidence_pred is not None:
            # Detach main loss to prevent gradients flowing through it
            loss_target = loss.detach().clone() * 10

            # Ensure proper shapes for confidence loss computation
            if confidence_pred.dim() == 2:  # [B, 1]
                loss_target = loss_target.unsqueeze(0).expand(confidence_pred.size(0), 1)
            else:  # confidence_pred is [B,]
                loss_target = loss_target.unsqueeze(0).expand(confidence_pred.size(0))

            # Compute confidence loss
            confidence_loss = self.confidence_loss_fn(confidence_pred.squeeze(), loss_target.squeeze())
            self.log("val/confidence_loss", confidence_loss)
            self.log("val/actual_loss", loss_target.mean())

        return {"loss": loss, "predictions": pred}
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        if self.confidence_token is None:
            pred, confidence_pred = self.forward(batch, padded=False), None
        else:
            pred, confidence_pred = self.forward(batch, padded=False)

        target = batch["pert_cell_emb"]
        pred = pred.reshape(1, -1, self.output_dim)
        target = target.reshape(1, -1, self.output_dim)
        loss = self.loss_fn(pred, target).mean()
        self.log("test_loss", loss)

        if confidence_pred is not None:
            # Detach main loss to prevent gradients flowing through it
            loss_target = loss.detach().clone() * 10.0

            # Ensure proper shapes for confidence loss computation
            if confidence_pred.dim() == 2:  # [B, 1]
                loss_target = loss_target.unsqueeze(0).expand(confidence_pred.size(0), 1)
            else:  # confidence_pred is [B,]
                loss_target = loss_target.unsqueeze(0).expand(confidence_pred.size(0))

            # Compute confidence loss
            confidence_loss = self.confidence_loss_fn(confidence_pred.squeeze(), loss_target.squeeze())
            self.log("test/confidence_loss", confidence_loss)
    
    def predict_step(self, batch, batch_idx, padded=True, **kwargs):
        """
        Typically used for final inference. We'll replicate old logic:s
         returning 'preds', 'X', 'pert_name', etc.
        """
        if self.confidence_token is None:
            latent_output = self.forward(batch, padded=padded)  # shape [B, ...]
            confidence_pred = None
        else:
            latent_output, confidence_pred = self.forward(batch, padded=padded)

        output_dict = {
            "preds": latent_output,
            "pert_cell_emb": batch.get("pert_cell_emb", None),
            "pert_cell_counts": batch.get("pert_cell_counts", None),
            "pert_name": batch.get("pert_name", None),
            "celltype_name": batch.get("cell_type", None),
            "batch": batch.get("batch", None),
            "ctrl_cell_emb": batch.get("ctrl_cell_emb", None),
            "pert_cell_barcode": batch.get("pert_cell_barcode", None),
            "ctrl_cell_barcode": batch.get("ctrl_cell_barcode", None),
        }

        # Add confidence prediction to output if available
        if confidence_pred is not None:
            output_dict["confidence_pred"] = confidence_pred

        if self.gene_decoder is not None:
            if isinstance(self.gene_decoder, NBDecoder):
                mu, _ = self.gene_decoder(latent_output)
                pert_cell_counts_preds = mu
            else:
                pert_cell_counts_preds = self.gene_decoder(latent_output)

            output_dict["pert_cell_counts_preds"] = pert_cell_counts_preds

        return output_dict
    def configure_optimizers(self):
        """Configure optimizers for the model."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def get_graph_info(self) -> Dict[str, Any]:
        """Get information about the graphs and model."""
        return {
            "model_type": "GraphPerturbationModel",
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "pert_dim": self.pert_dim,
            "num_graphs": len(self.graph_config),
            "graph_types": list(self.graph_config.keys()),
            "gnn_backend": self.gnn_backend,
            "device": str(self.device),
            "loss_function": str(type(self.loss_fn).__name__),
            "predict_residual": self.predict_residual,
            "regularization": self.regularization,
            "confidence_token": self.confidence_token is not None,
            "graph_encoder_info": self.graph_encoder.get_graph_info() if hasattr(self, 'graph_encoder') else None
        } 


class StateTransitionPerturbationModel(PerturbationModel):
    """
    This model:
      1) Projects basal expression and perturbation encodings into a shared latent space.
      2) Uses an OT-based distributional loss (energy, sinkhorn, etc.) from geomloss.
      3) Enables cells to attend to one another, learning a set-to-set function rather than
      a sample-to-sample single-cell map.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        pert_dim: int,
        batch_dim: int = None,
        predict_residual: bool = True,
        distributional_loss: str = "energy",
        transformer_backbone_key: str = "GPT2",
        transformer_backbone_kwargs: dict = None,
        output_space: str = "gene",
        gene_dim: Optional[int] = None,
        **kwargs,
    ):
        """
        Args:
            input_dim: dimension of the input expression (e.g. number of genes or embedding dimension).
            hidden_dim: not necessarily used, but required by PerturbationModel signature.
            output_dim: dimension of the output space (genes or latent).
            pert_dim: dimension of perturbation embedding.
            gpt: e.g. "TranslationTransformerSamplesModel".
            model_kwargs: dictionary passed to that model's constructor.
            loss: choice of distributional metric ("sinkhorn", "energy", etc.).
            **kwargs: anything else to pass up to PerturbationModel or not used.
        """
        # Call the parent PerturbationModel constructor
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            gene_dim=gene_dim,
            output_dim=output_dim,
            pert_dim=pert_dim,
            batch_dim=batch_dim,
            output_space=output_space,
            **kwargs,
        )

        # Save or store relevant hyperparams
        self.predict_residual = predict_residual
        self.output_space = output_space
        self.n_encoder_layers = kwargs.get("n_encoder_layers", 2)
        self.n_decoder_layers = kwargs.get("n_decoder_layers", 2)
        self.activation_class = get_activation_class(kwargs.get("activation", "gelu"))
        self.cell_sentence_len = kwargs.get("cell_set_len", 256)
        self.decoder_loss_weight = kwargs.get("decoder_weight", 1.0)
        self.regularization = kwargs.get("regularization", 0.0)
        self.detach_decoder = kwargs.get("detach_decoder", False)

        self.transformer_backbone_key = transformer_backbone_key
        self.transformer_backbone_kwargs = transformer_backbone_kwargs
        self.transformer_backbone_kwargs["n_positions"] = self.cell_sentence_len + kwargs.get("extra_tokens", 0)

        self.distributional_loss = distributional_loss
        self.gene_dim = gene_dim

        # Build the distributional loss from geomloss
        blur = kwargs.get("blur", 0.05)
        loss_name = kwargs.get("loss", "energy")
        if loss_name == "energy":
            self.loss_fn = SamplesLoss(loss=self.distributional_loss, blur=blur)
        elif loss_name == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss_name == "se":
            sinkhorn_weight = kwargs.get("sinkhorn_weight", 0.01)  # 1/100 = 0.01
            energy_weight = kwargs.get("energy_weight", 1.0)
            self.loss_fn = CombinedLoss(sinkhorn_weight=sinkhorn_weight, energy_weight=energy_weight, blur=blur)
        elif loss_name == "sinkhorn":
            self.loss_fn = SamplesLoss(loss="sinkhorn", blur=blur)
        else:
            raise ValueError(f"Unknown loss function: {loss_name}")

        self.use_basal_projection = kwargs.get("use_basal_projection", True)

        # Build the underlying neural OT network
        self._build_networks()

        # Add an optional encoder that introduces a batch variable
        self.batch_encoder = None
        self.batch_dim = None
        self.predict_mean = kwargs.get("predict_mean", False)
        if kwargs.get("batch_encoder", False) and batch_dim is not None:
            self.batch_encoder = nn.Embedding(
                num_embeddings=batch_dim,
                embedding_dim=hidden_dim,
            )
            self.batch_dim = batch_dim

        # if the model is outputting to counts space, apply relu
        # otherwise its in embedding space and we don't want to
        is_gene_space = kwargs["embed_key"] == "X_hvg" or kwargs["embed_key"] is None
        if is_gene_space or self.gene_decoder is None:
            self.relu = torch.nn.ReLU()

        # initialize a confidence token
        self.confidence_token = None
        self.confidence_loss_fn = None
        if kwargs.get("confidence_token", False):
            self.confidence_token = ConfidenceToken(hidden_dim=self.hidden_dim, dropout=self.dropout)
            self.confidence_loss_fn = nn.MSELoss()

        self.freeze_pert_backbone = kwargs.get("freeze_pert_backbone", False)
        if self.freeze_pert_backbone:
            modules_to_freeze = [
                self.transformer_backbone,
                self.project_out,
            ]
            for module in modules_to_freeze:
                for param in module.parameters():
                    param.requires_grad = False

        if kwargs.get("nb_decoder", False):
            self.gene_decoder = NBDecoder(
                latent_dim=self.output_dim + (self.batch_dim or 0),
                gene_dim=gene_dim,
                hidden_dims=[512, 512, 512],
                dropout=self.dropout,
            )

        control_pert = kwargs.get("control_pert", "non-targeting")
        if kwargs.get("finetune_vci_decoder", False):  # TODO: This will go very soon
            gene_names = []

            if output_space == "gene":
                # hvg's but for which dataset?
                if "DMSO_TF" in control_pert:
                    gene_names = np.load(
                        "/large_storage/ctc/userspace/aadduri/datasets/tahoe_19k_to_2k_names.npy", allow_pickle=True
                    )
                elif "non-targeting" in control_pert:
                    temp = ad.read_h5ad("/large_storage/ctc/userspace/aadduri/datasets/hvg/replogle/jurkat.h5")
                    # gene_names = temp.var.index.values
            else:
                assert output_space == "all"
                if "DMSO_TF" in control_pert:
                    gene_names = np.load(
                        "/large_storage/ctc/userspace/aadduri/datasets/tahoe_19k_names.npy", allow_pickle=True
                    )
                elif "non-targeting" in control_pert:
                    # temp = ad.read_h5ad('/scratch/ctc/ML/vci/paper_replogle/jurkat.h5')
                    # gene_names = temp.var.index.values
                    temp = ad.read_h5ad("/large_storage/ctc/userspace/aadduri/cross_dataset/replogle/jurkat.h5")
                    gene_names = temp.var.index.values

            self.gene_decoder = FinetuneVCICountsDecoder(
                genes=gene_names,
                # latent_dim=self.output_dim + (self.batch_dim or 0),
            )

        print(self)

    def _build_networks(self):
        """
        Here we instantiate the actual GPT2-based model.
        """
        self.pert_encoder = build_mlp(
            in_dim=self.pert_dim,
            out_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_encoder_layers,
            dropout=self.dropout,
            activation=self.activation_class,
        )

        # Simple linear layer that maintains the input dimension
        if self.use_basal_projection:
            self.basal_encoder = build_mlp(
                in_dim=self.input_dim,
                out_dim=self.hidden_dim,
                hidden_dim=self.hidden_dim,
                n_layers=self.n_encoder_layers,
                dropout=self.dropout,
                activation=self.activation_class,
            )
        else:
            self.basal_encoder = nn.Linear(self.input_dim, self.hidden_dim)

        self.transformer_backbone, self.transformer_model_dim = get_transformer_backbone(
            self.transformer_backbone_key,
            self.transformer_backbone_kwargs,
        )

        # Project from input_dim to hidden_dim for transformer input
        # self.project_to_hidden = nn.Linear(self.input_dim, self.hidden_dim)

        self.project_out = build_mlp(
            in_dim=self.hidden_dim,
            out_dim=self.output_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_decoder_layers,
            dropout=self.dropout,
            activation=self.activation_class,
        )

        if self.output_space == 'all':
            self.final_down_then_up = nn.Sequential(
                nn.Linear(self.output_dim, self.output_dim // 8),
                nn.GELU(),
                nn.Linear(self.output_dim // 8, self.output_dim),
            )

    def encode_perturbation(self, pert: torch.Tensor) -> torch.Tensor:
        """If needed, define how we embed the raw perturbation input."""
        return self.pert_encoder(pert)

    def encode_basal_expression(self, expr: torch.Tensor) -> torch.Tensor:
        """Define how we embed basal state input, if needed."""
        return self.basal_encoder(expr)

    def forward(self, batch: dict, padded=True) -> torch.Tensor:
        """
        The main forward call. Batch is a flattened sequence of cell sentences,
        which we reshape into sequences of length cell_sentence_len.

        Expects input tensors of shape (B, S, N) where:
        B = batch size
        S = sequence length (cell_sentence_len)
        N = feature dimension

        The `padded` argument here is set to True if the batch is padded. Otherwise, we
        expect a single batch, so that sentences can vary in length across batches.
        """
        if padded:
            pert = batch["pert_emb"].reshape(-1, self.cell_sentence_len, self.pert_dim)
            basal = batch["ctrl_cell_emb"].reshape(-1, self.cell_sentence_len, self.input_dim)
        else:
            # we are inferencing on a single batch, so accept variable length sentences
            pert = batch["pert_emb"].reshape(1, -1, self.pert_dim)
            basal = batch["ctrl_cell_emb"].reshape(1, -1, self.input_dim)

        # Shape: [B, S, input_dim]
        pert_embedding = self.encode_perturbation(pert)
        control_cells = self.encode_basal_expression(basal)

        # Add encodings in input_dim space, then project to hidden_dim
        combined_input = pert_embedding + control_cells  # Shape: [B, S, hidden_dim]
        seq_input = combined_input  # Shape: [B, S, hidden_dim]

        if self.batch_encoder is not None:
            # Extract batch indices (assume they are integers or convert from one-hot)
            batch_indices = batch["batch"]

            # Handle one-hot encoded batch indices
            if batch_indices.dim() > 1 and batch_indices.size(-1) == self.batch_dim:
                batch_indices = batch_indices.argmax(-1)

            # Reshape batch indices to match sequence structure
            if padded:
                batch_indices = batch_indices.reshape(-1, self.cell_sentence_len)
            else:
                batch_indices = batch_indices.reshape(1, -1)

            # Get batch embeddings and add to sequence input
            batch_embeddings = self.batch_encoder(batch_indices.long())  # Shape: [B, S, hidden_dim]
            seq_input = seq_input + batch_embeddings

        confidence_pred = None
        if self.confidence_token is not None:
            # Append confidence token: [B, S, E] -> [B, S+1, E]
            seq_input = self.confidence_token.append_confidence_token(seq_input)

        # forward pass + extract CLS last hidden state
        if self.hparams.get("mask_attn", False):
            batch_size, seq_length, _ = seq_input.shape
            device = seq_input.device

            self.transformer_backbone._attn_implementation = "eager"

            # create a [1,1,S,S] mask (now S+1 if confidence token is used)
            base = torch.eye(seq_length, device=device).view(1, seq_length, seq_length)

            # repeat out to [B,H,S,S]
            attn_mask = base.repeat(batch_size, 1, 1)

            outputs = self.transformer_backbone(inputs_embeds=seq_input, attention_mask=attn_mask)
            transformer_output = outputs.last_hidden_state
        else:
            transformer_output = self.transformer_backbone(inputs_embeds=seq_input).last_hidden_state

        # Extract confidence prediction if confidence token was used
        if self.confidence_token is not None:
            res_pred, confidence_pred = self.confidence_token.extract_confidence_prediction(transformer_output)
        else:
            res_pred = transformer_output

        # add to basal if predicting residual
        if self.predict_residual and self.output_space == "all":
            # Project control_cells to hidden_dim space to match res_pred
            # control_cells_hidden = self.project_to_hidden(control_cells)
            # treat the actual prediction as a residual sum to basal
            out_pred = self.project_out(res_pred) + basal
            out_pred = self.final_down_then_up(out_pred)
        elif self.predict_residual:
            out_pred = self.project_out(res_pred + control_cells)
        else:
            out_pred = self.project_out(res_pred)

        # apply relu if specified and we output to HVG space
        is_gene_space = self.hparams["embed_key"] == "X_hvg" or self.hparams["embed_key"] is None
        # logger.info(f"DEBUG: is_gene_space: {is_gene_space}")
        # logger.info(f"DEBUG: self.gene_decoder: {self.gene_decoder}")
        if is_gene_space or self.gene_decoder is None:
            out_pred = self.relu(out_pred)

        output = out_pred.reshape(-1, self.output_dim)

        if confidence_pred is not None:
            return output, confidence_pred
        else:
            return output

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, padded=True) -> torch.Tensor:
        """Training step logic for both main model and decoder."""
        # Get model predictions (in latent space)
        confidence_pred = None
        if self.confidence_token is not None:
            pred, confidence_pred = self.forward(batch, padded=padded)
        else:
            pred = self.forward(batch, padded=padded)

        target = batch["pert_cell_emb"]

        if padded:
            pred = pred.reshape(-1, self.cell_sentence_len, self.output_dim)
            target = target.reshape(-1, self.cell_sentence_len, self.output_dim)
        else:
            pred = pred.reshape(1, -1, self.output_dim)
            target = target.reshape(1, -1, self.output_dim)

        main_loss = self.loss_fn(pred, target).nanmean()
        self.log("train_loss", main_loss)
        
        # Log individual loss components if using combined loss
        if hasattr(self.loss_fn, 'sinkhorn_loss') and hasattr(self.loss_fn, 'energy_loss'):
            sinkhorn_component = self.loss_fn.sinkhorn_loss(pred, target).nanmean()
            energy_component = self.loss_fn.energy_loss(pred, target).nanmean()
            self.log("train/sinkhorn_loss", sinkhorn_component)
            self.log("train/energy_loss", energy_component)

        # Process decoder if available
        decoder_loss = None
        total_loss = main_loss

        if self.gene_decoder is not None and "pert_cell_counts" in batch:
            gene_targets = batch["pert_cell_counts"]
            # Train decoder to map latent predictions to gene space

            if self.detach_decoder:
                # with some random change, use the true targets
                if np.random.rand() < 0.1:
                    latent_preds = target.reshape_as(pred).detach()
                else:
                    latent_preds = pred.detach()
            else:
                latent_preds = pred

            if isinstance(self.gene_decoder, NBDecoder):
                mu, theta = self.gene_decoder(latent_preds)
                gene_targets = batch["pert_cell_counts"].reshape_as(mu)
                decoder_loss = nb_nll(gene_targets, mu, theta)
            else:
                pert_cell_counts_preds = self.gene_decoder(latent_preds)
                if padded:
                    gene_targets = gene_targets.reshape(-1, self.cell_sentence_len, self.gene_decoder.gene_dim())
                else:
                    gene_targets = gene_targets.reshape(1, -1, self.gene_decoder.gene_dim())

                decoder_loss = self.loss_fn(pert_cell_counts_preds, gene_targets).mean()

            # Log decoder loss
            self.log("decoder_loss", decoder_loss)

            total_loss = total_loss + self.decoder_loss_weight * decoder_loss

        if confidence_pred is not None:
            # Detach main loss to prevent gradients flowing through it
            loss_target = total_loss.detach().clone().unsqueeze(0) * 10

            # Ensure proper shapes for confidence loss computation
            if confidence_pred.dim() == 2:  # [B, 1]
                loss_target = loss_target.unsqueeze(0).expand(confidence_pred.size(0), 1)
            else:  # confidence_pred is [B,]
                loss_target = loss_target.unsqueeze(0).expand(confidence_pred.size(0))

            # Compute confidence loss
            confidence_loss = self.confidence_loss_fn(confidence_pred.squeeze(), loss_target.squeeze())
            self.log("train/confidence_loss", confidence_loss)
            self.log("train/actual_loss", loss_target.mean())

            # Add to total loss with weighting
            confidence_weight = 0.1  # You can make this configurable
            total_loss = total_loss + confidence_weight * confidence_loss

            # Add to total loss
            total_loss = total_loss + confidence_loss

        if self.regularization > 0.0:
            ctrl_cell_emb = batch["ctrl_cell_emb"].reshape_as(pred)
            delta = pred - ctrl_cell_emb

            # compute l1 loss
            l1_loss = torch.abs(delta).mean()

            # Log the regularization loss
            self.log("train/l1_regularization", l1_loss)

            # Add regularization to total loss
            total_loss = total_loss + self.regularization * l1_loss

        return total_loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Validation step logic."""
        if self.confidence_token is None:
            pred, confidence_pred = self.forward(batch), None
        else:
            pred, confidence_pred = self.forward(batch)

        pred = pred.reshape(-1, self.cell_sentence_len, self.output_dim)
        target = batch["pert_cell_emb"]
        target = target.reshape(-1, self.cell_sentence_len, self.output_dim)

        loss = self.loss_fn(pred, target).mean()
        self.log("val_loss", loss)
        
        # Log individual loss components if using combined loss
        if hasattr(self.loss_fn, 'sinkhorn_loss') and hasattr(self.loss_fn, 'energy_loss'):
            sinkhorn_component = self.loss_fn.sinkhorn_loss(pred, target).mean()
            energy_component = self.loss_fn.energy_loss(pred, target).mean()
            self.log("val/sinkhorn_loss", sinkhorn_component)
            self.log("val/energy_loss", energy_component)

        if self.gene_decoder is not None and "pert_cell_counts" in batch:
            gene_targets = batch["pert_cell_counts"]

            # Get model predictions from validation step
            latent_preds = pred

            # Train decoder to map latent predictions to gene space
            if isinstance(self.gene_decoder, NBDecoder):
                mu, theta = self.gene_decoder(latent_preds)
                gene_targets = batch["pert_cell_counts"].reshape_as(mu)
                decoder_loss = nb_nll(gene_targets, mu, theta)
            else:
                # Get decoder predictions
                pert_cell_counts_preds = self.gene_decoder(latent_preds).reshape(
                    -1, self.cell_sentence_len, self.gene_decoder.gene_dim()
                )
                gene_targets = gene_targets.reshape(-1, self.cell_sentence_len, self.gene_decoder.gene_dim())
                decoder_loss = self.loss_fn(pert_cell_counts_preds, gene_targets).mean()

            # Log the validation metric
            self.log("val/decoder_loss", decoder_loss)
            loss = loss + self.decoder_loss_weight * decoder_loss

        if confidence_pred is not None:
            # Detach main loss to prevent gradients flowing through it
            loss_target = loss.detach().clone() * 10

            # Ensure proper shapes for confidence loss computation
            if confidence_pred.dim() == 2:  # [B, 1]
                loss_target = loss_target.unsqueeze(0).expand(confidence_pred.size(0), 1)
            else:  # confidence_pred is [B,]
                loss_target = loss_target.unsqueeze(0).expand(confidence_pred.size(0))

            # Compute confidence loss
            confidence_loss = self.confidence_loss_fn(confidence_pred.squeeze(), loss_target.squeeze())
            self.log("val/confidence_loss", confidence_loss)
            self.log("val/actual_loss", loss_target.mean())

        return {"loss": loss, "predictions": pred}

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        if self.confidence_token is None:
            pred, confidence_pred = self.forward(batch, padded=False), None
        else:
            pred, confidence_pred = self.forward(batch, padded=False)

        target = batch["pert_cell_emb"]
        pred = pred.reshape(1, -1, self.output_dim)
        target = target.reshape(1, -1, self.output_dim)
        loss = self.loss_fn(pred, target).mean()
        self.log("test_loss", loss)

        if confidence_pred is not None:
            # Detach main loss to prevent gradients flowing through it
            loss_target = loss.detach().clone() * 10.0

            # Ensure proper shapes for confidence loss computation
            if confidence_pred.dim() == 2:  # [B, 1]
                loss_target = loss_target.unsqueeze(0).expand(confidence_pred.size(0), 1)
            else:  # confidence_pred is [B,]
                loss_target = loss_target.unsqueeze(0).expand(confidence_pred.size(0))

            # Compute confidence loss
            confidence_loss = self.confidence_loss_fn(confidence_pred.squeeze(), loss_target.squeeze())
            self.log("test/confidence_loss", confidence_loss)

    def predict_step(self, batch, batch_idx, padded=True, **kwargs):
        """
        Typically used for final inference. We'll replicate old logic:s
         returning 'preds', 'X', 'pert_name', etc.
        """
        if self.confidence_token is None:
            latent_output = self.forward(batch, padded=padded)  # shape [B, ...]
            confidence_pred = None
        else:
            latent_output, confidence_pred = self.forward(batch, padded=padded)

        output_dict = {
            "preds": latent_output,
            "pert_cell_emb": batch.get("pert_cell_emb", None),
            "pert_cell_counts": batch.get("pert_cell_counts", None),
            "pert_name": batch.get("pert_name", None),
            "celltype_name": batch.get("cell_type", None),
            "batch": batch.get("batch", None),
            "ctrl_cell_emb": batch.get("ctrl_cell_emb", None),
            "pert_cell_barcode": batch.get("pert_cell_barcode", None),
            "ctrl_cell_barcode": batch.get("ctrl_cell_barcode", None),
        }

        # Add confidence prediction to output if available
        if confidence_pred is not None:
            output_dict["confidence_pred"] = confidence_pred

        if self.gene_decoder is not None:
            if isinstance(self.gene_decoder, NBDecoder):
                mu, _ = self.gene_decoder(latent_output)
                pert_cell_counts_preds = mu
            else:
                pert_cell_counts_preds = self.gene_decoder(latent_output)

            output_dict["pert_cell_counts_preds"] = pert_cell_counts_preds

        return output_dict
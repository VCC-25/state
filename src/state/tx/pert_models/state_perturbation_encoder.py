"""
GNN-based perturbation encoder for STATE model replacement.

This module provides learned perturbation embeddings using biological graphs
to replace the simple one-hot encoding used in STATE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Union, Tuple

from ..graphs.graphmodule import GSPGraph
from .basic_gnn import GNN
from .exphormer import ExphormerModel
from .multi_graph import MultiGraph

# Import MMD loss (same as STATE uses)
try:
    from geomloss import SamplesLoss
except ImportError:
    print("Warning: geomloss not found. Install with: pip install geomloss")
    # Fallback MMD implementation
    class SamplesLoss:
        def __init__(self, loss="energy", blur=0.05):
            self.loss = loss
            self.blur = blur
        
        def __call__(self, pred, target):
            # Simple L2 distance as fallback
            return torch.mean((pred - target) ** 2)


def get_device():
    """Get the best available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


class StatePerturbationEncoder(nn.Module):
    """
    GNN-based perturbation encoder to replace STATE's one-hot encoder.
    
    This encoder learns perturbation embeddings using biological graphs and
    provides them in the format expected by STATE's perturbation encoder.
    """
    
    def __init__(
        self,
        graph: GSPGraph,
        pert2id: Dict[str, int],
        embedding_dim: int = 64,
        gnn_type: str = "gat_v2",
        num_layers: int = 4,
        hidden_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
        device: str = None,
        use_batch_norm: bool = True,
        activation: str = "gelu",
    ):
        super().__init__()
        
        # Auto-detect device if not specified
        if device is None:
            device = get_device()
        
        self.graph = graph
        self.pert2id = pert2id
        self.embedding_dim = embedding_dim
        self.device = device
        self.num_perts = len(pert2id)
        
        print(f"Initializing StatePerturbationEncoder on device: {device}")
        
        # GNN model for learning perturbation embeddings
        if gnn_type == "exphormer":
            self.gnn = ExphormerModel(
                graph=graph,
                layer_type="exphormer",
                num_layers=num_layers,
                hidden_dim=hidden_dim,
                out_dim=embedding_dim,
                num_heads=num_heads,
                dropout=dropout,
                device=device
            )
        elif gnn_type == "multilayer":
            self.gnn = MultiGraph(
                graph=graph,
                dropout=dropout,
                device=device,
                no_struct=False,
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                output_dim=embedding_dim,
                num_hidden=num_layers,
                n_heads=num_heads,
                activation=activation,
                psi=1.0,
                edge_dim=32,
                node_dim=64,
                phi_dim=64
            )
        else:
            # Default to GAT
            self.gnn = GNN(
                graph=graph,
                layer_type=gnn_type,
                num_layers=num_layers,
                hidden_dim=hidden_dim,
                out_dim=embedding_dim,
                num_heads=num_heads,
                dropout=dropout,
                device=device
            )
        
        # STATE-compatible perturbation encoder (4-layer MLP)
        self.state_encoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.BatchNorm1d(embedding_dim) if use_batch_norm else nn.Identity(),
            nn.Dropout(dropout),
            
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.BatchNorm1d(embedding_dim) if use_batch_norm else nn.Identity(),
            nn.Dropout(dropout),
            
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.BatchNorm1d(embedding_dim) if use_batch_norm else nn.Identity(),
            nn.Dropout(dropout),
            
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
        )
        
        # Move to device
        self.to(device)
        
        # Initialize perturbation embeddings
        self._init_perturbation_embeddings()
    
    def _init_perturbation_embeddings(self):
        """Initialize perturbation embeddings from GNN."""
        self.gnn.eval()
        with torch.no_grad():
            if hasattr(self.gnn, 'forward'):
                if hasattr(self.gnn, 'pert_embeddings'):
                    # For models with learned embeddings
                    self.pert_embeddings = self.gnn.pert_embeddings.weight.clone()
                else:
                    # For models that generate embeddings on-the-fly
                    embeddings = self.gnn()
                    self.register_buffer('pert_embeddings', embeddings)
    
    def forward(self, perturbation_ids: torch.Tensor) -> torch.Tensor:
        """
        Generate perturbation embeddings for STATE.
        
        Args:
            perturbation_ids: Tensor of shape (batch_size, num_cells) containing 
                           perturbation indices for each cell
                           
        Returns:
            perturbation_embeddings: Tensor of shape (batch_size, num_cells, embedding_dim)
                                   containing learned perturbation embeddings
        """
        batch_size, num_cells = perturbation_ids.shape
        
        # Get embeddings for each perturbation ID
        # Flatten to get all unique perturbation IDs
        unique_perts = torch.unique(perturbation_ids)
        
        # Generate embeddings using GNN
        if hasattr(self.gnn, 'pert_embeddings'):
            # Use pre-computed embeddings
            pert_emb = self.gnn.pert_embeddings(unique_perts)
        else:
            # Generate embeddings on-the-fly
            pert_emb = self.gnn()
            pert_emb = pert_emb[unique_perts]
        
        # Create mapping from perturbation ID to embedding
        pert_to_emb = {}
        for i, pert_id in enumerate(unique_perts):
            pert_to_emb[pert_id.item()] = pert_emb[i]
        
        # Map each cell's perturbation ID to its embedding
        embeddings = torch.zeros(batch_size, num_cells, self.embedding_dim, 
                               device=self.device)
        
        for b in range(batch_size):
            for c in range(num_cells):
                pert_id = perturbation_ids[b, c].item()
                if pert_id in pert_to_emb:
                    embeddings[b, c] = pert_to_emb[pert_id]
        
        # Apply STATE-compatible encoder
        # Reshape for batch processing
        orig_shape = embeddings.shape
        embeddings_flat = embeddings.view(-1, self.embedding_dim)
        
        # Apply the 4-layer MLP (STATE perturbation encoder)
        encoded_embeddings = self.state_encoder(embeddings_flat)
        
        # Reshape back to original dimensions
        encoded_embeddings = encoded_embeddings.view(orig_shape)
        
        return encoded_embeddings
    
    def get_perturbation_embeddings(self) -> torch.Tensor:
        """Get all perturbation embeddings for analysis."""
        self.gnn.eval()
        with torch.no_grad():
            if hasattr(self.gnn, 'pert_embeddings'):
                return self.gnn.pert_embeddings.weight.clone()
            else:
                return self.gnn()
    
    def update_embeddings(self):
        """Update perturbation embeddings from GNN."""
        self._init_perturbation_embeddings()


class StatePerturbationEncoderTrainer:
    """
    Trainer for the GNN-based perturbation encoder using STATE's MMD loss.
    
    This approach directly optimizes for the downstream task by using STATE's
    MMD loss instead of graph reconstruction loss.
    """
    
    def __init__(
        self,
        encoder: StatePerturbationEncoder,
        state_model=None,  # Frozen STATE model for computing MMD
        learning_rate: float = 1e-3,
        device: str = None,
        mmd_blur: float = 0.05,  # Same as STATE
        cell_set_size: int = 100,  # Default cell set size
        output_dim: int = 64,  # Default output dimension
    ):
        # Auto-detect device if not specified
        if device is None:
            device = get_device()
        
        self.encoder = encoder.to(device)
        self.state_model = state_model  # Can be None for standalone training
        self.device = device
        self.optimizer = torch.optim.Adam(self.encoder.parameters(), lr=learning_rate)
        self.mmd_loss = SamplesLoss(loss="energy", blur=mmd_blur)
        self.cell_set_size = cell_set_size
        self.output_dim = output_dim
        
        print(f"Initializing MMD-based trainer on device: {device}")
        if state_model is None:
            print("⚠️ No STATE model provided - using standalone MMD training")
    
    def train_step(self, batch=None) -> float:
        """
        Single training step using STATE's MMD loss.
        
        Args:
            batch: Optional batch data. If None, uses synthetic data for standalone training.
            
        Returns:
            loss: Training loss
        """
        self.encoder.train()
        self.optimizer.zero_grad()
        
        if batch is None:
            # Standalone training with synthetic data
            loss = self._standalone_training_step()
        else:
            # Full STATE integration training
            loss = self._state_integration_training_step(batch)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def _standalone_training_step(self) -> torch.Tensor:
        """
        Standalone training step when no STATE model is available.
        Uses synthetic perturbation effects to train the encoder.
        """
        # Generate synthetic perturbation effects
        batch_size = 32
        num_cells = self.cell_set_size
        
        # Create synthetic perturbation IDs
        pert_ids = torch.randint(0, len(self.encoder.pert2id), 
                               (batch_size, num_cells), device=self.device)
        
        # Get GNN perturbation embeddings
        pert_embeddings = self.encoder(pert_ids)  # (batch_size, num_cells, embedding_dim)
        
        # Create synthetic target effects (simulating what STATE would predict)
        # This simulates the perturbation effect on cell states
        target_effects = torch.randn(batch_size, num_cells, self.output_dim, 
                                   device=self.device)
        
        # Add perturbation-specific effects based on embeddings
        pert_effects = torch.mean(pert_embeddings, dim=1, keepdim=True)  # (batch_size, 1, embedding_dim)
        target_effects += 0.1 * pert_effects.expand(-1, num_cells, -1)
        
        # Compute MMD loss between current predictions and targets
        pred = pert_embeddings.view(-1, self.output_dim)
        target = target_effects.view(-1, self.output_dim)
        
        loss = self.mmd_loss(pred, target).mean()
        
        return loss
    
    def _state_integration_training_step(self, batch: Dict) -> torch.Tensor:
        """
        Full STATE integration training step.
        
        Args:
            batch: Dictionary containing:
                - pert_ids: Perturbation IDs
                - pert_cell_emb: Target cell embeddings
                - other STATE inputs
        """
        if self.state_model is None:
            raise ValueError("STATE model is required for integration training")
        
        # 1. Get your GNN perturbation embeddings
        pert_embeddings = self.encoder(batch["pert_ids"])
        
        # 2. Replace STATE's one-hot with your embeddings
        modified_batch = batch.copy()
        modified_batch["pert_emb"] = pert_embeddings
        
        # 3. Forward through STATE (frozen)
        with torch.no_grad():
            state_pred = self.state_model(modified_batch)
            
        # 4. Compute MMD loss between predictions and targets
        pred = state_pred.reshape(-1, self.cell_set_size, self.output_dim)
        target = batch["pert_cell_emb"].reshape(-1, self.cell_set_size, self.output_dim)
        
        loss = self.mmd_loss(pred, target).mean()
        
        return loss
    
    def validation_step(self, batch=None) -> float:
        """
        Single validation step.
        
        Returns:
            loss: Validation loss
        """
        self.encoder.eval()
        
        with torch.no_grad():
            if batch is None:
                loss = self._standalone_training_step()
            else:
                loss = self._state_integration_training_step(batch)
            
            return loss.item()
    
    def save_model(self, path: str):
        """Save the trained encoder."""
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'pert2id': self.encoder.pert2id,
            'embedding_dim': self.encoder.embedding_dim,
            'trainer_config': {
                'mmd_blur': self.mmd_loss.blur,
                'cell_set_size': self.cell_set_size,
                'output_dim': self.output_dim
            }
        }, path)
    
    @classmethod
    def load_model(cls, path: str, graph: GSPGraph, state_model=None, device: str = None):
        """Load a trained encoder."""
        if device is None:
            device = get_device()
            
        checkpoint = torch.load(path, map_location=device)
        
        encoder = StatePerturbationEncoder(
            graph=graph,
            pert2id=checkpoint['pert2id'],
            embedding_dim=checkpoint['embedding_dim'],
            device=device
        )
        
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        
        # Load trainer configuration
        trainer_config = checkpoint.get('trainer_config', {})
        trainer = cls(
            encoder=encoder,
            state_model=state_model,
            device=device,
            mmd_blur=trainer_config.get('mmd_blur', 0.05),
            cell_set_size=trainer_config.get('cell_set_size', 100),
            output_dim=trainer_config.get('output_dim', 64)
        )
        
        return trainer 
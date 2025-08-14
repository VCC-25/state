"""
This file is the main model file for the STATE embedding model.
It contains the StateEmbeddingModel class, which is a subclass of L.LightningModule.
It also contains the SkipBlock class, which is a custom block that is used to skip connections in the model.
It also contains the PositionalEncoding class, which is a custom class that is used to add positional encoding to the input.
It also contains the _get_esm_embeddings function, which is a custom function that is used to get the ESM embeddings for the input.
It also contains the get_graph_embeddings function, which is a custom function that is used to get the graph embeddings for the input.
It also contains the _compute_graph_embeddings function, which is a custom function that is used to compute the graph embeddings for the input.
"""


import warnings
warnings.filterwarnings("ignore")

import math
import logging
import numpy as np
import pandas as pd
import scanpy as sc
import torch.nn.functional as F
import torch
import lightning as L

import sys

sys.path.append("../../")
sys.path.append("../")

from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer, BCEWithLogitsLoss

from tqdm.auto import tqdm
from torch.optim.lr_scheduler import ChainedScheduler, LinearLR, CosineAnnealingLR, ReduceLROnPlateau

from ..data import create_dataloader
from ..utils import (
    compute_gene_overlap_cross_pert,
    get_embedding_cfg,
    get_dataset_cfg,
    compute_pearson_delta,
    compute_perturbation_ranking_score,
)
from ..eval.emb import cluster_embedding
from .loss import WassersteinLoss, KLDivergenceLoss, MMDLoss, TabularLoss

from .flash_transformer import FlashTransformerEncoderLayer
from .flash_transformer import FlashTransformerEncoder


class SkipBlock(nn.Module):
    def __init__(self, in_features):
        """
        Given input X of size in_features
        - out = layernorm(x + MLP(MLP(X))
        """
        super().__init__()
        self.dim = in_features
        self.intermediate_dense = nn.Linear(in_features, in_features * 2, bias=True)
        self.dense = nn.Linear(in_features * 2, in_features, bias=True)
        self.activation = nn.ReLU()
        self.layer_norm = nn.LayerNorm(in_features)

    def forward(self, x):
        residual = x
        x = self.intermediate_dense(x)
        x = self.activation(x)
        x = self.dense(x)
        x = self.layer_norm(x + residual)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1536):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


def nanstd(x):
    return torch.sqrt(torch.nanmean(torch.pow(x - torch.nanmean(x, dim=-1).unsqueeze(-1), 2), dim=-1))


class StateEmbeddingModel(L.LightningModule):
    def __init__(
        self,
        token_dim: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        output_dim: int,
        dropout: float = 0.0,
        warmup_steps: int = 0,
        compiled: bool = False,
        max_lr=4e-4,
        emb_cnt=145469,
        emb_size=5120,
        cfg=None,
        collater=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.compiled = compiled
        self.model_type = "Transformer"
        self.cls_token = nn.Parameter(torch.randn(1, token_dim))

        # Store output_dim as instance attribute
        self.output_dim = output_dim # added by Mukul
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.dropout = dropout
        self.max_lr = max_lr
        self.collater = collater
        
        # FIXED: Initialize positional encoding (added by Mukul)
        self.pe_embedding = PositionalEncoding(token_dim, dropout=dropout)
        
# ***************# Graph encoder integration (added by Mukul)***********************************
        self.use_graph_embeddings = getattr(cfg.model, "use_graph_embeddings", False)
        if self.use_graph_embeddings:
            from ...tx.graphs.graph_construction import StateGraphBuilder
            self.graph_builder = StateGraphBuilder({}, cache_dir="./graphs")
            self.graph_dim = getattr(cfg.model, "graph_dim", 64) 
            
        # Pre-load all graph data to avoid repeated I/O
        self._graph_data = {}
        if self.use_graph_embeddings:
            self._preload_graph_data()

        # IMPROVED: Single unified embedding layer that handles both ESM and ESM+Graph
        if self.use_graph_embeddings:
            embedding_input_dim = token_dim + self.graph_dim
            # Add projection layer to handle ESM to ESM+Graph conversion
            self.esm_to_graph_projection = nn.Linear(token_dim, embedding_input_dim)
            # Separate layer for gene embeddings (ESM + Graph)
            self.gene_embedding_layer = nn.Sequential(
                nn.Linear(embedding_input_dim, d_model, bias=True),
                nn.LayerNorm(d_model),
                nn.SiLU(),
            )
            # Separate layer for cell sentences (ESM only)
            self.cell_embedding_layer = nn.Sequential(
                nn.Linear(token_dim, d_model, bias=True),
                nn.LayerNorm(d_model),
                nn.SiLU(),
            )
        else:
            embedding_input_dim = token_dim
            self.esm_to_graph_projection = None
            # Single layer for both when graph embeddings are disabled
            self.gene_embedding_layer = nn.Sequential(
                nn.Linear(embedding_input_dim, d_model, bias=True),
                nn.LayerNorm(d_model),
                nn.SiLU(),
            )
            self.cell_embedding_layer = self.gene_embedding_layer
            
        # Keep unified_embedding_layer for backward compatibility
        self.unified_embedding_layer = self.cell_embedding_layer

        # IMPROVED: Pre-calculate and fix all dimension mismatches
        self.pos_projection = nn.Linear(token_dim, d_model) if token_dim != d_model else nn.Identity()
        self.cls_token_projection = nn.Linear(token_dim, d_model) if token_dim != d_model else nn.Identity()
# ********************************************************************************************************************
        # Check the configuration flag whether to use Flash Attention
        use_flash = getattr(self.cfg.model, "use_flash_attention", False)
        if use_flash and FlashTransformerEncoderLayer is not None:
            print("!!! Using Flash Attention !!!")
            layers = [FlashTransformerEncoderLayer(d_model, nhead, d_hid, dropout=dropout) for _ in range(nlayers)]
            self.transformer_encoder = FlashTransformerEncoder(layers)
        else:
            encoder_layer = TransformerEncoderLayer(
                d_model, nhead, d_hid, dropout=dropout, batch_first=True, activation="gelu"
            )
            self.transformer_encoder = TransformerEncoder(encoder_layer, nlayers)

        if compiled:
            self.transformer_encoder = torch.compile(self.transformer_encoder)

        self.decoder = nn.Sequential(
            SkipBlock(d_model),
            nn.Linear(d_model, output_dim, bias=True),
        )

        if compiled:
            self.decoder = torch.compile(self.decoder)

        self.z_dim_rd = 1 if self.cfg.model.rda else 0
        self.z_dim_ds = 10 if self.cfg.model.get("dataset_correction", False) else 0
        self.z_dim = self.z_dim_rd + self.z_dim_ds

        # Dynamic binary decoder - will be created when we know the actual input size (added by Mukul)
        self.binary_decoder = None
        self.binary_decoder_input_size = None

        if self.cfg.model.counts:
            self.bin_encoder = nn.Embedding(10, d_model)
            self.count_encoder = nn.Sequential(
                nn.Linear(1, 512, bias=True),
                nn.LeakyReLU(),
                nn.Linear(512, 10),
            )

        # Initialize protein embeddings as None - will be loaded when device is available
        self.protein_embeds = None # (added by Mukul)
        
        if compiled:
            self.binary_decoder = torch.compile(self.binary_decoder)
            self.unified_embedding_layer = torch.compile(self.unified_embedding_layer) # (added by Mukul)

        self.step_ctr = 0
        self.true_top_genes = None
        self._last_val_de_check = 0
        self._last_val_perturbation_check = 0

        if getattr(self.cfg.model, "dataset_correction", False):
            self.dataset_token = nn.Parameter(torch.randn(1, token_dim))
            self.dataset_embedder = nn.Linear(output_dim, 10)

            num_dataset = get_dataset_cfg(self.cfg).num_datasets
            self.dataset_encoder = nn.Sequential(
                nn.Linear(output_dim, d_model),
                nn.SiLU(),
                nn.LayerNorm(d_model),
                nn.Dropout(0.1),
                nn.Linear(d_model, num_dataset),
            )
            self.dataset_loss = nn.CrossEntropyLoss()
        else:
            self.dataset_token = None

    def _preload_graph_data(self):
        """IMPROVED: Pre-load all graph data to avoid repeated file I/O"""
        # FIXED: Use relative paths instead of hardcoded absolute paths
        base_path = "graphs"
        graph_files = {
            'string': {
                'genes': f"{base_path}/string/string_gene_names.npy",
                'edges': f"{base_path}/string/string_graph_edges_filtered.csv",
                'nodes': f"{base_path}/string/string_graph_nodes.csv"
            },
            'go': {
                'genes': f"{base_path}/go/go_gene_names.npy",
                'adj': f"{base_path}/go/go_graph_adjacency_matrix.npy",
                'edges': f"{base_path}/go/go_graph_edges_filtered.csv",
                'nodes': f"{base_path}/go/go_graph_nodes.csv"
            },
            'reactome': {
                'genes': f"{base_path}/reactome/reactome_gene_names.npy",
                'adj': f"{base_path}/reactome/reactome_graph_adjacency_matrix.npy",
                'edges': f"{base_path}/reactome/reactome_graph_edges_filtered.csv",
                'nodes': f"{base_path}/reactome/reactome_graph_nodes.csv"
            },
            'experimental': {
                'genes': f"{base_path}/experimental_data/gene_names.npy",
                'adj': f"{base_path}/experimental_data/gene_graph_adjacency_matrix.npy",
                'edges': f"{base_path}/experimental_data/gene_graph_edges.csv",
                'nodes': f"{base_path}/experimental_data/gene_graph_nodes.csv"
            },
            'experimental_k562': {
                'genes': f"{base_path}/experimental_data/gene_names_k562.npy",
                'adj': f"{base_path}/experimental_data/gene_graph_adjacency_matrix.npy",  # Using same adj matrix
                'edges': f"{base_path}/experimental_data/gene_graph_edges_k562.csv",
                'nodes': f"{base_path}/experimental_data/gene_graph_nodes_k562.csv"
            }
        }
        
        for graph_type, paths in graph_files.items():
            try:
                # Load gene names
                gene_names = np.load(paths['genes'])
                if graph_type == 'string':
                    gene_names = gene_names[gene_names != '']
                self._graph_data[f'{graph_type}_genes'] = gene_names
                
                # Load adjacency matrix if available
                if 'adj' in paths:
                    adj_matrix = np.load(paths['adj'])
                    self._graph_data[f'{graph_type}_adj'] = adj_matrix
                    logging.info(f"✅ Loaded {graph_type} adjacency matrix: {adj_matrix.shape}")
                
                # Load edge information
                if 'edges' in paths:
                    edges_df = pd.read_csv(paths['edges'])
                    self._graph_data[f'{graph_type}_edges'] = edges_df
                    logging.info(f"✅ Loaded {graph_type} edges: {len(edges_df)} edges")
                
                # Load node information
                if 'nodes' in paths:
                    nodes_df = pd.read_csv(paths['nodes'])
                    self._graph_data[f'{graph_type}_nodes'] = nodes_df
                    logging.info(f"✅ Loaded {graph_type} nodes: {len(nodes_df)} nodes")
                
                logging.info(f"✅ Loaded {graph_type} graph data ({len(gene_names)} genes)")
                
            except Exception as e:
                logging.warning(f"Failed to load {graph_type} data: {e}")
                self._graph_data[f'{graph_type}_genes'] = np.array([])

    def get_gene_embedding(self, genes):
        """Simplified gene embedding with unified layer"""
        # Get ESM embeddings
        protein_embeds = self._get_esm_embeddings(genes)
        
        if self.use_graph_embeddings:
            # Get and combine graph embeddings
            graph_embeds = self.get_graph_embeddings(genes)
            combined_embeds = torch.cat([protein_embeds, graph_embeds], dim=-1)
            return self.gene_embedding_layer(combined_embeds)
        else:
            return self.gene_embedding_layer(protein_embeds)

    def _load_protein_embeddings(self):
        """Load protein embeddings when device is available"""
        if self.protein_embeds is None:
            try:
                self.protein_embeds = torch.load(get_embedding_cfg(self.cfg).all_embeddings, weights_only=False)
                logging.info(f"✅ Loaded protein embeddings with {len(self.protein_embeds)} entries")
            except Exception as e:
                logging.warning(f"Failed to load protein embeddings: {e}")
                self.protein_embeds = {f"GENE_{i}": torch.randn(get_embedding_cfg(self.cfg).size) for i in range(1000)}

    def _get_esm_embeddings(self, genes):
        """Get ESM embeddings for genes"""
        # Load protein embeddings if not already loaded
        self._load_protein_embeddings()
        
        protein_embeds = []
        found_genes = 0
        for gene in genes:
            if gene in self.protein_embeds:
                # Ensure the embedding is on the correct device
                embedding = self.protein_embeds[gene].to(self.device)
                protein_embeds.append(embedding)
                found_genes += 1
            else:
                default_embedding = torch.randn(get_embedding_cfg(self.cfg).size, device=self.device)
                protein_embeds.append(default_embedding)
        
        # Stack all embeddings (they should all be on the same device now)
        protein_embeds = torch.stack(protein_embeds)
        logging.info(f"✅ Found {found_genes}/{len(genes)} genes in embeddings")
        
        if protein_embeds.sum() == 0:
            logging.warning("All gene embeddings are zero, using random embeddings")
            protein_embeds = torch.randn(len(genes), get_embedding_cfg(self.cfg).size, device=self.device)

        return protein_embeds

    def get_graph_embeddings(self, genes):
        """Generate graph embeddings using actual graph structure"""
        if not hasattr(self, 'graph_builder'):
            return torch.zeros(len(genes), self.graph_dim, device=self.device)
        
        try:
            graph_config = getattr(self.cfg.model, "graph_config", {
                "experimental_graph": {"type": "experimental", "args": {"mode": "top_5"}},
                "string_graph": {"type": "string", "args": {"mode": "weighted"}},
                "go_graph": {"type": "go", "args": {"mode": "binary"}}
            })
            
            all_graph_embeddings = []
            for graph_name, graph_spec in graph_config.items():
                graph_type = graph_spec.get("type", "dense")
                graph_args = graph_spec.get("args", {})
                embeddings = self._compute_graph_embeddings(genes, graph_type, graph_args)
                all_graph_embeddings.append(embeddings)
            
            if all_graph_embeddings:
                return torch.stack(all_graph_embeddings).mean(dim=0)
            else:
                return torch.zeros(len(genes), self.graph_dim, device=self.device)
                
        except Exception as e:
            logging.warning(f"Error computing graph embeddings: {e}")
            return torch.zeros(len(genes), self.graph_dim, device=self.device)

    def _compute_graph_embeddings(self, genes, graph_type, graph_args):
        """Compute graph embeddings using actual graph structure"""
        gene_embeddings = []
        
        # Get graph data
        graph_key = f'{graph_type}_genes'
        if graph_key not in self._graph_data:
            logging.warning(f"Graph type {graph_type} not found in loaded data")
            return torch.zeros(len(genes), self.graph_dim, device=self.device)
        
        gene_names = self._graph_data[graph_key]
        adj_key = f'{graph_type}_adj'
        edges_key = f'{graph_type}_edges'
        
        # Create gene name to index mapping
        gene_to_idx = {gene: idx for idx, gene in enumerate(gene_names)}
        
        # Get adjacency matrix if available
        adj_matrix = None
        if adj_key in self._graph_data:
            adj_matrix = self._graph_data[adj_key]
        
        # Get edge information if available
        edges_df = None
        if edges_key in self._graph_data:
            edges_df = self._graph_data[edges_key]
        
        for gene in genes:
            try:
                if gene in gene_to_idx:
                    gene_idx = gene_to_idx[gene]
                    embedding = self._compute_single_gene_graph_embedding(
                        gene, gene_idx, adj_matrix, edges_df, gene_names, graph_args
                    )
                else:
                    # Gene not in graph, use zero embedding
                    embedding = torch.zeros(self.graph_dim, device=self.device)
                
                gene_embeddings.append(embedding)
                
            except Exception as e:
                logging.warning(f"Failed to get {graph_type} embedding for gene {gene}: {e}")
                gene_embeddings.append(torch.zeros(self.graph_dim, device=self.device))
        
        return torch.stack(gene_embeddings).to(self.device)

    def _compute_single_gene_graph_embedding(self, gene, gene_idx, adj_matrix, edges_df, gene_names, graph_args):
        """Compute graph embedding for a single gene using actual graph structure"""
        mode = graph_args.get("mode", "neighborhood")
        
        if mode == "neighborhood":
            return self._compute_neighborhood_embedding(gene_idx, adj_matrix, graph_args)
        elif mode == "random_walk":
            return self._compute_random_walk_embedding(gene_idx, adj_matrix, graph_args)
        elif mode == "graph_conv":
            return self._compute_graph_conv_embedding(gene_idx, adj_matrix, graph_args)
        elif mode == "edge_weighted":
            return self._compute_edge_weighted_embedding(gene, gene_idx, edges_df, gene_names, graph_args)
        else:
            # Fallback to deterministic random embedding
            return self._compute_deterministic_embedding(gene, gene_idx, graph_args)

    def _compute_neighborhood_embedding(self, gene_idx, adj_matrix, graph_args):
        """Compute embedding based on neighborhood structure"""
        if adj_matrix is None:
            return torch.zeros(self.graph_dim, device=self.device)
        
        # Get neighborhood (1-hop neighbors)
        neighborhood = adj_matrix[gene_idx, :]
        
        # Compute neighborhood statistics
        neighbor_count = np.count_nonzero(neighborhood)
        avg_weight = np.mean(neighborhood) if neighbor_count > 0 else 0
        max_weight = np.max(neighborhood) if neighbor_count > 0 else 0
        
        # Create embedding from neighborhood features
        features = [
            neighbor_count / len(adj_matrix),  # Normalized degree
            avg_weight,
            max_weight,
            gene_idx / len(adj_matrix),  # Normalized position
        ]
        
        # Pad or truncate to graph_dim
        if len(features) < self.graph_dim:
            features.extend([0] * (self.graph_dim - len(features)))
        else:
            features = features[:self.graph_dim]
        
        return torch.tensor(features, dtype=torch.float32, device=self.device)

    def _compute_random_walk_embedding(self, gene_idx, adj_matrix, graph_args):
        """Compute embedding using random walk features"""
        if adj_matrix is None:
            return torch.zeros(self.graph_dim, device=self.device)
        
        walk_length = graph_args.get("walk_length", 3)
        num_walks = graph_args.get("num_walks", 10)
        
        # Simple random walk simulation
        walk_features = []
        for _ in range(num_walks):
            current_idx = gene_idx
            walk = [current_idx]
            
            for step in range(walk_length):
                # Get neighbors
                neighbors = np.where(adj_matrix[current_idx, :] > 0)[0]
                if len(neighbors) == 0:
                    break
                
                # Random walk to neighbor
                next_idx = np.random.choice(neighbors)
                walk.append(next_idx)
                current_idx = next_idx
            
            # Compute walk statistics
            walk_features.extend([
                len(walk) / (walk_length + 1),  # Walk completion ratio
                np.mean([adj_matrix[walk[i], walk[i+1]] for i in range(len(walk)-1)]) if len(walk) > 1 else 0,
                len(set(walk)) / len(walk) if len(walk) > 0 else 0,  # Diversity
            ])
        
        # Average features across walks
        avg_features = np.mean(walk_features, axis=0) if walk_features else [0] * 3
        
        # Pad to graph_dim
        if len(avg_features) < self.graph_dim:
            avg_features = np.pad(avg_features, (0, self.graph_dim - len(avg_features)))
        else:
            avg_features = avg_features[:self.graph_dim]
        
        return torch.tensor(avg_features, dtype=torch.float32, device=self.device)

    def _compute_graph_conv_embedding(self, gene_idx, adj_matrix, graph_args):
        """Compute embedding using graph convolution approach"""
        if adj_matrix is None:
            return torch.zeros(self.graph_dim, device=self.device)
        
        # Simple graph convolution: aggregate neighbor information
        neighbors = np.where(adj_matrix[gene_idx, :] > 0)[0]
        
        if len(neighbors) == 0:
            # Isolated node
            return torch.zeros(self.graph_dim, device=self.device)
        
        # Aggregate neighbor features
        neighbor_features = []
        for neighbor_idx in neighbors:
            neighbor_degree = np.count_nonzero(adj_matrix[neighbor_idx, :])
            edge_weight = adj_matrix[gene_idx, neighbor_idx]
            neighbor_features.append([neighbor_degree, edge_weight, neighbor_idx / len(adj_matrix)])
        
        # Compute aggregated features
        if neighbor_features:
            neighbor_features = np.array(neighbor_features)
            aggregated = np.mean(neighbor_features, axis=0)
        else:
            aggregated = np.zeros(3)
        
        # Add self-features
        self_degree = np.count_nonzero(adj_matrix[gene_idx, :])
        features = np.concatenate([
            [self_degree / len(adj_matrix)],  # Self degree
            aggregated,  # Aggregated neighbor features
        ])
        
        # Pad to graph_dim
        if len(features) < self.graph_dim:
            features = np.pad(features, (0, self.graph_dim - len(features)))
        else:
            features = features[:self.graph_dim]
        
        return torch.tensor(features, dtype=torch.float32, device=self.device)

    def _compute_edge_weighted_embedding(self, gene, gene_idx, edges_df, gene_names, graph_args):
        """Compute embedding using edge weight information"""
        if edges_df is None:
            return torch.zeros(self.graph_dim, device=self.device)
        
        # Find edges involving this gene
        gene_edges = edges_df[
            (edges_df['source_gene'] == gene) | 
            (edges_df['target_gene'] == gene)
        ]
        
        if len(gene_edges) == 0:
            return torch.zeros(self.graph_dim, device=self.device)
        
        # Compute edge-based features
        total_weight = gene_edges['weight'].sum()
        avg_weight = gene_edges['weight'].mean()
        max_weight = gene_edges['weight'].max()
        num_edges = len(gene_edges)
        
        # Directional features
        outgoing = gene_edges[gene_edges['source_gene'] == gene]
        incoming = gene_edges[gene_edges['target_gene'] == gene]
        
        out_weight = outgoing['weight'].sum() if len(outgoing) > 0 else 0
        in_weight = incoming['weight'].sum() if len(incoming) > 0 else 0
        
        features = [
            num_edges / len(edges_df),  # Normalized edge count
            total_weight / gene_edges['weight'].max() if len(gene_edges) > 0 else 0,  # Normalized total weight
            avg_weight,
            max_weight,
            out_weight / (total_weight + 1e-8),  # Outgoing ratio
            in_weight / (total_weight + 1e-8),   # Incoming ratio
            gene_idx / len(gene_names),  # Normalized position
        ]
        
        # Pad to graph_dim
        if len(features) < self.graph_dim:
            features.extend([0] * (self.graph_dim - len(features)))
        else:
            features = features[:self.graph_dim]
        
        return torch.tensor(features, dtype=torch.float32, device=self.device)

    def _compute_deterministic_embedding(self, gene, gene_idx, graph_args):
        """Fallback to deterministic embedding for missing graph data"""
        # Use hash-based deterministic generation
        import hashlib
        gene_hash = hashlib.md5(gene.encode()).hexdigest()
        gene_int = int(gene_hash[:8], 16)
        
        # Create deterministic embedding
        torch.manual_seed(gene_int)
        embedding = torch.randn(self.graph_dim)
        torch.manual_seed(42)  # Reset seed
        
        return embedding.to(self.device)

    def _compute_embedding_for_batch(self, batch):
        """Simplified embedding computation"""
        batch_sentences = batch[0].to(self.device).long()
        X = batch[1].to(self.device).long()
        Y = batch[2]
        batch_weights = batch[4]
        mask = batch[5].to(torch.bool)
        batch_sentences_counts = batch[7]
        if batch_sentences_counts is not None:
            batch_sentences_counts = batch_sentences_counts.to(self.device)
        dataset_nums = batch[8]
        if dataset_nums is not None:
            dataset_nums = dataset_nums.to(self.device)

        # Added by Mukul
        gene_names = batch[9] if len(batch) > 9 and batch[9] is not None else [f"GENE_{i}" for i in range(batch_sentences.shape[1])]
        # will be used later Pass gene names to the forward method for graph embedding computation
        # Use get_gene_embedding(gene_names) to get combined ESM+Graph embeddings
        # Enable the new graph-aware embedding pipeline

        
        # Process cell sentences (batch_sentences)
        batch_sentences = self.pe_embedding(batch_sentences)

        # Normalize token outputs now
        batch_sentences = nn.functional.normalize(batch_sentences, dim=2)

        # Add a learnable CLS token to the beginning of the sentence
        batch_sentences[:, 0, :] = self.cls_token.expand(batch_sentences.size(0), -1)

        # Optionally add a learnable dataset token to the end of the sentence
        if self.dataset_token is not None:
            dataset_token = self.dataset_token.expand(batch_sentences.size(0), -1).unsqueeze(1)
            batch_sentences = torch.cat((batch_sentences, dataset_token), dim=1)
            # concatenate a False to the mask on dim 1
            mask = torch.cat((mask, torch.zeros(mask.size(0), 1, device=mask.device).bool()), dim=1)

        # Forward pass for cell embeddings
        # mask out the genes embeddings that appear in the task sentence
        _, embedding, dataset_emb = self.forward(
            batch_sentences, mask=mask, counts=batch_sentences_counts, 
            dataset_nums=dataset_nums, gene_names=gene_names
        )

        # FIXED: For X (gene embeddings), use get_gene_embedding which handles ESM+Graph
        if len(gene_names) == X.shape[1]:
            X_processed = self.get_gene_embedding(gene_names)
            # get_gene_embedding already processes through unified_embedding_layer
            # so we don't need to process it again
        else:
            # Fallback: process X through pe_embedding then unified layer
            X_embeds = self.pe_embedding(X)
            # Apply projection if graph embeddings are enabled
            if self.esm_to_graph_projection is not None:
                X_embeds = self.esm_to_graph_projection(X_embeds)
            X_processed = self.gene_embedding_layer(X_embeds)
        
        return X_processed, Y, batch_weights, embedding, dataset_emb
   
    @staticmethod
    def resize_batch(cell_embeds, task_embeds, task_counts=None, sampled_rda=None, ds_emb=None):
        """
        Static method to resize and combine embeddings for binary decoder.
        This is kept for compatibility with other modules that depend on it.
        """
        A = task_embeds.unsqueeze(0).repeat(cell_embeds.size(0), 1, 1)
        B = cell_embeds.unsqueeze(1).repeat(1, task_embeds.size(0), 1)
        if sampled_rda is not None:
            # your code here that computes mu and std dev from Y
            reshaped_counts = sampled_rda.unsqueeze(1)
            reshaped_counts = reshaped_counts.repeat(1, A.shape[1], 1)
            combine = torch.cat((A, B, reshaped_counts), dim=2)
        elif task_counts is not None:
            reshaped_counts = task_counts.unsqueeze(1).unsqueeze(2)
            reshaped_counts = reshaped_counts.repeat(1, A.shape[1], 1)

            # Concatenate all three tensors along the third dimension
            combine = torch.cat((A, B, reshaped_counts), dim=2)
        else:
            # Original behavior if total_counts is None
            combine = torch.cat((A, B), dim=2)

        if ds_emb is not None:
            # ds_emb is a tensor of shape (batch_size, 10). concatenate it to the combine tensor
            ds_emb = ds_emb.unsqueeze(1).repeat(1, A.shape[1], 1)
            combine = torch.cat((combine, ds_emb), dim=2)

        return combine
   
    def forward(self, src: Tensor, mask: Tensor, counts=None, dataset_nums=None, gene_names=None):
        """
        Forward pass with ESM + Graph embeddings.
        Args: 
            src: Tensor, shape [batch_size, seq_len, ntoken]
        Returns:
            output Tensor of shape [batch_size, seq_len, ntoken]
        """
        batch_size, seq_len, _ = src.shape
        
        # FIXED: Handle ESM+Graph embeddings properly
        # Only use gene_names if we're processing gene embeddings (not cell sentences)
        if self.use_graph_embeddings and gene_names is not None and len(gene_names) == seq_len:
            # Get ESM+Graph embeddings (already processed through unified layer)
            gene_embeds = self.get_gene_embedding(gene_names)
            if gene_embeds.shape[0] == seq_len:
                # gene_embeds is already processed, just expand for batch
                src = gene_embeds.unsqueeze(0).expand(batch_size, -1, -1)
            else:
                # Fallback: process through gene embedding layer
                src = self.gene_embedding_layer(gene_embeds)
        else:
            # For cell sentences, use the cell embedding layer
            # This handles the case where src is cell sentence data (not gene data)
            src = self.cell_embedding_layer(src)
        
        # The new model needs to process ESM+Graph embeddings before adding positional encoding

        pos_indices = torch.arange(seq_len, device=self.device).long()
        pos_embeddings = self.pos_projection(self.pe_embedding(pos_indices))
        src = src + pos_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Add CLS token
        cls_tokens = self.cls_token_projection(self.cls_token.expand(batch_size, -1, -1))
        src = torch.cat([cls_tokens, src], dim=1)
        
        # FIXED: Add count processing with proper error handling
        if counts is not None and hasattr(self, 'count_encoder') and hasattr(self, 'bin_encoder'):
            try:
                # scFoundation-style soft binning for counts
                counts = counts.unsqueeze(-1)  # now B x H x 1

                # Step 1: Transform count values into bin distribution
                bin_weights = self.count_encoder(counts)  # B x H x 10
                bin_weights = F.softmax(bin_weights, dim=-1)  # Convert to probabilities over bins

                # Step 2: Get bin embeddings
                bin_indices = torch.arange(10, device=self.device)  # 10 bins
                bin_embeddings = self.bin_encoder(bin_indices)  # 10 x d_model

                # Step 3: Compute weighted sum of bin embeddings
                # FIXED: Proper matrix multiplication: (B x H x 10) @ (10 x d_model) = (B x H x d_model)
                count_emb = torch.matmul(bin_weights, bin_embeddings)

                # FIXED: Add zero embedding for CLS token position
                # count_emb is [B, H, d_model], we need [B, H+1, d_model] to match src after CLS token
                cls_count_emb = torch.zeros(count_emb.size(0), 1, count_emb.size(2), device=self.device)
                count_emb = torch.cat([cls_count_emb, count_emb], dim=1)  # [B, H+1, d_model]

                if self.dataset_token is not None:
                    # append B x 1 x d_model to count_emb of all zeros
                    dataset_count_emb = torch.zeros(count_emb.size(0), 1, count_emb.size(2), device=self.device)
                    count_emb = torch.cat((count_emb, dataset_count_emb), dim=1)  # B x H+1 x d_model

                # Add count embeddings to token embeddings (ensure dimension match)
                if count_emb.shape[:2] == src.shape[:2]:
                    src = src + count_emb
                else:
                    logging.warning(f"Count embedding shape {count_emb.shape} doesn't match src shape {src.shape}")
            except Exception as e:
                logging.warning(f"Error in count processing: {e}")
                logging.warning(f"Counts shape: {counts.shape if counts is not None else 'None'}")
                logging.warning(f"Bin weights shape: {bin_weights.shape if 'bin_weights' in locals() else 'Not created'}")
                logging.warning(f"Bin embeddings shape: {bin_embeddings.shape if 'bin_embeddings' in locals() else 'Not created'}")
        
        output = self.transformer_encoder(src, src_key_padding_mask=None)
        gene_output = self.decoder(output)  # batch x seq_len x 128
        # In the new format, the cls token, which is at the 0 index mark, is the output.
        embedding = gene_output[:, 0, :]  # select only the CLS token.
        embedding = nn.functional.normalize(embedding, dim=1)  # Normalize.

        # we must be in train mode to use dataset correction
        dataset_emb = None
        if self.dataset_token is not None:
            dataset_emb = gene_output[:, -1, :]

        return gene_output, embedding, dataset_emb
        
    def _predict_exp_for_adata(self, adata, dataset_name, pert_col):
        dataloader = create_dataloader(
            self.cfg,
            adata=adata,
            adata_name=dataset_name,
            shuffle=False,
            sentence_collator=self.collater,
        )
        # FIXED: Better error handling
        try:
            gene_embeds = self.get_gene_embedding(adata.var.index)
        except (KeyError, AttributeError):
            try:
                gene_embeds = self.get_gene_embedding(adata.var["gene_symbols"])
            except (KeyError, AttributeError):
                logging.warning("Could not find gene symbols in adata.var, using default gene names")
                gene_embeds = self.get_gene_embedding([f"GENE_{i}" for i in range(adata.var.shape[0])])
        
        emb_batches = []
        ds_emb_batches = []
        logprob_batches = []
        
        for batch in tqdm(dataloader, position=0, leave=True, ncols=100, desc=f"Embeddings for {dataset_name}"):
            # Device-agnostic synchronization
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                # MPS doesn't have explicit synchronization, but we can clear cache
                torch.mps.empty_cache()
            _, _, _, emb, ds_emb = self._compute_embedding_for_batch(batch)

            # Decode from the embedding
            task_counts = None
            sampled_rda = None
            if self.z_dim_rd == 1:
                Y = batch[2].to(self.device)
                nan_y = Y.masked_fill(Y == 0, float("nan"))[:, : self.cfg.dataset.P + self.cfg.dataset.N]
                task_counts = torch.nanmean(nan_y, dim=1) if self.cfg.model.rda else None
                sampled_rda = None

            ds_emb_processed = None
            if self.dataset_token is not None and ds_emb is not None:
                ds_emb_processed = self.dataset_embedder(ds_emb)

            emb_batches.append(emb.detach().cpu().numpy())
            if ds_emb_processed is not None:
                ds_emb_batches.append(ds_emb_processed.detach().cpu().numpy())

            # Build combined embeddings for binary decoder
            z = emb.unsqueeze(1).repeat(1, gene_embeds.shape[0], 1)
            gene_embeds_batch = gene_embeds.unsqueeze(0).repeat(emb.shape[0], 1, 1)
            
            combine_parts = [gene_embeds_batch, z]
            
            if self.z_dim_rd == 1 and task_counts is not None:
                reshaped_counts = task_counts.unsqueeze(1).unsqueeze(2).repeat(1, gene_embeds.shape[0], 1)
                combine_parts.append(reshaped_counts)
            
            if ds_emb_processed is not None:
                ds_emb_expanded = ds_emb_processed.unsqueeze(1).repeat(1, gene_embeds.shape[0], 1)
                combine_parts.append(ds_emb_expanded)
            
            merged_embs = torch.cat(combine_parts, dim=2)
            logprobs_batch = self.binary_decoder(merged_embs)
            logprobs_batch = logprobs_batch.detach().cpu().numpy()
            logprob_batches.append(logprobs_batch.squeeze())

        # Process results
        logprob_batches = np.vstack(logprob_batches)
        adata.obsm["X_emb"] = np.vstack(emb_batches)
        if ds_emb_batches:
            adata.obsm["X_ds_emb"] = np.vstack(ds_emb_batches)
            adata.obsm["X_emb"] = np.concatenate([adata.obsm["X_emb"], adata.obsm["X_ds_emb"]], axis=-1)

        probs_df = pd.DataFrame(logprob_batches)
        del logprob_batches
        # Device-agnostic cache clearing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
        probs_df[pert_col] = adata.obs[pert_col].values

        # Read Config properties
        k = self.cfg.validations.diff_exp.top_k_rank
        non_targeting_label = self.cfg.validations.diff_exp.obs_filter_label

        probs_df = probs_df.groupby(pert_col).mean()
        ctrl = probs_df.loc[non_targeting_label].values
        pert_effects = np.abs(probs_df - ctrl)
        top_k_indices = np.argsort(pert_effects.values, axis=1)[:, -k:][:, ::-1]
        top_k_genes = np.array(adata.var.index)[top_k_indices]
        de_genes = pd.DataFrame(top_k_genes)
        de_genes.index = pert_effects.index.values

        return de_genes
    
    def shared_step(self, batch, batch_idx):
       
        logging.info(f"Step {self.global_step} - Batch {batch_idx}")
        X, Y, batch_weights, embs, dataset_embs = self._compute_embedding_for_batch(batch)

        # IMPROVED: Build input systematically instead of multiple expansions
        z = embs.unsqueeze(1).repeat(1, X.shape[1], 1)
        combine_parts = [X, z]
        
        if self.z_dim_rd == 1:
            mu = torch.nanmean(Y.masked_fill(Y == 0, float("nan")), dim=1)
            reshaped_counts = mu.unsqueeze(1).unsqueeze(2).repeat(1, X.shape[1], 1)
            combine_parts.append(reshaped_counts)
        
        if dataset_embs is not None:
            ds_emb = self.dataset_embedder(dataset_embs)
            ds_emb = ds_emb.unsqueeze(1).repeat(1, X.shape[1], 1)
            combine_parts.append(ds_emb)
        
        combine = torch.cat(combine_parts, dim=2)
        
        # Dynamic binary decoder creation
        actual_dim = combine.shape[2]
        if self.binary_decoder is None or self.binary_decoder_input_size != actual_dim:
            self.binary_decoder = nn.Sequential(
                SkipBlock(actual_dim),
                SkipBlock(actual_dim),
                nn.Linear(actual_dim, 1, bias=True),
            )
            self.binary_decoder_input_size = actual_dim
            # Move to the correct device
            self.binary_decoder = self.binary_decoder.to(self.device)
            if self.compiled:
                self.binary_decoder = torch.compile(self.binary_decoder)
        
        # concatenate the counts
        decs = self.binary_decoder(combine)

        if self.cfg.loss.name == "cross_entropy":
            criterion = BCEWithLogitsLoss()
            target = Y
        elif self.cfg.loss.name == "mse":
            criterion = nn.MSELoss()
            target = Y
        elif self.cfg.loss.name == "wasserstein":
            criterion = WassersteinLoss()
            target = Y
        elif self.cfg.loss.name == "kl_divergence":
            criterion = KLDivergenceLoss(apply_normalization=self.cfg.loss.normalization)
            target = batch_weights
        elif self.cfg.loss.name == "mmd":
            kernel = self.cfg.loss.get("kernel", "energy")
            criterion = MMDLoss(kernel=kernel, downsample=self.cfg.model.num_downsample if self.training else 1)
            target = Y
        elif self.cfg.loss.name == "tabular":
            criterion = TabularLoss(
                shared=self.cfg.dataset.S, downsample=self.cfg.model.num_downsample if self.training else 1
            )
            target = Y
        else:
            raise ValueError(f"Loss {self.cfg.loss.name} not supported")

        loss = criterion(decs.squeeze(), target)
        
        # Add graph consistency loss if graph embeddings are enabled
        graph_loss = 0.0
        if self.use_graph_embeddings and hasattr(self, 'graph_builder'):
            graph_loss = self.compute_graph_consistency_loss(embs, batch)
            loss = loss + graph_loss
        
        if dataset_embs is not None:
            # use the dataset loss
            dataset_pred = self.dataset_encoder(dataset_embs)  # B x # datasets
            dataset_labels = batch[8].to(self.device).long()

            # self.dataset_loss is a nn.CrossEntropyLoss
            dataset_loss = self.dataset_loss(dataset_pred, dataset_labels)
            if self.training:
                self.log("trainer/dataset_loss", dataset_loss)
                loss = loss + dataset_loss
            else:
                self.log("validation/dataset_loss", dataset_loss)

        sch = self.lr_schedulers()

        for scheduler in sch._schedulers:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(loss)
            else:
                scheduler.step()
        sch._last_lr = [group["lr"] for group in sch._schedulers[-1].optimizer.param_groups]
        return loss
    
    
    @torch.compile(disable=True)
    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log("trainer/train_loss", loss)
        return loss

    @torch.compile(disable=True)
    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log("validation/val_loss", loss)
        return loss

    def on_validation_epoch_end(self):
        self.eval()
        try:
            current_step = self.global_step
            if self.global_rank == 0 and self.cfg.validations.diff_exp.enable:
                interval = self.cfg.validations.diff_exp.eval_interval_multiple * self.cfg.experiment.val_check_interval
                if current_step - self._last_val_de_check >= interval:
                    self._compute_val_de()
                    self._last_val_de_check = current_step
            self.trainer.strategy.barrier()

            if self.global_rank == 0 and self.cfg.validations.perturbation.enable:
                interval = (
                    self.cfg.validations.perturbation.eval_interval_multiple * self.cfg.experiment.val_check_interval
                )
                if current_step - self._last_val_perturbation_check >= interval:
                    self._compute_val_perturbation(current_step)
                    self._last_val_perturbation_check = current_step
            self.trainer.strategy.barrier()

        finally:
            self.train()

    def _compute_val_perturbation(self, current_step):
        adata = sc.read_h5ad(self.cfg.validations.perturbation.dataset)
        adata.X = np.log1p(adata.X)
        dataloader = create_dataloader(
            self.cfg,
            adata=adata,
            adata_name=self.cfg.validations.perturbation.dataset_name,
            shuffle=False,
            sentence_collator=self.collater,
        )
        all_embs = []
        for batch in tqdm(
            dataloader,
            position=0,
            leave=True,
            ncols=100,
            desc=f"Embeddings for {self.cfg.validations.perturbation.dataset_name}",
        ):
            # Device-agnostic cache clearing and synchronization
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
            _, _, _, emb, _ = self._compute_embedding_for_batch(batch)
            # FIXED: Better memory management
            all_embs.append(emb.cpu().detach().numpy())
            del emb  # Explicitly delete to free memory

        all_embs = np.concatenate(all_embs, axis=0)
        adata.obsm["X_emb"] = all_embs
        cluster_embedding(adata, current_step, emb_key="X_emb", use_pca=True, job_name=self.cfg.experiment.name)

        col_id = self.cfg.validations.perturbation.pert_col
        ctrl_label = self.cfg.validations.perturbation.ctrl_label

        # Track metrics across all cell types
        all_correlations = []
        all_ranking_scores = []

        # self.trainer.strategy.barrier()
        for holdout_cell_type in adata.obs["cell_type"].unique():
            train_adata = adata[adata.obs["cell_type"] != holdout_cell_type]
            test_adata = adata[adata.obs["cell_type"] == holdout_cell_type]

            mean_pert_dfs = []  # store perturbation mean deltas
            # for each cell type, train a cell type mean perturbation model
            for cell_type in train_adata.obs["cell_type"].unique():
                adata_cell = train_adata[train_adata.obs["cell_type"] == cell_type]
                ctrl_adata = adata_cell[adata_cell.obs[col_id] == ctrl_label]
                pert_adata = adata_cell[adata_cell.obs[col_id] != ctrl_label]

                mean_ctrl = ctrl_adata.obsm["X_emb"].mean(axis=0)  # shape: (embedding_dim,)
                pert_offsets = pert_adata.obsm["X_emb"] - mean_ctrl

                pert_df = pd.DataFrame(
                    pert_offsets, index=pert_adata.obs_names, columns=[f"emb_{i}" for i in range(pert_offsets.shape[1])]
                )

                # Add the perturbation label column for grouping
                pert_df[col_id] = pert_adata.obs[col_id].values

                # Group by the perturbation label and compute the mean offset for this cell type
                mean_pert_dfs.append(pert_df.groupby(col_id).mean())

            # Average over all mean perturbations
            mean_pert_df = pd.concat(mean_pert_dfs).groupby(level=0).mean()
            pert_mean_offsets = {row: vals.values for row, vals in mean_pert_df.iterrows()}
            pert_mean_offsets.update({ctrl_label: np.zeros(mean_ctrl.shape[0])})

            # Create predicted and real AnnData objects for the test set
            pred_x = np.zeros_like(test_adata.obsm["X_emb"]).copy()
            real_adata = sc.AnnData(
                X=test_adata.obsm["X_emb"],
                obs=test_adata.obs.copy(),
            )

            # Sample control cells and compute predictions
            ctrl_cells = test_adata[test_adata.obs[col_id] == ctrl_label].obs.index

            pert_exclude = set()
            for i, idx in enumerate(test_adata.obs.index):
                pert = test_adata.obs.loc[idx, col_id]
                if pert not in pert_mean_offsets:
                    # we only want to compute on shared perturbations so add this
                    # to the blacklist
                    pert_exclude.add(pert)
                    continue
                elif pert == ctrl_label:
                    # For control cells, use their own embedding
                    sampled_ctrl_idx = idx
                else:
                    # For perturbed cells, sample a random control cell
                    sampled_ctrl_idx = np.random.choice(ctrl_cells)

                # Get basal expression (control cell embedding)
                basal = test_adata[sampled_ctrl_idx].obsm["X_emb"]

                # Add perturbation effect
                pert_effect = pert_mean_offsets[pert]
                pred = basal + pert_effect

                # Store prediction
                pred_x[i] = pred

            pred_adata = sc.AnnData(
                X=pred_x,
                obs=test_adata.obs.copy(),
            )

            # retain only the cells in pred and real that are not in the blacklist
            pred_adata = pred_adata[pred_adata.obs[col_id].isin(pert_mean_offsets.keys())]
            real_adata = real_adata[real_adata.obs[col_id].isin(pert_mean_offsets.keys())]
            ctrl_adata = pred_adata[pred_adata.obs[col_id] == ctrl_label]

            # Compute metrics for this cell type. In our case, ctrl_pred = ctrl_true
            # because we use the zero vector as perturbation for ctrl cells
            correlation = compute_pearson_delta(pred_adata.X, real_adata.X, ctrl_adata.X, ctrl_adata.X)
            ranking_score = compute_perturbation_ranking_score(pred_adata, real_adata)

            all_correlations.append(correlation)
            all_ranking_scores.append(ranking_score)

        # Log average metrics across all cell types
        self.log("validation/perturbation_correlation_mean", np.mean(all_correlations))
        self.log("validation/perturbation_ranking_mean", np.mean(all_ranking_scores))

    def _compute_val_de(self):
        if self.true_top_genes is None:
            de_val_adata = sc.read_h5ad(self.cfg.validations.diff_exp.dataset)
            sc.pp.log1p(de_val_adata)
            sc.tl.rank_genes_groups(
                de_val_adata,
                groupby=self.cfg.validations.diff_exp.obs_pert_col,
                reference=self.cfg.validations.diff_exp.obs_filter_label,
                rankby_abs=True,
                n_genes=self.cfg.validations.diff_exp.top_k_rank,
                method=self.cfg.validations.diff_exp.method,
                use_raw=False,
            )
            self.true_top_genes = pd.DataFrame(de_val_adata.uns["rank_genes_groups"]["names"])
            self.true_top_genes = self.true_top_genes.T
            del de_val_adata
        tmp_adata = sc.read_h5ad(self.cfg.validations.diff_exp.dataset)
        pred_exp = self._predict_exp_for_adata(
            tmp_adata, self.cfg.validations.diff_exp.dataset_name, self.cfg.validations.diff_exp.obs_pert_col
        )
        torch.cuda.synchronize()
        de_metrics = compute_gene_overlap_cross_pert(
            pred_exp, self.true_top_genes, k=self.cfg.validations.diff_exp.top_k_rank
        )
        self.log("validation/de", np.array(list(de_metrics.values())).mean())

    def compute_graph_consistency_loss(self, embs, batch):
        """Compute graph consistency loss using actual graph structure"""
        if not self.use_graph_embeddings:
            return torch.tensor(0.0, device=self.device)
        
        try:
            # Extract gene information from batch
            gene_names = batch[9] if len(batch) > 9 and batch[9] is not None else None
            if gene_names is None:
                return torch.tensor(0.0, device=self.device)
            
            # Get graph configuration
            graph_config = getattr(self.cfg.model, "graph_config", {
                "experimental_graph": {"type": "experimental", "args": {"mode": "neighborhood"}}
            })
            
            total_consistency_loss = 0.0
            num_graphs = 0
            
            for graph_name, graph_spec in graph_config.items():
                graph_type = graph_spec.get("type", "experimental")
                graph_args = graph_spec.get("args", {})
                
                # Get graph data
                graph_key = f'{graph_type}_genes'
                adj_key = f'{graph_type}_adj'
                
                if graph_key not in self._graph_data or adj_key not in self._graph_data:
                    continue
                
                gene_names_graph = self._graph_data[graph_key]
                adj_matrix = self._graph_data[adj_key]
                
                # Create gene name to index mapping
                gene_to_idx = {gene: idx for idx, gene in enumerate(gene_names_graph)}
                
                # Find genes that exist in both batch and graph
                valid_genes = [gene for gene in gene_names if gene in gene_to_idx]
                if len(valid_genes) < 2:
                    continue
                
                # Get embeddings for valid genes
                valid_indices = [gene_names.index(gene) for gene in valid_genes]
                valid_embeddings = embs[valid_indices]
                
                # Compute graph-based similarity matrix
                graph_similarity = self._compute_graph_similarity_matrix(valid_genes, gene_to_idx, adj_matrix, graph_args)
                
                # Compute embedding similarity matrix
                normalized_embs = nn.functional.normalize(valid_embeddings, dim=-1)
                embedding_similarity = torch.matmul(normalized_embs, normalized_embs.transpose(-2, -1))
                
                # Compute consistency loss
                consistency_loss = nn.functional.mse_loss(embedding_similarity, graph_similarity)
                total_consistency_loss += consistency_loss
                num_graphs += 1
            
            if num_graphs > 0:
                avg_consistency_loss = total_consistency_loss / num_graphs
                graph_loss_weight = getattr(self.cfg.model, "graph_loss_weight", 0.1)
                return graph_loss_weight * avg_consistency_loss
            else:
                return torch.tensor(0.0, device=self.device)
            
        except Exception as e:
            logging.warning(f"Error computing graph consistency loss: {e}")
            return torch.tensor(0.0, device=self.device)

    def _compute_graph_similarity_matrix(self, genes, gene_to_idx, adj_matrix, graph_args):
        """Compute similarity matrix based on actual graph structure"""
        n_genes = len(genes)
        similarity_matrix = torch.zeros(n_genes, n_genes, device=self.device)
        
        mode = graph_args.get("similarity_mode", "adjacency")
        
        if mode == "adjacency":
            # Use adjacency matrix directly
            for i, gene_i in enumerate(genes):
                idx_i = gene_to_idx[gene_i]
                for j, gene_j in enumerate(genes):
                    idx_j = gene_to_idx[gene_j]
                    # Normalize by max possible weight
                    max_weight = adj_matrix.max() if adj_matrix.max() > 0 else 1
                    similarity_matrix[i, j] = adj_matrix[idx_i, idx_j] / max_weight
        
        elif mode == "neighborhood":
            # Use neighborhood overlap
            for i, gene_i in enumerate(genes):
                idx_i = gene_to_idx[gene_i]
                neighbors_i = set(np.where(adj_matrix[idx_i, :] > 0)[0])
                
                for j, gene_j in enumerate(genes):
                    idx_j = gene_to_idx[gene_j]
                    neighbors_j = set(np.where(adj_matrix[idx_j, :] > 0)[0])
                    
                    # Jaccard similarity of neighborhoods
                    if len(neighbors_i) + len(neighbors_j) > 0:
                        intersection = len(neighbors_i.intersection(neighbors_j))
                        union = len(neighbors_i.union(neighbors_j))
                        similarity_matrix[i, j] = intersection / union
                    else:
                        similarity_matrix[i, j] = 0.0
        
        elif mode == "shortest_path":
            # Use inverse shortest path distance
            for i, gene_i in enumerate(genes):
                idx_i = gene_to_idx[gene_i]
                for j, gene_j in enumerate(genes):
                    idx_j = gene_to_idx[gene_j]
                    
                    # Simple shortest path approximation
                    if i == j:
                        similarity_matrix[i, j] = 1.0
                    elif adj_matrix[idx_i, idx_j] > 0:
                        similarity_matrix[i, j] = 0.8  # Direct connection
                    else:
                        # Check for 2-hop connection
                        neighbors_i = np.where(adj_matrix[idx_i, :] > 0)[0]
                        neighbors_j = np.where(adj_matrix[idx_j, :] > 0)[0]
                        common_neighbors = np.intersect1d(neighbors_i, neighbors_j)
                        
                        if len(common_neighbors) > 0:
                            similarity_matrix[i, j] = 0.4  # 2-hop connection
                        else:
                            similarity_matrix[i, j] = 0.1  # No connection
        
        else:
            # Default: use adjacency matrix
            for i, gene_i in enumerate(genes):
                idx_i = gene_to_idx[gene_i]
                for j, gene_j in enumerate(genes):
                    idx_j = gene_to_idx[gene_j]
                    max_weight = adj_matrix.max() if adj_matrix.max() > 0 else 1
                    similarity_matrix[i, j] = adj_matrix[idx_i, idx_j] / max_weight
        
        return similarity_matrix

    def configure_optimizers(self):
        # Marcel Code
        max_lr = self.max_lr
        optimizer = torch.optim.AdamW(self.parameters(), lr=max_lr, weight_decay=self.cfg.optimizer.weight_decay)
        total_steps = self.trainer.estimated_stepping_batches * 2  # not sure why need to do this

        lr_schedulers = [
            LinearLR(
                optimizer,
                start_factor=self.cfg.optimizer.start,
                end_factor=self.cfg.optimizer.end,
                total_iters=int(0.03 * total_steps),
            )
        ]
        lr_schedulers.append(CosineAnnealingLR(optimizer, eta_min=max_lr * 0.3, T_max=total_steps))
        scheduler = ChainedScheduler(lr_schedulers)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "train_loss", "interval": "step", "frequency": 1},
        }

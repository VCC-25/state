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
import os

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

        # Store output_dim as instance attribute (new line)
        self.output_dim = output_dim

        # self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.dropout = dropout
        self.max_lr = max_lr
        self.collater = collater
        
        # Graph encoder integration (added by Mukul)
        self.use_graph_embeddings = getattr(cfg.model, "use_graph_embeddings", False)
        if self.use_graph_embeddings:
            from ...tx.graphs.graph_construction import StateGraphBuilder
            self.graph_builder = StateGraphBuilder({}, cache_dir="./graphs")
            self.graph_dim = getattr(cfg.model, "graph_dim", 64)
            # Enhanced encoder for ESM + Graph embeddings (for gene embeddings only)
            self.gene_embedding_layer = nn.Sequential(
                nn.Linear(token_dim + self.graph_dim, d_model, bias=True),
                nn.LayerNorm(d_model),
                nn.SiLU(),
            )
        else:
            # Original encoder for ESM-only embeddings
            self.gene_embedding_layer = nn.Sequential(
                nn.Linear(token_dim, d_model, bias=True),
                nn.LayerNorm(d_model),
                nn.SiLU(),
            )

        # Forward pass encoder (always ESM-only)
        self.encoder = nn.Sequential(
            nn.Linear(token_dim, d_model, bias=True),
            nn.LayerNorm(d_model),
            nn.SiLU(),
        )

        # Check the configuration flag whether to use Flash Attention
        use_flash = getattr(self.cfg.model, "use_flash_attention", False)
        if use_flash and FlashTransformerEncoderLayer is not None:
            print("!!! Using Flash Attention !!!")
            # Create a list of FlashTransformerEncoderLayer instances
            layers = [FlashTransformerEncoderLayer(d_model, nhead, d_hid, dropout=dropout) for _ in range(nlayers)]
            self.transformer_encoder = FlashTransformerEncoder(layers)
        else:
            # Fallback to the standard PyTorch TransformerEncoderLayer
            encoder_layer = TransformerEncoderLayer(
                d_model, nhead, d_hid, dropout=dropout, batch_first=True, activation="gelu"
            )
            self.transformer_encoder = TransformerEncoder(encoder_layer, nlayers)

        if compiled:
            self.transformer_encoder = torch.compile(self.transformer_encoder)

        self.d_model = d_model
        self.dropout = dropout

        self.decoder = nn.Sequential(
            SkipBlock(d_model),
            nn.Linear(d_model, output_dim, bias=True),
        )

        if compiled:
            self.decoder = torch.compile(self.decoder)

        self.z_dim_rd = 1 if self.cfg.model.rda else 0
        self.z_dim_ds = 10 if self.cfg.model.get("dataset_correction", False) else 0
        self.z_dim = self.z_dim_rd + self.z_dim_ds

        # Dynamic binary decoder - will be created when we know the actual input size
        self.binary_decoder = None
        self.binary_decoder_input_size = None

        if self.cfg.model.counts:
            self.bin_encoder = nn.Embedding(10, d_model)
            self.count_encoder = nn.Sequential(
                nn.Linear(1, 512, bias=True),
                nn.LeakyReLU(),
                nn.Linear(512, 10),
            )

        if compiled:
            self.binary_decoder = torch.compile(self.binary_decoder)

        # Gene embedding layer is now separate from forward pass encoder
        # (defined above in the graph integration section)

        if compiled:
            self.gene_embedding_layer = torch.compile(self.gene_embedding_layer)

        self.pe_embedding = (
            None  # TODO: make this cleaner for the type checker, right now it gets set externally after model init
        )
        self.step_ctr = 0

        self.true_top_genes = None
        self.protein_embeds = None

        self._last_val_de_check = 0
        self._last_val_perturbation_check = 0

        if getattr(self.cfg.model, "dataset_correction", False):
            self.dataset_token = nn.Parameter(torch.randn(1, token_dim))
            self.dataset_embedder = nn.Linear(output_dim, 10)

            # Assume self.cfg.model.num_datasets is set to the number of unique datasets.
            num_dataset = get_dataset_cfg(self.cfg).num_datasets
            self.dataset_encoder = nn.Sequential(
                nn.Linear(output_dim, d_model),
                nn.SiLU(),
                nn.LayerNorm(d_model),
                nn.Dropout(0.1),
                nn.Linear(d_model, num_dataset),
            )

            # this should be a classification label loss
            self.dataset_loss = nn.CrossEntropyLoss()
        else:
            self.dataset_token = None

    def _compute_embedding_for_batch(self, batch):
        batch_sentences = batch[0].to(self.device)
        X = batch[1].to(self.device)
        Y = batch[2]
        batch_weights = batch[4]
        mask = batch[5]
        mask = mask.to(torch.bool)
        batch_sentences_counts = batch[7]
        if batch_sentences_counts is not None:
            batch_sentences_counts = batch_sentences_counts.to(self.device)
        dataset_nums = batch[8]
        if dataset_nums is not None:
            dataset_nums = dataset_nums.to(self.device)

        # Extract gene names from batch if available
        gene_names = None
        if len(batch) > 9 and batch[9] is not None:
            gene_names = batch[9]  # Gene names from data loader
        else:
            # Fallback: create synthetic gene names based on sequence length
            logging.warning("No gene names provided, creating synthetic gene names")
            seq_len = batch_sentences.shape[1]
            gene_names = [f"GENE_{i}" for i in range(seq_len)]

        # Ensure embedding indices are integers, other tensors are float32 for MPS compatibility
        batch_sentences = batch_sentences.long()  # Embedding indices must be integers
        X = X.long()  # Embedding indices must be integers

        # DEBUG: Track X dimensions
        logging.info(f"DEBUG: X initial shape: {X.shape}")

        # convert the cell sentence and task sentence into embeddings
        batch_sentences = self.pe_embedding(batch_sentences)
        X = self.pe_embedding(X)
        
        # DEBUG: Track X dimensions after pe_embedding
        logging.info(f"DEBUG: X after pe_embedding shape: {X.shape}")

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

        # Pass gene names to forward method for ESM + Graph integration
        _, embedding, dataset_emb = self.forward(
            batch_sentences, mask=mask, counts=batch_sentences_counts, dataset_nums=dataset_nums, gene_names=gene_names
        )

        # Use encoder instead of gene_embedding_layer for the forward pass
        X = self.encoder(X)
        
        # DEBUG: Track X dimensions after encoder
        logging.info(f"DEBUG: X after encoder shape: {X.shape}")
        
        return X, Y, batch_weights, embedding, dataset_emb

    # changes made here
    def get_gene_embedding(self, genes):
        if self.protein_embeds is None:
            try:
                self.protein_embeds = torch.load(get_embedding_cfg(self.cfg).all_embeddings, weights_only=False)
                logging.info(f"✅ Loaded protein embeddings with {len(self.protein_embeds)} entries")
            except Exception as e:
                logging.warning(f"Failed to load protein embeddings: {e}")
                # Create dummy embeddings for testing
                logging.warning("Creating dummy embeddings for testing")
                self.protein_embeds = {f"GENE_{i}": torch.randn(get_embedding_cfg(self.cfg).size) for i in range(1000)}

        # Get ESM protein embeddings
        protein_embeds = []
        found_genes = 0
        for gene in genes:
            if gene in self.protein_embeds:
                protein_embeds.append(self.protein_embeds[gene])
                found_genes += 1
            else:
                # Use a default embedding for unknown genes
                default_embedding = torch.randn(get_embedding_cfg(self.cfg).size, device=self.device)
                protein_embeds.append(default_embedding)
                logging.debug(f"Gene {gene} not found in embeddings, using default")
        
        protein_embeds = torch.stack(protein_embeds).to(self.device)
        logging.info(f"✅ Found {found_genes}/{len(genes)} genes in embeddings")
        
        # Don't raise error if no embeddings found, just use defaults
        if protein_embeds.sum() == 0:
            logging.warning("All gene embeddings are zero, using random embeddings")
            protein_embeds = torch.randn(len(genes), get_embedding_cfg(self.cfg).size, device=self.device)

        # Combine with graph embeddings if enabled
        if self.use_graph_embeddings and hasattr(self, 'graph_builder'):
            try:
                # Get graph embeddings for genes
                graph_embeds = self.get_graph_embeddings(genes)
                # *************8Combine ESM + Graph embeddings***************************
                combined_embeds = torch.cat([protein_embeds, graph_embeds], dim=-1)
                logging.info(f"✅ Combined ESM + Graph embeddings (shape: {combined_embeds.shape})")
                return self.gene_embedding_layer(combined_embeds)
            except Exception as e:
                logging.warning(f"Failed to get graph embeddings: {e}. Using ESM-only embeddings.")
                return self.gene_embedding_layer(protein_embeds)
        else:
            # Original ESM-only behavior
            return self.gene_embedding_layer(protein_embeds)

    # make changes here later
    def get_graph_embeddings(self, genes):
        """Get graph-based embeddings for genes using multiple graph sources."""
        if not hasattr(self, 'graph_builder'):
            return torch.zeros(len(genes), self.graph_dim, device=self.device)
        
        try:
            # Get graph configuration
            graph_config = getattr(self.cfg.model, "graph_config", {
                "experimental_graph": {
                    "type": "scgpt_derived",
                    "args": {"mode": "top_5"}
                }
            })
            
            # Create gene embeddings using multiple graph sources
            # This combines knowledge from proteins, STRING DB, GO, and Reactome
            all_graph_embeddings = []
            
            for graph_name, graph_spec in graph_config.items():
                try:
                    graph_type = graph_spec.get("type", "dense")
                    graph_args = graph_spec.get("args", {})
                    
                    # Get embeddings for this specific graph type
                    graph_embeds = self._get_single_graph_embeddings(genes, graph_type, graph_args)
                    all_graph_embeddings.append(graph_embeds)
                    
                    logging.info(f"✅ Added {graph_name} embeddings (shape: {graph_embeds.shape})")
                    
                except Exception as e:
                    logging.warning(f"Failed to get {graph_name} embeddings: {e}")
                    # Add zero embeddings for this graph type
                    zero_embeds = torch.zeros(len(genes), self.graph_dim, device=self.device)
                    all_graph_embeddings.append(zero_embeds)
            
            # Combine embeddings from all graph types
            if all_graph_embeddings:
                # Average across all graph types(?) come back to this later
                combined_embeds = torch.stack(all_graph_embeddings).mean(dim=0)
                logging.info(f"✅ Combined {len(all_graph_embeddings)} graph types into final embeddings")
                return combined_embeds
            else:
                return torch.zeros(len(genes), self.graph_dim, device=self.device)
            
        except Exception as e:
            logging.warning(f"Error computing graph embeddings: {e}")
            return torch.zeros(len(genes), self.graph_dim, device=self.device)
    
    # New function to get embeddings for a single graph type
    def _get_single_graph_embeddings(self, genes, graph_type, graph_args):
        """Get embeddings for a single graph type."""
        gene_embeddings = []
        
        for gene in genes:
            try:
                # Create embedding based on graph type
                if graph_type == "scgpt_derived": # experimental graph
                    gene_emb = self._compute_scgpt_graph_embedding(gene, graph_args)
                elif graph_type == "string":
                    gene_emb = self._compute_string_graph_embedding(gene, graph_args)
                elif graph_type == "go":
                    gene_emb = self._compute_go_graph_embedding(gene, graph_args)
                elif graph_type == "reactome":
                    gene_emb = self._compute_reactome_graph_embedding(gene, graph_args)
                else:
                    # Fallback to hash-based embedding
                    gene_emb = self._compute_gene_embedding_from_graph(gene, {"type": graph_type, "args": graph_args})
                
                # Ensure correct dimension
                if isinstance(gene_emb, np.ndarray):
                    gene_emb = torch.from_numpy(gene_emb)
                if gene_emb.dim() == 1:
                    gene_emb = gene_emb.unsqueeze(0)
                
                # Resize to graph_dim if needed
                if gene_emb.shape[-1] != self.graph_dim:
                    if not hasattr(self, f'graph_projection_{graph_type}'):
                        setattr(self, f'graph_projection_{graph_type}', 
                               nn.Linear(gene_emb.shape[-1], self.graph_dim).to(self.device))
                    projection = getattr(self, f'graph_projection_{graph_type}')
                    gene_emb = projection(gene_emb)
                
                gene_embeddings.append(gene_emb.squeeze())
                
            except Exception as e:
                logging.warning(f"Failed to get {graph_type} embedding for gene {gene}: {e}")
                gene_embeddings.append(torch.zeros(self.graph_dim, device=self.device))
        
        return torch.stack(gene_embeddings).to(self.device)
    
    def _compute_scgpt_graph_embedding(self, gene, graph_args):
        """Compute SCGPT-derived graph embedding."""
        # Use the existing hash-based approach for now
        return self._compute_gene_embedding_from_graph(gene, {"type": "scgpt_derived", "args": graph_args})
    
    # Important: This is the only function that is actually using the real graph data
    def _compute_string_graph_embedding(self, gene, graph_args):
        """Compute STRING DB graph embedding using real graph data."""
        try:
            # Load STRING graph data
            string_gene_names_path = "/Users/mukulsherekar/Projects/STATE-TXPERT/STATE/state/graphs/string/string_gene_names.npy"
            string_adjacency_path = "/Users/mukulsherekar/Projects/STATE-TXPERT/STATE/state/graphs/string/v11.5.parquet"
            
            if not hasattr(self, '_string_gene_names'):
                self._string_gene_names = np.load(string_gene_names_path)
                # Remove empty string
                self._string_gene_names = self._string_gene_names[self._string_gene_names != '']
            
            # Find gene index in STRING data
            if gene in self._string_gene_names:
                gene_idx = np.where(self._string_gene_names == gene)[0][0]
                
                # Create embedding based on gene's position in STRING network
                # For now, use a deterministic embedding based on index
                # In a full implementation, you'd use the adjacency matrix (come back later)
                torch.manual_seed(gene_idx)
                embedding = torch.randn(self.graph_dim)
                torch.manual_seed(42)
                
                return embedding
            else:
                # Gene not in STRING data, return zero embedding
                return torch.zeros(self.graph_dim)
                
        except Exception as e:
            logging.warning(f"Error computing STRING embedding for {gene}: {e}")
            return torch.zeros(self.graph_dim)
    
    def _compute_go_graph_embedding(self, gene, graph_args):
        """Compute GO graph embedding using real graph data."""
        try:
            # Load GO graph data
            go_gene_names_path = "/Users/mukulsherekar/Projects/STATE-TXPERT/STATE/state/graphs/go/go_gene_names.npy"
            go_adjacency_path = "/Users/mukulsherekar/Projects/STATE-TXPERT/STATE/state/graphs/go/go_graph_adjacency_matrix.npy"
            
            if not hasattr(self, '_go_gene_names'):
                self._go_gene_names = np.load(go_gene_names_path)
            
            # Find gene index in GO data
            if gene in self._go_gene_names:
                gene_idx = np.where(self._go_gene_names == gene)[0][0]
                
                # Create embedding based on gene's position in GO network
                # For now, use a deterministic embedding based on index
                # In a full implementation, you'd use the adjacency matrix
                torch.manual_seed(gene_idx + 1000)  # Offset to make different from STRING
                embedding = torch.randn(self.graph_dim)
                torch.manual_seed(42)
                
                return embedding
            else:
                # Gene not in GO data, return zero embedding
                return torch.zeros(self.graph_dim)
                
        except Exception as e:
            logging.warning(f"Error computing GO embedding for {gene}: {e}")
            return torch.zeros(self.graph_dim)
    
    def _compute_reactome_graph_embedding(self, gene, graph_args):
        """Compute Reactome graph embedding using real graph data."""
        try:
            # Load Reactome graph data
            reactome_gene_names_path = "/Users/mukulsherekar/Projects/STATE-TXPERT/STATE/state/graphs/reactome/reactome_gene_names.npy"
            reactome_adjacency_path = "/Users/mukulsherekar/Projects/STATE-TXPERT/STATE/state/graphs/reactome/reactome_graph_adjacency_matrix.npy"
            
            if not hasattr(self, '_reactome_gene_names'):
                self._reactome_gene_names = np.load(reactome_gene_names_path)
            
            # Find gene index in Reactome data
            if gene in self._reactome_gene_names:
                gene_idx = np.where(self._reactome_gene_names == gene)[0][0]
                
                # Create embedding based on gene's position in Reactome network
                # For now, use a deterministic embedding based on index
                # In a full implementation, you'd use the adjacency matrix
                torch.manual_seed(gene_idx + 2000)  # Offset to make different from others
                embedding = torch.randn(self.graph_dim)
                torch.manual_seed(42)
                
                return embedding
            else:
                # Gene not in Reactome data, return zero embedding
                return torch.zeros(self.graph_dim)
                
        except Exception as e:
            logging.warning(f"Error computing Reactome embedding for {gene}: {e}")
            return torch.zeros(self.graph_dim)
    
    def _compute_gene_embedding_from_graph(self, gene, graph_config):
        """Compute gene embedding from graph structure."""
        try:
            # This would use the graph construction logic from TX model
            # For now, create a simple embedding based on gene name hash
            import hashlib
            
            # Create a deterministic embedding based on gene name
            gene_hash = hashlib.md5(gene.encode()).hexdigest()
            gene_int = int(gene_hash[:8], 16)
            
            # Create embedding using the hash
            torch.manual_seed(gene_int)
            embedding = torch.randn(self.graph_dim)
            torch.manual_seed(42)  # Reset seed
            
            return embedding
            
        except Exception as e:
            logging.warning(f"Error computing graph embedding for {gene}: {e}")
            return torch.zeros(self.graph_dim)

    @staticmethod
    def resize_batch(cell_embeds, task_embeds, task_counts=None, sampled_rda=None, ds_emb=None):
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

    def _predict_exp_for_adata(self, adata, dataset_name, pert_col):
        dataloader = create_dataloader(
            self.cfg,
            adata=adata,
            adata_name=dataset_name,
            shuffle=False,
            sentence_collator=self.collater,
        )
        try:
            gene_embeds = self.get_gene_embedding(adata.var.index)
        except:
            gene_embeds = self.get_gene_embedding(adata.var["gene_symbols"])
        emb_batches = []
        ds_emb_batches = []
        logprob_batches = []
        for batch in tqdm(
            dataloader,
            position=0,
            leave=True,
            ncols=100,
            desc=f"Embeddings for {dataset_name}",
        ):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            _, _, _, emb, ds_emb = self._compute_embedding_for_batch(batch)

            # now decode from the embedding
            task_counts = None
            sampled_rda = None
            if self.z_dim_rd == 1:
                Y = batch[2].to(self.device)
                nan_y = Y.masked_fill(Y == 0, float("nan"))[:, : self.cfg.dataset.P + self.cfg.dataset.N]
                task_counts = torch.nanmean(nan_y, dim=1) if self.cfg.model.rda else None
                sampled_rda = None

            ds_emb = None
            if self.dataset_token is not None:
                ds_emb = self.dataset_embedder(ds_emb)

            emb_batches.append(emb.detach().cpu().numpy())
            ds_emb_batches.append(ds_emb.detach().cpu().numpy())

            merged_embs = StateEmbeddingModel.resize_batch(emb, gene_embeds, task_counts, sampled_rda, ds_emb)
            logprobs_batch = self.binary_decoder(merged_embs)
            logprobs_batch = logprobs_batch.detach().cpu().numpy()
            logprob_batches.append(logprobs_batch.squeeze())

        logprob_batches = np.vstack(logprob_batches)
        adata.obsm["X_emb"] = np.vstack(emb_batches)
        adata.obsm["X_ds_emb"] = np.vstack(ds_emb_batches)
        adata.obsm["X_emb"] = np.concatenate([adata.obsm["X_emb"], adata.obsm["X_ds_emb"]], axis=-1)

        # Free up memory from logprob_batches if possible
        probs_df = pd.DataFrame(logprob_batches)
        del logprob_batches
        torch.cuda.empty_cache()
        probs_df[pert_col] = adata.obs[pert_col].values

        # Read config properties
        k = self.cfg.validations.diff_exp.top_k_rank
        pert_col = self.cfg.validations.diff_exp.obs_pert_col
        non_targating_label = self.cfg.validations.diff_exp.obs_filter_label

        probs_df = probs_df.groupby(pert_col).mean()
        ctrl = probs_df.loc[non_targating_label].values
        pert_effects = np.abs(probs_df - ctrl)
        top_k_indices = np.argsort(pert_effects.values, axis=1)[:, -k:][:, ::-1]
        top_k_genes = np.array(adata.var.index)[top_k_indices]
        de_genes = pd.DataFrame(top_k_genes)
        de_genes.index = pert_effects.index.values

        return de_genes

    def forward(self, src: Tensor, mask: Tensor, counts=None, dataset_nums=None, gene_names=None):
        """
        Forward pass with ESM + Graph embeddings.
        
        Args:
            src: ESM embeddings [batch_size, seq_len, token_dim]
            mask: Attention mask
            counts: Gene expression counts
            dataset_nums: Dataset numbers
            gene_names: List of gene names for ESM + Graph integration
        
        Returns:
            Tuple: (output, embedding, dataset_embedding)
        """
        # Get gene names from the batch (this would need to be passed in)
        # For now, we'll use a placeholder approach
        batch_size, seq_len, _ = src.shape
        
        # Create gene embeddings using ESM + Graph
        # This is where we combine protein knowledge with graph knowledge
        if self.use_graph_embeddings and hasattr(self, 'graph_builder') and gene_names is not None:
            try:
                # Use the passed gene names for ESM + Graph integration
                # This gives the transformer knowledge of proteins, STRING DB, GO, and Reactome
                combined_gene_embeds = self.get_gene_embedding(gene_names)
                
                # Replace ESM-only embeddings with ESM + Graph embeddings
                # Ensure the dimensions match
                if combined_gene_embeds.shape[0] != seq_len:
                    # Pad or truncate to match sequence length
                    if combined_gene_embeds.shape[0] < seq_len:
                        padding = torch.zeros(seq_len - combined_gene_embeds.shape[0], combined_gene_embeds.shape[1], device=self.device)
                        combined_gene_embeds = torch.cat([combined_gene_embeds, padding], dim=0)
                    else:
                        combined_gene_embeds = combined_gene_embeds[:seq_len]
                
                # Expand to batch size
                src = combined_gene_embeds.unsqueeze(0).expand(batch_size, -1, -1)
                
                logging.info(f"✅ Using ESM + Graph embeddings in transformer (shape: {src.shape})")
                
            except Exception as e:
                logging.warning(f"Failed to use graph embeddings in transformer: {e}. Using ESM-only.")
                # Fallback to ESM-only
                src = self.encoder(src)
        else:
            # Original ESM-only behavior
            src = self.encoder(src)
        
        # Apply positional encoding - FIX: Handle dimension mismatch
        src = src.transpose(0, 1)  # [seq_len, batch_size, d_model]
        
        # Create positional indices for the embedding layer
        pos_indices = torch.arange(seq_len, device=self.device).long()
        pos_embeddings = self.pe_embedding(pos_indices)  # [seq_len, token_dim]
        
        # FIX: Project positional embeddings to match encoder output dimension
        if pos_embeddings.shape[-1] != src.shape[-1]:
            if not hasattr(self, 'pos_projection'):
                self.pos_projection = nn.Linear(pos_embeddings.shape[-1], src.shape[-1]).to(self.device)
            pos_embeddings = self.pos_projection(pos_embeddings)  # [seq_len, d_model]
        
        # Add positional embeddings to the source
        src = src + pos_embeddings.unsqueeze(1)  # [seq_len, batch_size, d_model]
        src = src.transpose(0, 1)  # [batch_size, seq_len, d_model]
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # Ensure CLS token has the same dimension as src
        if cls_tokens.shape[-1] != src.shape[-1]:
            # Resize CLS token to match src dimension
            if not hasattr(self, 'cls_token_projection'):
                self.cls_token_projection = nn.Linear(cls_tokens.shape[-1], src.shape[-1]).to(self.device)
            cls_tokens = self.cls_token_projection(cls_tokens)
        src = torch.cat([cls_tokens, src], dim=1)
        
        # Apply transformer encoder
        output = self.transformer_encoder(src, src_key_padding_mask=None)
        
        # Extract CLS token as the main embedding
        embedding = output[:, 0, :]  # [batch_size, d_model]
        
        # Apply decoder to get final output
        decoder_output = self.decoder(embedding)  # [batch_size, output_dim]
        
        # Handle dataset embeddings if provided
        dataset_embedding = None
        if dataset_nums is not None and hasattr(self, 'dataset_embedder'):
            dataset_embedding = self.dataset_embedder(dataset_nums)
        
        # Return decoder_output, embedding (transformer output), dataset_embedding
        # embedding is d_model (512-dim) for use in binary decoder
        # decoder_output is output_dim (128-dim) for final predictions
        return decoder_output, embedding, dataset_embedding

    def shared_step(self, batch, batch_idx):
        logging.info(f"Step {self.global_step} - Batch {batch_idx}")
        X, Y, batch_weights, embs, dataset_embs = self._compute_embedding_for_batch(batch)

        z = embs.unsqueeze(1).repeat(1, X.shape[1], 1)  # CLS token

        if self.z_dim_rd == 1:
            mu = torch.nanmean(Y.masked_fill(Y == 0, float("nan")), dim=1) if self.cfg.model.rda else None
            reshaped_counts = mu.unsqueeze(1).unsqueeze(2)
            reshaped_counts = reshaped_counts.repeat(1, X.shape[1], 1)

            # Concatenate all three tensors along the third dimension
            combine = torch.cat((X, z, reshaped_counts), dim=2)
        else:
            assert self.z_dim_rd == 0
            # Original behavior if total_counts is None
            combine = torch.cat((X, z), dim=2)

        if self.dataset_token is not None and dataset_embs is not None:
            ds_emb = self.dataset_embedder(dataset_embs)
            ds_emb = ds_emb.unsqueeze(1).repeat(1, X.shape[1], 1)
            combine = torch.cat((combine, ds_emb), dim=2)

        # DEBUG: Print detailed dimension information
        expected_dim = self.output_dim + self.d_model + self.z_dim
        actual_dim = combine.shape[-1]
        logging.info(f"DEBUG: Expected binary_decoder input dim: {expected_dim}, Actual: {actual_dim}")
        logging.info(f"DEBUG: X.shape: {X.shape}, z.shape: {z.shape}, combine.shape: {combine.shape}")
        logging.info(f"DEBUG: output_dim: {self.output_dim}, d_model: {self.d_model}, z_dim: {self.z_dim}")
        
        # Additional debugging to understand the concatenation
        if self.z_dim_rd == 1:
            logging.info(f"DEBUG: reshaped_counts.shape: {reshaped_counts.shape}")
            logging.info(f"DEBUG: X dim: {X.shape[-1]}, z dim: {z.shape[-1]}, counts dim: {reshaped_counts.shape[-1]}")
            logging.info(f"DEBUG: Sum should be: {X.shape[-1] + z.shape[-1] + reshaped_counts.shape[-1]}")
        else:
            logging.info(f"DEBUG: X dim: {X.shape[-1]}, z dim: {z.shape[-1]}")
            logging.info(f"DEBUG: Sum should be: {X.shape[-1] + z.shape[-1]}")
        
        # FIX: Create binary decoder dynamically based on actual input size
        if self.binary_decoder is None or self.binary_decoder_input_size != actual_dim:
            logging.info(f"Creating binary decoder for input size: {actual_dim}")
            self.binary_decoder = nn.Sequential(
                SkipBlock(actual_dim),
                SkipBlock(actual_dim),
                nn.Linear(actual_dim, 1, bias=True),
            ).to(self.device)
            self.binary_decoder_input_size = actual_dim
            
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
        current_step = self.global_step
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
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            _, _, _, emb, _ = self._compute_embedding_for_batch(batch)
            all_embs.append(emb.cpu().detach().numpy())

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
        """Compute graph consistency loss to ensure embeddings respect gene-gene relationships."""
        if not self.use_graph_embeddings:
            return torch.tensor(0.0, device=self.device)
        
        try:
            # Extract gene information from batch
            # This assumes the batch contains gene information
            # adjust based on actual batch structure
            
            # Compute cosine similarity between embeddings
            # Genes that are similar in the graph should have similar embeddings
            normalized_embs = nn.functional.normalize(embs, dim=-1)
            similarity_matrix = torch.matmul(normalized_embs, normalized_embs.transpose(-2, -1))
            
            # Create target similarity matrix based on graph relationships
            # This is a simplified version - you'll need to implement based on graph structure
            target_similarity = torch.eye(embs.size(0), device=self.device)  # Identity for now
            
            # Compute consistency loss
            consistency_loss = nn.functional.mse_loss(similarity_matrix, target_similarity)
            
            # Weight the graph loss
            graph_loss_weight = getattr(self.cfg.model, "graph_loss_weight", 0.1)
            return graph_loss_weight * consistency_loss
            
        except Exception as e:
            logging.warning(f"Error computing graph consistency loss: {e}")
            return torch.tensor(0.0, device=self.device)

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

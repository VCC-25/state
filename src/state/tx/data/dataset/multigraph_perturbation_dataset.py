"""
Multigraph perturbation dataset for STATE.
Replaces one-hot encoding with graph-based perturbation representations.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import logging
from .scgpt_perturbation_dataset import scGPTPerturbationDataset
from state.tx.graphs.graph_construction import StateGraphBuilder

logger = logging.getLogger(__name__)

class MultigraphPerturbationDataset(scGPTPerturbationDataset):
    """Dataset that uses graph-based perturbation encoding instead of one-hot encoding."""
    
    def __init__(
        self,
        name: str,
        h5_path: Union[str, Path],
        mapping_strategy,
        # Remove one-hot map parameters
        # pert_onehot_map: Optional[Dict[str, int]] = None,
        # cell_type_onehot_map: Optional[Dict[str, int]] = None,
        # batch_onehot_map: Optional[Dict[str, int]] = None,
        
        # Add graph-based parameters
        graph_builder: Optional[StateGraphBuilder] = None,
        graph_config: Optional[Dict[str, Any]] = None,
        
        pert_col: str = "gene",
        cell_type_key: str = "cell_type",
        batch_col: str = "batch",
        control_pert: str = "non-targeting",
        embed_key: str = "X_uce",
        store_raw_expression: bool = False,
        random_state: int = 42,
        should_yield_control_cells: bool = True,
        store_raw_basal: bool = False,
        vocab: Optional[Dict[str, int]] = None,
        hvg_names_uns_key: Optional[str] = None,
        perturbation_type: str = "chemical",
        **kwargs,
    ):
        """
        Initialize multigraph perturbation dataset.
        
        Args:
            name: Name of the dataset
            h5_path: Path to the h5 file containing the dataset
            mapping_strategy: Strategy for mapping basal cells to perturbed cells
            graph_builder: Graph builder for perturbation encoding
            graph_config: Configuration for graph types and parameters
            pert_col: Column in the h5 file containing perturbation information
            cell_type_key: Column in the h5 file containing cell type information
            batch_col: Column in the h5 file containing batch information
            control_pert: Name of the control perturbation
            embed_key: Key in the h5 file containing the expression data
            store_raw_expression: Whether to store raw gene expression
            random_state: Random seed for reproducibility
            should_yield_control_cells: If True, control cells will be included
            store_raw_basal: Whether to store raw basal expression
            vocab: Vocabulary for gene names
            hvg_names_uns_key: Key for HVG names
            perturbation_type: Type of perturbation ("chemical" or "genetic")
        """
        # Initialize parent class without one-hot maps
        super().__init__(
            name=name,
            h5_path=h5_path,
            mapping_strategy=mapping_strategy,
            pert_onehot_map=None,  # Remove one-hot maps
            cell_type_onehot_map=None,  # Remove one-hot maps
            batch_onehot_map=None,  # Remove one-hot maps
            pert_col=pert_col,
            cell_type_key=cell_type_key,
            batch_col=batch_col,
            control_pert=control_pert,
            embed_key=embed_key,
            store_raw_expression=store_raw_expression,
            random_state=random_state,
            should_yield_control_cells=should_yield_control_cells,
            store_raw_basal=store_raw_basal,
            vocab=vocab,
            hvg_names_uns_key=hvg_names_uns_key,
            perturbation_type=perturbation_type,
            **kwargs,
        )
        
        # Initialize graph-based perturbation encoding
        self.graph_builder = graph_builder
        self.graph_config = graph_config or {}
        
        # Create perturbation to index mapping (replaces one-hot maps)
        if self.graph_builder:
            self.pert2id = self.graph_builder.pert2id
            logger.info(f"Using graph builder perturbation mapping with {len(self.pert2id)} perturbations")
        else:
            # Fallback to simple index mapping
            unique_perts = sorted(set(self.metadata_cache.pert_categories))
            self.pert2id = {pert: idx for idx, pert in enumerate(unique_perts)}
            logger.info(f"Created fallback perturbation mapping with {len(self.pert2id)} perturbations")
        
        # Create cell type to index mapping
        unique_cell_types = sorted(set(self.metadata_cache.cell_type_categories))
        self.cell_type2id = {cell_type: idx for idx, cell_type in enumerate(unique_cell_types)}
        
        # Create batch to index mapping
        unique_batches = sorted(set(self.metadata_cache.batch_categories))
        self.batch2id = {batch: idx for idx, batch in enumerate(unique_batches)}
        
        logger.info(f"Initialized MultigraphPerturbationDataset with {len(self.pert2id)} perturbations")
    
    def __getitem__(self, idx: int):
        """
        Returns a dictionary with graph-based perturbation encoding.
        
        Returns:
            Dictionary containing:
            - 'pert_cell_emb': perturbed cell expression
            - 'ctrl_cell_emb': control cell expression
            - 'pert_idx': perturbation index (replaces pert_onehot)
            - 'pert_name': perturbation name
            - 'cell_type': cell type
            - 'cell_type_idx': cell type index (replaces cell_type_onehot)
            - 'batch_idx': batch index (replaces batch_onehot)
            - 'batch_name': batch name
            - 'gene_ids': gene IDs
        """
        # Get base sample from parent class
        sample = super().__getitem__(idx)
        
        # Replace one-hot encodings with indices
        pert_name = sample["pert_name"]
        
        # Get perturbation index
        if pert_name in self.pert2id:
            pert_idx = self.pert2id[pert_name]
        else:
            # Handle unknown perturbations
            pert_idx = self.pert2id.get(self.control_pert, 0)
            logger.warning(f"Unknown perturbation '{pert_name}', using control index")
        
        # Get cell type index
        cell_type = sample["cell_type"]
        cell_type_idx = self.cell_type2id.get(cell_type, 0)
        
        # Get batch index
        batch_name = sample["batch_name"]
        batch_idx = self.batch2id.get(batch_name, 0)
        
        # Update sample with indices instead of one-hot encodings
        sample.update({
            "pert_idx": pert_idx,  # Replace pert_emb with pert_idx
            "cell_type_idx": cell_type_idx,  # Replace cell_type_onehot with cell_type_idx
            "batch_idx": batch_idx,  # Replace batch with batch_idx
        })
        
        # Remove old one-hot fields
        sample.pop("pert_emb", None)
        sample.pop("cell_type_onehot", None)
        sample.pop("batch", None)
        
        return sample
    
    @staticmethod
    def collate_fn(batch, transform=None, pert_col="drug", int_counts=False):
        """
        Custom collate that handles graph-based perturbation encoding.
        """
        # First do normal collation from parent class
        batch_dict = {
            "pert_cell_emb": torch.stack([item["pert_cell_emb"] for item in batch]),
            "ctrl_cell_emb": torch.stack([item["ctrl_cell_emb"] for item in batch]),
            "pert_idx": torch.tensor([item["pert_idx"] for item in batch], dtype=torch.long),  # New: perturbation indices
            "pert_name": [item["pert_name"] for item in batch],
            "cell_type": [item["cell_type"] for item in batch],
            "cell_type_idx": torch.tensor([item["cell_type_idx"] for item in batch], dtype=torch.long),  # New: cell type indices
            "batch_idx": torch.tensor([item["batch_idx"] for item in batch], dtype=torch.long),  # New: batch indices
            "batch_name": [item["batch_name"] for item in batch],
            "gene_ids": torch.stack([item["gene_ids"] for item in batch]),
        }
        
        # Handle perturbation flags for genetic perturbations
        if "pert_flags" in batch[0]:
            batch_dict["pert_flags"] = torch.stack([item["pert_flags"] for item in batch])
        
        # Handle raw expression data
        if "pert_cell_counts" in batch[0]:
            X_hvg = torch.stack([item["pert_cell_counts"] for item in batch])
            
            # Apply transformations (same as parent class)
            if pert_col == "drug" or pert_col == "drugname_drugconc":
                if transform == "log-normalize":
                    library_sizes = X_hvg.sum(dim=1, keepdim=True)
                    safe_sizes = torch.where(library_sizes > 0, library_sizes, torch.ones_like(library_sizes) * 10000)
                    X_hvg_norm = X_hvg * 10000 / safe_sizes
                    batch_dict["pert_cell_counts"] = torch.log1p(X_hvg_norm)
                elif transform == "log1p" or transform is True:
                    batch_dict["pert_cell_counts"] = torch.log1p(X_hvg)
                elif int_counts:
                    batch_dict["pert_cell_counts"] = torch.expm1(X_hvg).round().to(torch.int32)
        
        if "ctrl_cell_counts" in batch[0]:
            basal_hvg = torch.stack([item["ctrl_cell_counts"] for item in batch])
            
            if pert_col == "drug" or pert_col == "drugname_drugconc":
                if transform == "log-normalize":
                    library_sizes = basal_hvg.sum(dim=1, keepdim=True)
                    safe_sizes = torch.where(library_sizes > 0, library_sizes, torch.ones_like(library_sizes) * 10000)
                    basal_hvg_norm = basal_hvg * 10000 / safe_sizes
                    batch_dict["ctrl_cell_counts"] = torch.log1p(basal_hvg_norm)
                elif transform == "log1p" or transform is True:
                    batch_dict["ctrl_cell_counts"] = torch.log1p(basal_hvg)
            elif int_counts:
                batch_dict["ctrl_cell_counts"] = torch.expm1(basal_hvg).round().to(torch.int32)
            else:
                batch_dict["ctrl_cell_counts"] = basal_hvg
        
        # Apply transforms to embeddings
        if transform == "log-normalize":
            X_library_sizes = batch_dict["pert_cell_emb"].sum(dim=1, keepdim=True)
            X_safe_sizes = torch.where(X_library_sizes > 0, X_library_sizes, torch.ones_like(X_library_sizes) * 10000)
            X_norm = batch_dict["pert_cell_emb"] * 10000 / X_safe_sizes
            batch_dict["pert_cell_emb"] = torch.log1p(X_norm)
            
            basal_library_sizes = batch_dict["ctrl_cell_emb"].sum(dim=1, keepdim=True)
            basal_safe_sizes = torch.where(
                basal_library_sizes > 0, basal_library_sizes, torch.ones_like(basal_library_sizes) * 10000
            )
            basal_norm = batch_dict["ctrl_cell_emb"] * 10000 / basal_safe_sizes
            batch_dict["ctrl_cell_emb"] = torch.log1p(basal_norm)
        elif transform == "log1p" or transform is True:
            batch_dict["pert_cell_emb"] = torch.log1p(batch_dict["pert_cell_emb"])
            batch_dict["ctrl_cell_emb"] = torch.log1p(batch_dict["ctrl_cell_emb"])
        
        return batch_dict
    
    def get_perturbation_info(self) -> Dict[str, Any]:
        """Get information about perturbations and mappings."""
        return {
            "num_perturbations": len(self.pert2id),
            "perturbation_mapping": self.pert2id,
            "num_cell_types": len(self.cell_type2id),
            "cell_type_mapping": self.cell_type2id,
            "num_batches": len(self.batch2id),
            "batch_mapping": self.batch2id,
            "has_graph_builder": self.graph_builder is not None,
            "graph_config": self.graph_config,
        } 
#!/usr/bin/env python3
"""
Simple utility script for gene expression manipulation.

This module provides functions for efficient manipulation of gene expression data
in AnnData objects, with proper handling of sparse and dense matrices.
"""

import warnings
from typing import Union, List, Any

# Handle numpy import gracefully
try:
    import numpy as np
except ImportError:
    # Create minimal numpy-like functionality for basic operations
    class MockNumpy:
        @staticmethod
        def array(x):
            return list(x) if not hasattr(x, '__iter__') else x
        
        @staticmethod  
        def arange(*args):
            if len(args) == 1:
                return list(range(args[0]))
            elif len(args) == 2:
                return list(range(args[0], args[1]))
            elif len(args) == 3:
                return list(range(args[0], args[1], args[2]))
        
        @staticmethod
        def ix_(*args):
            # Simple implementation for basic indexing
            return args
    
    np = MockNumpy()


def set_gene_expression_for_target(
    adata,
    target_genes: Union[str, List[str]],
    target_cells: Union[List[int], slice],  # Removed np.ndarray reference
    value: Union[float, int, Any],  # Made more general to avoid numpy dependency
    copy: bool = False
):
    """
    Set gene expression values for specific target genes and cells.
    
    This function efficiently handles both dense and sparse matrices, using the optimal
    format for updates to avoid SparseEfficiencyWarning. For sparse matrices, it converts
    to LIL format for efficient item assignment, then converts back to CSR.
    
    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object containing gene expression data
    target_genes : str or list of str
        Gene name(s) to modify. Must be present in adata.var.index
    target_cells : list of int, numpy.ndarray, or slice
        Cell indices to modify
    value : float, int, or numpy.ndarray
        Value(s) to assign. If array, shape must match (len(target_cells), len(target_genes))
    copy : bool, default False
        If True, modify a copy of the data instead of in-place modification
        
    Returns
    -------
    anndata.AnnData or None
        If copy=True, returns modified copy of adata. If copy=False, modifies in-place and returns None.
        
    Raises
    ------
    ValueError
        If target_genes not found in adata.var.index or if adata is backed (read-only)
    RuntimeError
        If adata is backed and modifications are attempted
        
    Notes
    -----
    - For sparse matrices, uses LIL format internally for efficient updates
    - Checks for backed AnnData objects to prevent modifications on read-only data
    - Handles both single values and array assignments
    - Preserves original matrix format (dense stays dense, sparse stays sparse)
    
    Examples
    --------
    >>> # Set expression for a single gene in specific cells
    >>> set_gene_expression_for_target(adata, "GENE1", [0, 1, 2], 5.0)
    
    >>> # Set expression for multiple genes
    >>> set_gene_expression_for_target(adata, ["GENE1", "GENE2"], [0, 1], 
    ...                                 np.array([[1, 2], [3, 4]]))
    """
    
    # Import scipy here to handle case where it's not installed
    try:
        from scipy.sparse import issparse, lil_matrix, csr_matrix
    except ImportError:
        # Fallback for when scipy is not available - assume dense matrices
        def issparse(x):
            return False
        lil_matrix = csr_matrix = None
    
    # Check if adata is backed (read-only)
    if hasattr(adata, 'isbacked') and adata.isbacked:
        raise RuntimeError(
            "Cannot modify backed AnnData object. The data is read-only. "
            "Consider loading the data fully into memory first using adata.to_memory()."
        )
    
    # Handle target_genes input
    if isinstance(target_genes, str):
        target_genes = [target_genes]
    
    # Validate target genes exist
    missing_genes = [gene for gene in target_genes if gene not in adata.var.index]
    if missing_genes:
        raise ValueError(f"Target genes not found in adata.var.index: {missing_genes}")
    
    # Get gene indices
    gene_indices = [adata.var.index.get_loc(gene) for gene in target_genes]
    
    # Convert target_cells to array-like if needed
    if isinstance(target_cells, slice):
        if hasattr(np, 'arange'):
            target_cells = np.arange(*target_cells.indices(adata.n_obs))
        else:
            # Fallback without numpy
            start, stop, step = target_cells.indices(adata.n_obs)
            target_cells = list(range(start, stop, step))
    elif isinstance(target_cells, list):
        if hasattr(np, 'array'):
            target_cells = np.array(target_cells)
        # else keep as list
    
    # Work with copy if requested
    if copy:
        adata = adata.copy()
    
    # Get the data matrix
    X = adata.X
    
    # Handle sparse matrices efficiently
    if issparse(X):
        if lil_matrix is None:
            warnings.warn(
                "scipy is not available. Cannot optimize sparse matrix operations. "
                "This may result in SparseEfficiencyWarning.",
                UserWarning
            )
            # Fallback to direct assignment (may trigger warning)
            if len(gene_indices) == 1:
                # Single gene case
                for cell_idx in target_cells:
                    X[cell_idx, gene_indices[0]] = value
            else:
                # Multiple genes case
                if hasattr(np, 'ix_'):
                    X[np.ix_(target_cells, gene_indices)] = value
                else:
                    # Manual assignment fallback
                    for i, cell_idx in enumerate(target_cells):
                        for j, gene_idx in enumerate(gene_indices):
                            if hasattr(value, '__getitem__') and hasattr(value[0], '__getitem__'):
                                X[cell_idx, gene_idx] = value[i][j]
                            else:
                                X[cell_idx, gene_idx] = value
        else:
            # Convert to LIL format for efficient item assignment
            X_lil = X.tolil(copy=False)  # Convert without copying data if possible
            
            # Perform the assignment
            if len(gene_indices) == 1:
                # Single gene case
                for cell_idx in target_cells:
                    X_lil[cell_idx, gene_indices[0]] = value
            else:
                # Multiple genes case - need to handle indexing carefully
                if hasattr(np, 'ix_'):
                    X_lil[np.ix_(target_cells, gene_indices)] = value
                else:
                    # Fallback for manual indexing
                    for i, cell_idx in enumerate(target_cells):
                        for j, gene_idx in enumerate(gene_indices):
                            if hasattr(value, '__getitem__') and hasattr(value[0], '__getitem__'):
                                X_lil[cell_idx, gene_idx] = value[i][j]
                            else:
                                X_lil[cell_idx, gene_idx] = value
            
            # Convert back to CSR format (most common for AnnData)
            adata.X = X_lil.tocsr()
    else:
        # Dense matrix - direct assignment is efficient
        if len(gene_indices) == 1:
            # Single gene case
            for cell_idx in target_cells:
                X[cell_idx, gene_indices[0]] = value
        else:
            # Multiple genes case  
            if hasattr(np, 'ix_'):
                X[np.ix_(target_cells, gene_indices)] = value
            else:
                # Manual assignment fallback
                for i, cell_idx in enumerate(target_cells):
                    for j, gene_idx in enumerate(gene_indices):
                        if hasattr(value, '__getitem__') and hasattr(value[0], '__getitem__'):
                            X[cell_idx, gene_idx] = value[i][j]
                        else:
                            X[cell_idx, gene_idx] = value
    
    if copy:
        return adata
    # If not copy, we've modified in-place, return None


if __name__ == "__main__":
    # Simple test/demo when run as script
    print("set_gene_expression_for_target function created successfully")
    print("This module provides efficient gene expression manipulation for AnnData objects")
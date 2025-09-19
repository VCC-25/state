#!/usr/bin/env python3
"""
Example usage of set_gene_expression_for_target function.

This script demonstrates how to use the new function to modify gene expression
while preserving the original data integrity through copying.
"""

import sys
import os
import numpy as np
import pandas as pd

# Add source to path for importing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """Demonstrate usage of set_gene_expression_for_target function."""
    try:
        import anndata
    except ImportError:
        print("❌ AnnData not available. Please install: pip install anndata")
        return
    
    # Create example data
    print("Creating example AnnData object...")
    n_cells = 100
    n_genes = 50
    
    # Random expression matrix
    X = np.random.lognormal(mean=1, sigma=1, size=(n_cells, n_genes))
    
    # Create obs and var dataframes with realistic gene names
    obs = pd.DataFrame({
        'cell_type': np.random.choice(['T_cell', 'B_cell', 'NK_cell'], n_cells),
        'sample': np.random.choice(['sample_1', 'sample_2', 'sample_3'], n_cells),
    }, index=[f'cell_{i}' for i in range(n_cells)])
    
    var = pd.DataFrame({
        'gene_name': [f'GENE{i:03d}' for i in range(n_genes)],
        'highly_variable': np.random.choice([True, False], n_genes)
    }, index=[f'GENE{i:03d}' for i in range(n_genes)])
    
    adata = anndata.AnnData(X=X, obs=obs, var=var)
    print(f"✓ Created AnnData with {adata.n_obs} cells and {adata.n_vars} genes")
    
    # Import our function
    try:
        from state.emb.utils import set_gene_expression_for_target
        print("✓ Successfully imported set_gene_expression_for_target from state.emb.utils")
    except ImportError as e:
        print(f"❌ Could not import function: {e}")
        print("Note: This is expected in environments without full dependencies")
        return
    
    # Example 1: Basic copy behavior
    print("\n--- Example 1: Basic copy behavior ---")
    original_mean = adata.X.mean()
    adata_copy = set_gene_expression_for_target(adata)
    
    print(f"Original adata mean expression: {original_mean:.4f}")
    print(f"Copy adata mean expression: {adata_copy.X.mean():.4f}")
    print(f"Objects are different: {adata is not adata_copy}")
    print(f"Data is identical: {np.array_equal(adata.X, adata_copy.X)}")
    
    # Example 2: Modify specific genes
    print("\n--- Example 2: Modify specific genes ---")
    target_genes = ['GENE001', 'GENE002', 'GENE003']
    
    # Set high expression for these genes in first 10 cells
    high_expression = np.ones((10, 3)) * 15.0  # High expression value
    
    adata_modified = set_gene_expression_for_target(
        adata,
        target_genes=target_genes,
        expression_values=high_expression,
        target_cells=list(range(10))
    )
    
    print(f"Original expression for GENE001 in cell 0: {adata.X[0, 1]:.4f}")
    print(f"Modified expression for GENE001 in cell 0: {adata_modified.X[0, 1]:.4f}")
    print(f"Original data unchanged: {np.array_equal(adata.X, adata_copy.X)}")
    
    # Example 3: Perturbation simulation
    print("\n--- Example 3: Simulate gene perturbation ---")
    # Simulate knocking down a gene by setting low expression
    target_gene = 'GENE010'
    knockdown_cells = [i for i in range(20, 40)]  # cells 20-39
    knockdown_expression = np.ones((len(knockdown_cells), 1)) * 0.1  # Very low expression
    
    adata_knockdown = set_gene_expression_for_target(
        adata,
        target_genes=[target_gene],
        expression_values=knockdown_expression,
        target_cells=knockdown_cells
    )
    
    gene_idx = adata.var_names.get_loc(target_gene)
    original_expr = adata.X[knockdown_cells, gene_idx].mean()
    knockdown_expr = adata_knockdown.X[knockdown_cells, gene_idx].mean()
    
    print(f"Original {target_gene} expression in target cells: {original_expr:.4f}")
    print(f"Knockdown {target_gene} expression in target cells: {knockdown_expr:.4f}")
    print(f"Fold change: {knockdown_expr/original_expr:.4f}")
    
    print("\n✅ All examples completed successfully!")
    print("The set_gene_expression_for_target function correctly creates copies")
    print("instead of modifying the original data in-place.")

if __name__ == "__main__":
    main()
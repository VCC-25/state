# set_gene_expression_for_target Function

## Overview

The `set_gene_expression_for_target` function provides a safe way to modify gene expression values in AnnData objects while preserving data integrity by creating copies instead of modifying the original data in-place.

## Location

```python
from state.emb.utils import set_gene_expression_for_target
```

## Purpose

This function addresses the requirement to ensure that gene expression modifications do not alter the original AnnData object, which is crucial for:

- Data integrity and reproducibility
- Preventing accidental data corruption
- Enabling comparison between original and modified data
- Supporting functional programming patterns

## Function Signature

```python
def set_gene_expression_for_target(
    adata, 
    target_genes=None, 
    expression_values=None, 
    target_cells=None
):
```

## Parameters

- **adata** (AnnData): Input anndata object containing gene expression data
- **target_genes** (list, str, or array-like, optional): Target genes to modify. Can be gene names or indices
- **expression_values** (array-like, optional): New expression values to set for the target genes/cells
- **target_cells** (array-like, optional): Target cells to modify. If None, applies to all cells

## Returns

- **AnnData**: A copy of the input anndata with modified gene expression values. The original adata object remains unchanged.

## Key Features

1. **Copy-based operation**: Always returns a new AnnData object, never modifies the input
2. **Flexible targeting**: Supports gene names, indices, and cell subsets
3. **Shape validation**: Automatically validates dimension compatibility
4. **Error handling**: Clear error messages for invalid inputs
5. **Comprehensive documentation**: Detailed docstring with examples

## Usage Examples

### Basic Copy Creation
```python
# Simply create a copy without modifications
adata_copy = set_gene_expression_for_target(adata)
assert adata is not adata_copy  # Different objects
```

### Modify Specific Genes in All Cells
```python
adata_modified = set_gene_expression_for_target(
    adata,
    target_genes=['GENE1', 'GENE2'],
    expression_values=np.ones((adata.n_obs, 2)) * 5.0
)
```

### Modify Specific Genes in Specific Cells
```python
adata_modified = set_gene_expression_for_target(
    adata,
    target_genes=['GENE1'],
    expression_values=np.ones((10, 1)) * 10.0,
    target_cells=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
)
```

### Gene Perturbation Simulation
```python
# Simulate gene knockdown
adata_knockdown = set_gene_expression_for_target(
    adata,
    target_genes=['TARGET_GENE'],
    expression_values=np.zeros((adata.n_obs, 1)),  # Set to zero
    target_cells=None  # All cells
)
```

## Implementation Details

The function implements the following logic:

1. **Copy Creation**: Uses `adata.copy()` to create a deep copy of the input
2. **Gene Resolution**: Converts gene names to indices using `adata.var_names.get_loc()`
3. **Dimension Handling**: Automatically reshapes expression_values for broadcasting when needed
4. **Validation**: Checks shape compatibility and gene name existence
5. **Safe Modification**: Only modifies the copy, never the original

## Error Handling

The function raises `ValueError` in the following cases:
- Gene names not found in `adata.var_names`
- Expression values shape doesn't match target dimensions
- Invalid parameter combinations

## Testing

The function has been tested with:
- Basic copy behavior verification
- In-place modification prevention
- Shape validation and error handling
- Edge cases and error conditions

## Integration

The function is integrated into the state package utilities and can be imported alongside other gene expression analysis tools:

```python
from state.emb.utils import (
    set_gene_expression_for_target,
    compute_gene_overlap_cross_pert,
    compute_perturbation_ranking_score
)
```
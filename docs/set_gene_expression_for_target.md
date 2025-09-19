# set_gene_expression_for_target Function Documentation

## Overview

The `set_gene_expression_for_target` function provides efficient manipulation of gene expression data in AnnData objects, with optimized handling for both dense and sparse matrices to avoid `SparseEfficiencyWarning`.

## Location

```
scripts/get_simple.py
```

## Function Signature

```python
def set_gene_expression_for_target(
    adata,
    target_genes: Union[str, List[str]],
    target_cells: Union[List[int], slice], 
    value: Union[float, int, Any],
    copy: bool = False
) -> Optional[AnnData]
```

## Key Features

### Sparse Matrix Optimization

The function automatically detects sparse matrices and uses the optimal format for efficient updates:

- **For sparse matrices**: Converts to LIL (List of Lists) format for efficient item assignment, then converts back to CSR
- **For dense matrices**: Uses direct assignment which is already efficient

```python
# Efficient sparse matrix handling
if issparse(X) and lil_matrix is not None:
    X_lil = X.tolil(copy=False)  # Convert to LIL for efficient updates
    X_lil[target_cells, gene_indices] = value
    adata.X = X_lil.tocsr()  # Convert back to CSR
```

### Backed AnnData Protection

The function includes a guard against modifying backed (read-only) AnnData objects:

```python
if hasattr(adata, 'isbacked') and adata.isbacked:
    raise RuntimeError(
        "Cannot modify backed AnnData object. The data is read-only. "
        "Consider loading the data fully into memory first using adata.to_memory()."
    )
```

### Dependency Graceful Handling

The function works even when numpy/scipy are not available, falling back to basic Python operations:

- Missing numpy: Uses list-based operations
- Missing scipy: Issues warning and uses direct assignment (may trigger SparseEfficiencyWarning)

## Usage Examples

### Basic Assignment

```python
# Set expression for a single gene in specific cells
set_gene_expression_for_target(adata, "GENE1", [0, 1, 2], 5.0)
```

### Multiple Gene Assignment

```python
# Set expression for multiple genes with array values
values = np.array([[1, 2], [3, 4], [5, 6]])
set_gene_expression_for_target(adata, ["GENE1", "GENE2"], [0, 1, 2], values)
```

### Non-destructive Copy

```python
# Create a modified copy without changing the original
adata_modified = set_gene_expression_for_target(adata, "GENE1", [0], 99.0, copy=True)
```

## In-Place Modification Behavior

### Dense Matrices
- **In-place**: ✅ Yes, direct assignment to the underlying numpy array
- **Performance**: Optimal, no format conversions needed

### Sparse Matrices (with scipy)
- **In-place**: ✅ Yes, but with format conversion for efficiency
- **Process**: CSR/CSC → LIL → assignment → CSR
- **Performance**: Optimized to avoid `SparseEfficiencyWarning`

### Sparse Matrices (without scipy)
- **In-place**: ⚠️ Yes, but may trigger `SparseEfficiencyWarning`
- **Fallback**: Direct assignment with warning about suboptimal performance

## Error Handling

### Missing Genes
```python
# Raises ValueError with specific missing gene names
set_gene_expression_for_target(adata, "NONEXISTENT", [0], 1.0)
# ValueError: Target genes not found in adata.var.index: ['NONEXISTENT']
```

### Backed AnnData
```python
# Raises RuntimeError for read-only data
adata.isbacked = True
set_gene_expression_for_target(adata, "GENE1", [0], 1.0)
# RuntimeError: Cannot modify backed AnnData object. The data is read-only.
```

## Performance Characteristics

| Matrix Type | Scipy Available | Performance | SparseEfficiencyWarning |
|-------------|----------------|-------------|-------------------------|
| Dense       | Any            | Optimal     | No                      |
| Sparse CSR  | Yes            | Optimal     | No                      |
| Sparse CSC  | Yes            | Optimal     | No                      |
| Sparse      | No             | Suboptimal  | Possible                |

## Testing

The function includes comprehensive tests covering:

- Dense matrix operations
- Sparse matrix optimization
- Error condition handling
- Copy functionality
- Backed AnnData protection
- Dependency fallback behavior

Run tests with:
```bash
python tests/test_set_gene_expression.py
```

## Dependencies

### Required
- Python 3.7+

### Optional (for optimal performance)
- numpy: For efficient array operations
- scipy: For optimized sparse matrix handling
- anndata: For compatibility with AnnData objects

### Fallback Behavior
When optional dependencies are missing, the function provides fallback implementations with reduced performance but maintained functionality.
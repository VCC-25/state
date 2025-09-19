#!/usr/bin/env python3
"""
Example usage of set_gene_expression_for_target function.

This script demonstrates how to use the function for various scenarios.
"""

import sys
import os
from pathlib import Path

# Add the scripts directory to the path
scripts_dir = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))

from get_simple import set_gene_expression_for_target


def create_mock_adata():
    """Create a mock AnnData object for demonstration."""
    
    class SimpleMatrix:
        def __init__(self, data):
            self.data = data
            self.shape = (len(data), len(data[0]) if data else 0)
            
        def __getitem__(self, key):
            if isinstance(key, tuple) and len(key) == 2:
                row, col = key
                return self.data[row][col]
            return self.data[key]
        
        def __setitem__(self, key, value):
            if isinstance(key, tuple) and len(key) == 2:
                row, col = key
                self.data[row][col] = value
            else:
                self.data[key] = value
        
        def copy(self):
            return SimpleMatrix([row[:] for row in self.data])
    
    class SimpleVar:
        def __init__(self, names):
            self.index = SimpleIndex(names)
    
    class SimpleIndex:
        def __init__(self, names):
            self._names = list(names)
        
        def __contains__(self, item):
            return item in self._names
        
        def get_loc(self, item):
            return self._names.index(item)
    
    class SimpleAnnData:
        def __init__(self, X, var_names):
            self.X = X
            self.var = SimpleVar(var_names)
            self.n_obs = X.shape[0]
            self.isbacked = False
        
        def copy(self):
            return SimpleAnnData(self.X.copy(), self.var.index._names[:])
    
    # Create sample gene expression data
    expression_data = [
        [1.5, 2.3, 0.8, 4.1, 0.0],  # Cell 0
        [2.1, 0.0, 1.9, 3.2, 2.5],  # Cell 1
        [0.7, 1.8, 2.4, 0.0, 1.3],  # Cell 2
        [3.1, 2.9, 0.5, 2.8, 3.4],  # Cell 3
        [1.2, 0.0, 3.1, 1.9, 2.2],  # Cell 4
    ]
    
    gene_names = ["TP53", "EGFR", "BRCA1", "MYC", "GAPDH"]
    
    X = SimpleMatrix(expression_data)
    adata = SimpleAnnData(X, gene_names)
    
    return adata


def print_matrix(adata, title="Expression Matrix"):
    """Print the expression matrix in a readable format."""
    print(f"\n{title}:")
    print("       " + "  ".join(f"{gene:>6}" for gene in adata.var.index._names))
    for i in range(adata.n_obs):
        row_values = "  ".join(f"{adata.X[i, j]:6.1f}" for j in range(len(adata.var.index._names)))
        print(f"Cell{i}  {row_values}")


def example_basic_assignment():
    """Demonstrate basic gene expression assignment."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Assignment")
    print("="*60)
    
    adata = create_mock_adata()
    print_matrix(adata, "Original Expression Matrix")
    
    # Set TP53 expression to 10.0 for cells 0 and 1
    print("\nSetting TP53 expression to 10.0 for cells 0 and 1...")
    set_gene_expression_for_target(adata, "TP53", [0, 1], 10.0)
    
    print_matrix(adata, "After TP53 Modification")


def example_multiple_genes():
    """Demonstrate multiple gene assignment."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Multiple Gene Assignment")
    print("="*60)
    
    adata = create_mock_adata()
    print_matrix(adata, "Original Expression Matrix")
    
    # Set multiple genes for specific cells
    print("\nSetting EGFR and BRCA1 expression for cells 2 and 3...")
    print("Values: Cell2=[5.0, 6.0], Cell3=[7.0, 8.0]")
    
    # Note: In real usage with numpy, you'd use np.array([[5.0, 6.0], [7.0, 8.0]])
    # Here we simulate with nested lists
    values = [[5.0, 6.0], [7.0, 8.0]]
    set_gene_expression_for_target(adata, ["EGFR", "BRCA1"], [2, 3], values)
    
    print_matrix(adata, "After Multiple Gene Modification")


def example_copy_functionality():
    """Demonstrate copy functionality."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Copy Functionality")
    print("="*60)
    
    adata = create_mock_adata()
    print_matrix(adata, "Original Expression Matrix")
    
    # Create a modified copy without changing the original
    print("\nCreating modified copy with MYC=99.0 for cell 0...")
    adata_modified = set_gene_expression_for_target(adata, "MYC", [0], 99.0, copy=True)
    
    print_matrix(adata, "Original (unchanged)")
    print_matrix(adata_modified, "Modified Copy")


def example_error_handling():
    """Demonstrate error handling."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Error Handling")
    print("="*60)
    
    adata = create_mock_adata()
    
    # Test missing gene error
    print("\nTesting missing gene error...")
    try:
        set_gene_expression_for_target(adata, "NONEXISTENT_GENE", [0], 1.0)
    except ValueError as e:
        print(f"✓ Caught expected error: {e}")
    
    # Test backed AnnData error
    print("\nTesting backed AnnData error...")
    adata.isbacked = True
    try:
        set_gene_expression_for_target(adata, "TP53", [0], 1.0)
    except RuntimeError as e:
        print(f"✓ Caught expected error: {e}")


def example_slice_indexing():
    """Demonstrate slice-based cell indexing."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Slice Indexing")
    print("="*60)
    
    adata = create_mock_adata()
    print_matrix(adata, "Original Expression Matrix")
    
    # Use slice to select cells 1-3
    print("\nSetting GAPDH expression to 15.0 for cells 1-3 (using slice)...")
    set_gene_expression_for_target(adata, "GAPDH", slice(1, 4), 15.0)
    
    print_matrix(adata, "After Slice-based Modification")


def main():
    """Run all examples."""
    print("set_gene_expression_for_target Function Examples")
    print("=" * 60)
    
    try:
        example_basic_assignment()
        example_multiple_genes()
        example_copy_functionality()
        example_error_handling()
        example_slice_indexing()
        
        print("\n" + "="*60)
        print("✅ All examples completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
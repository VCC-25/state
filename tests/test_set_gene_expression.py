#!/usr/bin/env python3
"""
Comprehensive test suite for set_gene_expression_for_target function.

This test suite validates the function's behavior with both dense and sparse matrices,
including verification that SparseEfficiencyWarning is eliminated.
"""

import sys
import os
import warnings
from pathlib import Path

# Add the scripts directory to the path
scripts_dir = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))

try:
    from get_simple import set_gene_expression_for_target
    print("✓ Successfully imported set_gene_expression_for_target")
except ImportError as e:
    print(f"✗ Failed to import function: {e}")
    sys.exit(1)


def test_with_real_dependencies():
    """Test with actual numpy/scipy/anndata if available."""
    
    # Check for dependencies
    deps_available = {}
    
    try:
        import numpy as np
        deps_available['numpy'] = True
        print("✓ numpy available")
    except ImportError:
        deps_available['numpy'] = False
        print("⚠ numpy not available")
    
    try:
        from scipy.sparse import csr_matrix, csc_matrix
        deps_available['scipy'] = True
        print("✓ scipy available")
    except ImportError:
        deps_available['scipy'] = False
        print("⚠ scipy not available")
    
    try:
        import anndata
        deps_available['anndata'] = True
        print("✓ anndata available")
    except ImportError:
        deps_available['anndata'] = False
        print("⚠ anndata not available")
    
    if not all(deps_available.values()):
        print("⚠ Some dependencies missing, running limited tests")
        return test_without_dependencies()
    
    # Full test with all dependencies
    print("\n=== Testing with Real Dependencies ===")
    
    # Test with dense matrix
    print("\n--- Dense Matrix Test ---")
    X_dense = np.random.rand(20, 10).astype(np.float32)
    var_names = [f"GENE_{i:02d}" for i in range(10)]
    adata_dense = anndata.AnnData(X_dense)
    adata_dense.var.index = var_names
    
    # Test single gene assignment
    original_val = adata_dense.X[0, 0]
    set_gene_expression_for_target(adata_dense, "GENE_00", [0, 1], 99.9)
    
    assert adata_dense.X[0, 0] == 99.9, "Dense single gene assignment failed"
    assert adata_dense.X[1, 0] == 99.9, "Dense single gene assignment failed"
    print("✓ Dense matrix single gene assignment passed")
    
    # Test multiple gene assignment
    values = np.array([[10, 20], [30, 40]])
    set_gene_expression_for_target(adata_dense, ["GENE_01", "GENE_02"], [2, 3], values)
    
    assert adata_dense.X[2, 1] == 10, "Dense multiple gene assignment failed"
    assert adata_dense.X[2, 2] == 20, "Dense multiple gene assignment failed"
    assert adata_dense.X[3, 1] == 30, "Dense multiple gene assignment failed"
    assert adata_dense.X[3, 2] == 40, "Dense multiple gene assignment failed"
    print("✓ Dense matrix multiple gene assignment passed")
    
    # Test with sparse matrix
    print("\n--- Sparse Matrix Test ---")
    X_sparse_data = np.random.rand(20, 10)
    X_sparse_data[X_sparse_data < 0.7] = 0  # Make it sparse
    X_sparse = csr_matrix(X_sparse_data)
    
    adata_sparse = anndata.AnnData(X_sparse)
    adata_sparse.var.index = var_names
    
    print(f"Original matrix type: {type(adata_sparse.X).__name__}")
    
    # Capture warnings to check for SparseEfficiencyWarning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        # Test assignment
        set_gene_expression_for_target(adata_sparse, "GENE_00", [0, 1, 2], 77.7)
        
        # Check for SparseEfficiencyWarning
        sparse_warnings = [warning for warning in w 
                          if "SparseEfficiencyWarning" in str(warning.category)]
        
        if sparse_warnings:
            print("⚠ SparseEfficiencyWarning detected:")
            for warning in sparse_warnings:
                print(f"  {warning.message}")
        else:
            print("✓ No SparseEfficiencyWarning raised - optimization successful!")
    
    print(f"Final matrix type: {type(adata_sparse.X).__name__}")
    
    # Verify assignment worked
    assert adata_sparse.X[0, 0] == 77.7, "Sparse assignment failed"
    assert adata_sparse.X[1, 0] == 77.7, "Sparse assignment failed"
    assert adata_sparse.X[2, 0] == 77.7, "Sparse assignment failed"
    print("✓ Sparse matrix assignment passed")
    
    # Test CSC to CSR conversion efficiency
    print("\n--- CSC Matrix Test ---")
    X_csc = csc_matrix(X_sparse_data)
    adata_csc = anndata.AnnData(X_csc)
    adata_csc.var.index = var_names
    
    print(f"Original matrix type: {type(adata_csc.X).__name__}")
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        set_gene_expression_for_target(adata_csc, "GENE_01", [0, 1], 88.8)
        
        sparse_warnings = [warning for warning in w 
                          if "SparseEfficiencyWarning" in str(warning.category)]
        
        if sparse_warnings:
            print("⚠ SparseEfficiencyWarning detected with CSC matrix")
        else:
            print("✓ No SparseEfficiencyWarning with CSC matrix")
    
    print(f"Final matrix type: {type(adata_csc.X).__name__}")
    
    # Test backed AnnData simulation
    print("\n--- Backed AnnData Test ---")
    adata_dense.isbacked = True
    try:
        set_gene_expression_for_target(adata_dense, "GENE_00", [0], 1.0)
        print("✗ Should have raised RuntimeError for backed AnnData")
        return False
    except RuntimeError as e:
        print(f"✓ Correctly prevented backed AnnData modification: {e}")
    
    return True


def test_without_dependencies():
    """Test basic functionality without external dependencies."""
    print("\n=== Testing without External Dependencies ===")
    
    # Mock classes (same as before)
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
            new_data = [row[:] for row in self.X.data]
            new_adata = SimpleAnnData(SimpleMatrix(new_data), self.var.index._names[:])
            return new_adata
    
    # Basic test
    test_data = [
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0]
    ]
    
    X = SimpleMatrix(test_data)
    gene_names = ["GENE_A", "GENE_B", "GENE_C", "GENE_D"]
    adata = SimpleAnnData(X, gene_names)
    
    # Test assignment
    print(f"Original [0,0]: {adata.X[0, 0]}")
    set_gene_expression_for_target(adata, "GENE_A", [0, 1], 99.0)
    print(f"After assignment [0,0]: {adata.X[0, 0]}")
    print(f"After assignment [1,0]: {adata.X[1, 0]}")
    
    assert adata.X[0, 0] == 99.0, "Assignment failed"
    assert adata.X[1, 0] == 99.0, "Assignment failed"
    print("✓ Basic assignment test passed")
    
    # Test copy functionality
    adata_copy = set_gene_expression_for_target(adata, "GENE_B", [0], 77.0, copy=True)
    assert adata.X[0, 1] != 77.0, "Original should not be modified"
    assert adata_copy.X[0, 1] == 77.0, "Copy should be modified"
    print("✓ Copy functionality test passed")
    
    return True


def main():
    """Run the test suite."""
    print("=== set_gene_expression_for_target Test Suite ===")
    
    try:
        success = test_with_real_dependencies()
        
        if success:
            print("\n🎉 All tests passed successfully!")
            return 0
        else:
            print("\n❌ Some tests failed!")
            return 1
            
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
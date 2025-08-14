"""
Test script for StateGraphBuilder.
Validates graph construction functionality with rigorous testing.
"""

import torch
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path
import logging
from state.tx.graphs.graph_construction import StateGraphBuilder

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_dense_graph_creation():
    """Test dense graph creation."""
    logger.info("Testing dense graph creation...")
    
    # Create test perturbation mapping
    pert2id = {f"gene_{i}": i for i in range(5)}
    
    # Initialize graph builder
    graph_builder = StateGraphBuilder(pert2id)
    
    # Create dense graph
    edge_index, edge_weight, num_nodes = graph_builder.create_graph("dense")
    
    # Validate output
    assert num_nodes == 5, f"Expected 5 nodes, got {num_nodes}"
    assert edge_index.shape[0] == 2, f"Expected 2 rows in edge_index, got {edge_index.shape[0]}"
    assert edge_index.shape[1] == 20, f"Expected 20 edges (5*4), got {edge_index.shape[1]}"
    assert edge_weight.shape[0] == 20, f"Expected 20 edge weights, got {edge_weight.shape[0]}"
    assert torch.all(edge_weight == 1.0), "All edge weights should be 1.0 for dense graph"
    
    logger.info("✅ Dense graph creation test passed")

def test_graph_processing():
    """Test network processing functionality."""
    logger.info("Testing graph processing...")
    
    # Create test perturbation mapping
    pert2id = {"gene_A": 0, "gene_B": 1, "gene_C": 2}
    
    # Create test network data
    network_data = {
        'regulator': ['gene_A', 'gene_B', 'gene_A', 'gene_C'],
        'target': ['gene_B', 'gene_C', 'gene_C', 'gene_A'],
        'weight': [0.8, 0.6, 0.9, 0.7]
    }
    network = pd.DataFrame(network_data)
    
    # Initialize graph builder
    graph_builder = StateGraphBuilder(pert2id)
    
    # Test processing with different arguments
    graph_args = {
        "reduce2perts": True,
        "reduce2positive": False,
        "norm_weights": True,
        "mode": "top_2"
    }
    
    edge_index, edge_weight, num_nodes = graph_builder._process_network(network, graph_args)
    
    # Validate output
    assert num_nodes == 3, f"Expected 3 nodes, got {num_nodes}"
    assert edge_index.shape[0] == 2, f"Expected 2 rows in edge_index, got {edge_index.shape[0]}"
    assert edge_weight.shape[0] == edge_index.shape[1], "Edge weights and indices should have same length"
    
    logger.info("✅ Graph processing test passed")

def test_fallback_behavior():
    """Test fallback behavior when graph files don't exist."""
    logger.info("Testing fallback behavior...")
    
    # Create test perturbation mapping
    pert2id = {"gene_1": 0, "gene_2": 1}
    
    # Create temporary directory without graph files
    with tempfile.TemporaryDirectory() as temp_dir:
        graph_builder = StateGraphBuilder(pert2id, cache_dir=temp_dir)
        
        # Test GO graph fallback
        edge_index, edge_weight, num_nodes = graph_builder.create_graph("go")
        assert num_nodes == 2, "Should create dense graph as fallback"
        
        # Test STRING graph fallback
        edge_index, edge_weight, num_nodes = graph_builder.create_graph("string")
        assert num_nodes == 2, "Should create dense graph as fallback"
        
        # Test scGPT graph fallback
        edge_index, edge_weight, num_nodes = graph_builder.create_graph("scgpt_derived")
        assert num_nodes == 2, "Should create dense graph as fallback"
    
    logger.info("✅ Fallback behavior test passed")

def test_graph_info():
    """Test graph information retrieval."""
    logger.info("Testing graph information...")
    
    pert2id = {"gene_A": 0, "gene_B": 1, "gene_C": 2}
    graph_builder = StateGraphBuilder(pert2id)
    
    info = graph_builder.get_graph_info()
    
    assert info["num_perturbations"] == 3
    assert info["perturbation_mapping"] == pert2id
    assert "dense" in info["available_graph_types"]
    
    logger.info("✅ Graph information test passed")

def test_error_handling():
    """Test error handling for invalid inputs."""
    logger.info("Testing error handling...")
    
    pert2id = {"gene_A": 0, "gene_B": 1}
    graph_builder = StateGraphBuilder(pert2id)
    
    # Test invalid graph type
    try:
        graph_builder.create_graph("invalid_type")
        assert False, "Should have raised ValueError"
    except ValueError:
        logger.info("✅ Correctly caught invalid graph type")
    
    # Test invalid network data
    invalid_network = pd.DataFrame({
        'invalid_col': ['gene_A', 'gene_B'],
        'another_invalid': [1, 2]
    })
    
    try:
        graph_builder._process_network(invalid_network)
        assert False, "Should have raised ValueError"
    except ValueError:
        logger.info("✅ Correctly caught invalid network format")
    
    logger.info("✅ Error handling test passed")

def test_scgpt_graph_creation():
    """Test scGPT graph creation with mock data."""
    logger.info("Testing scGPT graph creation...")
    
    pert2id = {"gene_A": 0, "gene_B": 1, "gene_C": 2}
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create mock scGPT embeddings
        experimental_dir = Path(temp_dir) / "experimental_data"
        experimental_dir.mkdir(exist_ok=True)
        
        # Create mock embeddings and gene names
        gene_names = ["gene_A", "gene_B", "gene_C", "gene_D"]
        embeddings = np.random.randn(4, 10)  # 4 genes, 10-dim embeddings
        
        np.save(experimental_dir / "gene_embeddings.npy", embeddings)
        np.save(experimental_dir / "gene_names.npy", gene_names)
        
        # Test scGPT graph creation
        graph_builder = StateGraphBuilder(pert2id, cache_dir=temp_dir)
        edge_index, edge_weight, num_nodes = graph_builder.create_graph("scgpt_derived")
        
        assert num_nodes == 3, f"Expected 3 nodes, got {num_nodes}"
        assert edge_index.shape[0] == 2, f"Expected 2 rows in edge_index, got {edge_index.shape[0]}"
        assert edge_weight.shape[0] == edge_index.shape[1], "Edge weights and indices should match"
        
        # Check that we have edges (should be 6 edges for 3 genes: 3*2)
        assert edge_index.shape[1] == 6, f"Expected 6 edges for 3 genes, got {edge_index.shape[1]}"
    
    logger.info("✅ scGPT graph creation test passed")

def run_all_tests():
    """Run all tests."""
    logger.info("🚀 Starting StateGraphBuilder tests...")
    
    try:
        test_dense_graph_creation()
        test_graph_processing()
        test_fallback_behavior()
        test_graph_info()
        test_error_handling()
        test_scgpt_graph_creation()
        
        logger.info("🎉 All tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1) 
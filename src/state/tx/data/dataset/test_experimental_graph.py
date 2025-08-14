"""
Test experimental graph construction from real data.
Uses random.h5ad to create experimental similarity graphs.
"""

import torch
import numpy as np
import logging
import anndata as ad
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import tempfile
import h5py
from state.tx.graphs.graph_construction import StateGraphBuilder
from state.tx.models.multigraph_perturbation import StateMultigraphPerturbationModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_experimental_graph_from_data():
    """Create experimental graph from real expression data."""
    logger.info("Creating experimental graph from real data...")
    
    # Load real data
    adata = ad.read_h5ad("examples/random.h5ad")
    
    # Extract expression data and gene names
    expression_data = adata.obsm['X_hvg']  # Shape: (10000, 11)
    gene_names = adata.var.index.tolist()  # ['GENE1', 'GENE2', ...]
    
    # Only use the first 11 genes to match the expression data
    gene_names = gene_names[:expression_data.shape[1]]  # Take only the first 11 genes
    
    logger.info(f"Expression data shape: {expression_data.shape}")
    logger.info(f"Gene names (using first {len(gene_names)}): {gene_names}")
    
    # Create experimental graph directory structure
    experimental_dir = Path("graphs/experimental_data")
    experimental_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute gene-gene similarity matrix
    # For this example, we'll treat each gene as a "perturbation" and compute similarities
    # In practice, you'd have gene embeddings from scGPT or similar
    
    # Simulate gene embeddings by averaging expression across cells for each gene
    gene_embeddings = expression_data.T  # Shape: (11, 10000) - each gene is a vector across cells
    
    # Compute pairwise cosine similarities between genes
    gene_similarities = cosine_similarity(gene_embeddings)
    
    # Take absolute values as in the paper
    abs_similarities = np.abs(gene_similarities)
    
    logger.info(f"Gene similarity matrix shape: {abs_similarities.shape}")
    logger.info(f"Similarity range: [{abs_similarities.min():.4f}, {abs_similarities.max():.4f}]")
    
    # Save experimental data
    np.save(experimental_dir / "gene_embeddings.npy", gene_embeddings)
    np.save(experimental_dir / "gene_names.npy", np.array(gene_names))
    np.save(experimental_dir / "gene_similarity_matrix.npy", abs_similarities)
    
    logger.info(f"Saved experimental data to {experimental_dir}")
    
    return gene_names, abs_similarities

def test_experimental_graph_construction():
    """Test building experimental graph from real data."""
    logger.info("Testing experimental graph construction...")
    
    # Create experimental graph data
    gene_names, similarities = create_experimental_graph_from_data()
    
    # Create perturbation mapping using gene names
    pert2id = {gene: idx for idx, gene in enumerate(gene_names)}
    
    logger.info(f"Perturbation mapping: {pert2id}")
    
    # Create graph builder
    graph_builder = StateGraphBuilder(pert2id, cache_dir="./graphs")
    
    # Test scGPT-derived graph creation
    try:
        edge_index, edge_weight, num_nodes = graph_builder.create_graph("scgpt_derived")
        logger.info(f"✅ Created experimental graph: {num_nodes} nodes, {edge_index.shape[1]} edges")
        
        # Check that we have reasonable number of edges
        assert edge_index.shape[1] > 0, "Should have at least some edges"
        assert num_nodes == len(gene_names), f"Expected {len(gene_names)} nodes, got {num_nodes}"
        
        # Check edge weights
        logger.info(f"Edge weight range: [{edge_weight.min():.4f}, {edge_weight.max():.4f}]")
        assert not torch.isnan(edge_weight).any(), "Edge weights should not contain NaN"
        
    except Exception as e:
        logger.error(f"❌ Failed to create experimental graph: {e}")
        raise
    
    logger.info("✅ Experimental graph construction test passed")

def test_multigraph_model_with_experimental_graph():
    """Test multigraph model with experimental graph."""
    logger.info("Testing multigraph model with experimental graph...")
    
    # Create experimental graph data
    gene_names, similarities = create_experimental_graph_from_data()
    
    # Create perturbation mapping
    pert2id = {gene: idx for idx, gene in enumerate(gene_names)}
    
    # Create graph builder
    graph_builder = StateGraphBuilder(pert2id, cache_dir="./graphs")
    
    # Create multigraph model with experimental graph
    model = StateMultigraphPerturbationModel(
        graph_builder=graph_builder,
        input_dim=32,
        hidden_dim=64,
        output_dim=32,
        graph_config={
            "experimental_graph": {
                "type": "scgpt_derived",
                "args": {"mode": "top_5"}
            }
        },
        device="cpu"
    )
    
    # Test model with real perturbation indices
    batch_size = 4
    pert_indices = torch.randint(0, len(pert2id), (batch_size,))
    
    logger.info(f"Input perturbation indices: {pert_indices.tolist()}")
    logger.info(f"Corresponding genes: {[gene_names[i] for i in pert_indices.tolist()]}")
    
    # Forward pass
    output = model(pert_indices)
    
    logger.info(f"Model output shape: {output.shape}")
    logger.info(f"Output statistics - Mean: {output.mean():.4f}, Std: {output.std():.4f}")
    
    # Validate output
    assert output.shape == (batch_size, 32), f"Expected shape (4, 32), got {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains infinite values"
    
    # Test gradient flow
    target = torch.randn_like(output)
    loss = torch.nn.functional.mse_loss(output, target)
    loss.backward()
    
    # Check gradients
    has_gradients = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_gradients = True
            logger.info(f"Parameter {name} has gradients")
            break
    
    assert has_gradients, "No gradients found in model parameters"
    
    # Test model info
    info = model.get_graph_info()
    logger.info(f"Model info: {info}")
    
    assert info["num_graphs"] >= 1, "Should have at least one graph"
    assert info["num_perturbations"] == len(gene_names)
    
    logger.info("✅ Multigraph model with experimental graph test passed")

def test_experimental_graph_analysis():
    """Analyze the experimental graph properties."""
    logger.info("Analyzing experimental graph properties...")
    
    # Create experimental graph data
    gene_names, similarities = create_experimental_graph_from_data()
    
    # Analyze similarity matrix
    logger.info(f"Similarity matrix statistics:")
    logger.info(f"  - Shape: {similarities.shape}")
    logger.info(f"  - Mean: {similarities.mean():.4f}")
    logger.info(f"  - Std: {similarities.std():.4f}")
    logger.info(f"  - Min: {similarities.min():.4f}")
    logger.info(f"  - Max: {similarities.max():.4f}")
    
    # Find most similar gene pairs
    np.fill_diagonal(similarities, 0)  # Remove self-similarities
    max_sim_idx = np.unravel_index(similarities.argmax(), similarities.shape)
    max_sim_value = similarities[max_sim_idx]
    
    logger.info(f"Most similar gene pair: {gene_names[max_sim_idx[0]]} - {gene_names[max_sim_idx[1]]} (similarity: {max_sim_value:.4f})")
    
    # Create perturbation mapping
    pert2id = {gene: idx for idx, gene in enumerate(gene_names)}
    
    # Create graph builder and test graph creation
    graph_builder = StateGraphBuilder(pert2id, cache_dir="./graphs")
    
    edge_index, edge_weight, num_nodes = graph_builder.create_graph("scgpt_derived")
    
    logger.info(f"Experimental graph properties:")
    logger.info(f"  - Nodes: {num_nodes}")
    logger.info(f"  - Edges: {edge_index.shape[1]}")
    logger.info(f"  - Edge weight range: [{edge_weight.min():.4f}, {edge_weight.max():.4f}]")
    logger.info(f"  - Average edge weight: {edge_weight.mean():.4f}")
    
    # Check graph connectivity
    unique_nodes = torch.unique(edge_index).tolist()
    logger.info(f"  - Unique nodes in edges: {len(unique_nodes)}/{num_nodes}")
    
    assert len(unique_nodes) == num_nodes, "All nodes should be connected"
    
    logger.info("✅ Experimental graph analysis completed")

def run_all_tests():
    """Run all experimental graph tests."""
    logger.info("🚀 Starting experimental graph tests...")
    
    try:
        test_experimental_graph_construction()
        test_multigraph_model_with_experimental_graph()
        test_experimental_graph_analysis()
        
        logger.info("🎉 All experimental graph tests passed!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise

if __name__ == "__main__":
    run_all_tests() 
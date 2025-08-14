"""
Test script for graph integration in STATE Embedding model.
Validates that graph embeddings are properly integrated with ESM embeddings.
"""

import torch
import logging
import tempfile
import os
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from state.emb.nn.model import StateEmbeddingModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_graph_integration_config():
    """Create a test configuration with graph integration enabled."""
    config = {
        "name": "test_graph_integration",
        "output_dir": "./test_output",
        "overwrite": True,
        "use_wandb": False,
        "model": {
            "name": "StateEmbeddingModel",
            "use_graph_embeddings": True,
            "graph_dim": 64,
            "graph_loss_weight": 0.1,
            "graph_config": {
                "experimental_graph": {
                    "type": "scgpt_derived",
                    "args": {"mode": "top_5"}
                },
                "string_graph": {
                    "type": "string",
                    "args": {"mode": "top_10", "threshold": 0.4}
                },
                "go_graph": {
                    "type": "go",
                    "args": {"mode": "biological_process", "top_k": 50}
                },
                "reactome_graph": {
                    "type": "reactome",
                    "args": {"mode": "pathway", "top_k": 30}
                }
            },
            "token_dim": 5120,
            "d_model": 512,
            "nhead": 8,
            "d_hid": 2048,
            "nlayers": 2,
            "output_dim": 128,
            "dropout": 0.1,
            "batch_size": 16,
            "max_lr": 4e-4,
            "warmup_steps": 100,
            "loss": {"name": "tabular"},
            "graph_cache_dir": "./graphs",
            "graph_preprocessing": True,
            "graph_memory_efficient": True,
            # Add missing required keys
            "rda": False,
            "counts": True,
            "dataset_correction": False,
            "num_downsample": 1,
            "use_flash_attention": False
        },
        "dataset": {
            "current": "default",
            "default": {
                "num_cells": 1000,
                "seed": 42,  # Add missing seed key
                "pad_length": 512,
                "P": 64,
                "N": 64
            }
        },
        "training": {
            "batch_size": 16,
            "max_steps": 100,
            "val_freq": 50,
            "ckpt_every_n_steps": 50,
            "gradient_clip_val": 1.0,
            # Device configuration - will auto-detect MPS/GPU/CPU
            "devices": "auto",  # Will use MPS on M1, GPU on CUDA systems, CPU otherwise
            "strategy": "auto",  # Auto-select best strategy for device
            "accelerator": "auto",  # Auto-detect accelerator type
            "precision": "auto"  # Auto-select precision (16-bit for GPU, 32-bit for MPS/CPU)
        },
        # Use real ESM embeddings file
        "embeddings": {
            "current": "default",
            "default": {
                "size": 5120,
                "num": 145469,
                "all_embeddings": "/Users/mukulsherekar/Projects/STATE-TXPERT/STATE/state/embeddings/ESM2_pert_features.pt",
                "ds_emb_mapping": "./mock_embeddings/ds_emb_mapping_5120.pt",  # Fixed path without {size}
                "valid_genes_masks": "./mock_embeddings/valid_genes_masks.pt"
            }
        },
        # Add missing required sections
        "task": {
            "mask": 0.15
        },
        "loss": {
            "name": "tabular",
            "normalization": False
        },
        # Experiment configuration (required by _fit.py)
        "experiment": {
            "name": "test_graph_integration",
            "port": 12355,
            "num_gpus_per_node": 1,
            "ddp_timeout": 1800,  # Distributed training timeout in seconds
            # Device-specific settings
            "device_type": "auto",  # auto, mps, cuda, cpu
            "mixed_precision": True,  # Enable mixed precision for faster training
            "compile_model": False  # Disable torch.compile for MPS compatibility
        }
    }
    return DictConfig(config)

def create_mock_embeddings():
    """Create mock dataset embeddings for testing."""
    import os
    
    # Create mock embeddings directory
    mock_dir = "./mock_embeddings"
    os.makedirs(mock_dir, exist_ok=True)
    
    # Create mock ds_emb_mapping
    mock_ds_mapping = {
        "dataset1": torch.randint(0, 5120, (1000,)),
        "dataset2": torch.randint(0, 5120, (1000,)),
    }
    torch.save(mock_ds_mapping, os.path.join(mock_dir, "ds_emb_mapping_5120.pt"))
    
    # Create mock valid_genes_masks
    mock_masks = {
        "dataset1": torch.ones(1000, dtype=torch.bool),
        "dataset2": torch.ones(1000, dtype=torch.bool),
    }
    torch.save(mock_masks, os.path.join(mock_dir, "valid_genes_masks.pt"))
    
    logger.info(f"✅ Created mock dataset embeddings in {mock_dir}")

def test_model_creation_with_graph():
    """Test model creation with graph integration enabled."""
    logger.info("Testing model creation with graph integration...")
    
    try:
        # Create mock dataset embeddings first
        create_mock_embeddings()
        
        # Create configuration
        config = create_graph_integration_config()
        
        # Create model
        model = StateEmbeddingModel(
            token_dim=config.model.token_dim,
            d_model=config.model.d_model,
            nhead=config.model.nhead,
            d_hid=config.model.d_hid,
            nlayers=config.model.nlayers,
            output_dim=config.model.output_dim,
            dropout=config.model.dropout,
            warmup_steps=config.model.warmup_steps,
            max_lr=config.model.max_lr,
            cfg=config
        )
        
        logger.info("✅ Model created successfully with graph integration")
        
        # Test model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"✅ Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        # Verify graph integration components
        assert hasattr(model, 'use_graph_embeddings'), "Graph embeddings flag not set"
        assert model.use_graph_embeddings, "Graph embeddings not enabled"
        assert hasattr(model, 'graph_builder'), "Graph builder not initialized"
        assert hasattr(model, 'graph_dim'), "Graph dimension not set"
        
        logger.info(f"✅ Graph integration components verified:")
        logger.info(f"   - Graph embeddings enabled: {model.use_graph_embeddings}")
        logger.info(f"   - Graph dimension: {model.graph_dim}")
        logger.info(f"   - Graph builder initialized: {model.graph_builder is not None}")
        logger.info(f"   - Number of graph types: {len(config.model.graph_config)}")
        
        return model
        
    except Exception as e:
        logger.error(f"❌ Model creation failed: {e}")
        raise

def test_gene_embedding_with_graph():
    """Test gene embedding with graph integration."""
    logger.info("Testing gene embedding with graph integration...")
    
    try:
        # Create model
        model = test_model_creation_with_graph()
        
        # Test genes - use real gene names from ESM embeddings
        # These are actual genes that exist in the ESM embeddings file
        test_genes = ["A1BG", "A2M", "AAAS", "AACS", "AADAC"]
        
        # Test gene embedding
        gene_embeddings = model.get_gene_embedding(test_genes)
        
        logger.info(f"✅ Gene embeddings shape: {gene_embeddings.shape}")
        logger.info(f"✅ Expected shape: ({len(test_genes)}, {model.d_model})")
        
        # Verify embeddings are not all zeros
        assert not torch.allclose(gene_embeddings, torch.zeros_like(gene_embeddings)), "All embeddings are zero"
        
        logger.info("✅ Gene embeddings computed successfully")
        
        return model, gene_embeddings
        
    except Exception as e:
        logger.error(f"❌ Gene embedding test failed: {e}")
        raise

def test_forward_pass_with_graph():
    """Test forward pass with graph integration."""
    logger.info("Testing forward pass with graph integration...")
    
    try:
        # Create model
        model = test_model_creation_with_graph()
        
        # Initialize positional encoding (this is normally done in trainer)
        if model.pe_embedding is None:
            # Create a simple positional encoding for testing
            from state.emb.nn.model import PositionalEncoding
            model.pe_embedding = PositionalEncoding(model.d_model, dropout=0.1)
        
        # Create mock input data
        batch_size = 4
        seq_len = 256
        token_dim = 5120
        
        # Mock input tensors
        src = torch.randn(batch_size, seq_len, token_dim)
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        counts = torch.randn(batch_size, seq_len)
        
        # Test forward pass with real gene names from ESM embeddings
        test_genes = ["A1BG", "A2M", "AAAS", "AACS", "AADAC"]
        
        model.eval()
        with torch.no_grad():
            gene_output, embedding, dataset_emb = model(src, mask, counts, gene_names=test_genes)
        
        logger.info(f"✅ Forward pass completed successfully:")
        logger.info(f"   - Gene output shape: {gene_output.shape}")
        logger.info(f"   - Embedding shape: {embedding.shape}")
        logger.info(f"   - Dataset embedding shape: {dataset_emb.shape if dataset_emb is not None else 'None'}")
        
        # Verify output shapes
        expected_gene_output_shape = (batch_size, model.output_dim)  # output_dim = 128
        expected_embedding_shape = (batch_size, model.d_model)  # d_model = 512
        
        assert gene_output.shape == expected_gene_output_shape, f"Gene output shape mismatch: {gene_output.shape} vs {expected_gene_output_shape}"
        assert embedding.shape == expected_embedding_shape, f"Embedding shape mismatch: {embedding.shape} vs {expected_embedding_shape}"
        
        logger.info("✅ Forward pass shapes verified")
        
    except Exception as e:
        logger.error(f"❌ Forward pass test failed: {e}")
        raise

def test_graph_consistency_loss():
    """Test graph consistency loss computation."""
    logger.info("Testing graph consistency loss...")
    
    try:
        # Create model
        model = test_model_creation_with_graph()
        
        # Create mock embeddings and batch
        batch_size = 4
        embedding_dim = 128
        embs = torch.randn(batch_size, embedding_dim)
        
        # Mock batch (simplified)
        batch = {
            "genes": ["GENE1", "GENE2", "GENE3", "GENE4"],
            "embeddings": embs
        }
        
        # Test graph consistency loss
        graph_loss = model.compute_graph_consistency_loss(embs, batch)
        
        logger.info(f"✅ Graph consistency loss computed: {graph_loss.item():.6f}")
        
        # Verify loss is a scalar tensor
        assert graph_loss.dim() == 0, "Graph loss should be a scalar"
        assert graph_loss.item() >= 0, "Graph loss should be non-negative"
        
        logger.info("✅ Graph consistency loss verified")
        
    except Exception as e:
        logger.error(f"❌ Graph consistency loss test failed: {e}")
        raise

def test_multiple_graph_types():
    """Test that multiple graph types are properly integrated."""
    logger.info("Testing multiple graph types integration...")
    
    try:
        # Create model
        model = test_model_creation_with_graph()
        
        # Test genes - use real gene names from ESM embeddings
        test_genes = ["A1BG", "A2M", "AAAS", "AACS", "AADAC"]
        
        # Test graph embeddings with multiple graph types
        graph_embeddings = model.get_graph_embeddings(test_genes)
        
        logger.info(f"✅ Multi-graph embeddings shape: {graph_embeddings.shape}")
        logger.info(f"✅ Expected shape: ({len(test_genes)}, {model.graph_dim})")
        
        # Verify embeddings are not all zeros
        assert not torch.allclose(graph_embeddings, torch.zeros_like(graph_embeddings)), "All graph embeddings are zero"
        
        logger.info("✅ Multiple graph types integration verified")
        
    except Exception as e:
        logger.error(f"❌ Multiple graph types test failed: {e}")
        raise

def test_end_to_end_integration():
    """Test end-to-end graph integration."""
    logger.info("Testing end-to-end graph integration...")
    
    try:
        # Test all components
        test_model_creation_with_graph()
        test_gene_embedding_with_graph()
        test_forward_pass_with_graph()
        test_graph_consistency_loss()
        test_multiple_graph_types()  # Added this line
        
        logger.info("✅ All graph integration tests passed!")
        
    except Exception as e:
        logger.error(f"❌ End-to-end integration test failed: {e}")
        raise

def run_all_tests():
    """Run all graph integration tests."""
    logger.info("🚀 Starting graph integration tests for STATE Embedding model...")
    
    try:
        test_end_to_end_integration()
        
        logger.info("🎉 All graph integration tests passed!")
        
    except Exception as e:
        logger.error(f"❌ Graph integration test failed: {e}")
        raise

if __name__ == "__main__":
    run_all_tests() 
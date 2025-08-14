"""
Training script for STATE Embedding model with graph integration.
Trains the SE model using both ESM and graph embeddings.
"""

import torch
import logging
import os
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from state.emb.nn.model import StateEmbeddingModel
from state.emb.data import create_dataloader
from state.emb.train.trainer import get_embeddings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_training_config():
    """Create training configuration with graph integration."""
    config = {
        "name": "state_emb_graph_training",
        "output_dir": "./output/emb_graph_training",
        "overwrite": True,
        "use_wandb": True,
        "model": {
            "name": "StateEmbeddingModel",
            "use_graph_embeddings": True,
            "graph_dim": 64,
            "graph_loss_weight": 0.1,
            "graph_config": {
                "experimental_graph": {
                    "type": "scgpt_derived",
                    "args": {"mode": "top_5"}
                }
            },
            "token_dim": 5120,
            "d_model": 512,
            "nhead": 8,
            "d_hid": 2048,
            "nlayers": 6,
            "output_dim": 128,
            "dropout": 0.1,
            "batch_size": 32,
            "max_lr": 4e-4,
            "warmup_steps": 1000,
            "loss": {"name": "tabular"},
            "graph_cache_dir": "./graphs",
            "graph_preprocessing": True,
            "graph_memory_efficient": True,
            "rda": False,
            "counts": True,
            "dataset_correction": False,
            "num_downsample": 1,
            "use_flash_attention": False
        },
        "dataset": {
            "pad_length": 512,
            "P": 64,
            "N": 64,
            "S": 128,
            "cls_token_idx": 0
        },
        "training": {
            "batch_size": 32,
            "max_steps": 10000,
            "val_freq": 100,
            "ckpt_every_n_steps": 500,
            "gradient_clip_val": 1.0,
            "devices": 1,
            "strategy": "auto"
        },
        "embeddings": {
            "current": "default",
            "default": {
                "size": 5120,
                "num": 145469,
                "all_embeddings": "/Users/mukulsherekar/Projects/STATE-TXPERT/STATE/state/embeddings/ESM2_pert_features.pt",
                "ds_emb_mapping": "./mock_embeddings/ds_emb_mapping_{size}.pt",
                "valid_genes_masks": "./mock_embeddings/valid_genes_masks.pt"
            }
        },
        "task": {
            "mask": 0.15
        },
        "loss": {
            "name": "tabular",
            "normalization": False
        },
        "data": {
            "train": "./data/train.h5ad",
            "val": "./data/val.h5ad",
            "filter_by_species": "human"
        }
    }
    return DictConfig(config)

def create_mock_dataset_embeddings():
    """Create mock dataset embeddings for training."""
    import os
    import torch
    
    mock_dir = "./mock_embeddings"
    os.makedirs(mock_dir, exist_ok=True)
    
    # Load real ESM embeddings to get gene names
    esm_path = "/Users/mukulsherekar/Projects/STATE-TXPERT/STATE/state/embeddings/ESM2_pert_features.pt"
    esm_data = torch.load(esm_path, weights_only=False)
    gene_names = list(esm_data.keys())
    
    # Create dataset mapping using real gene names
    # Map each gene to its index in the ESM embeddings
    gene_to_idx = {gene: idx for idx, gene in enumerate(gene_names)}
    
    # Create mock ds_emb_mapping for the 'random' dataset
    # Use the first 50 genes from ESM embeddings
    random_genes = gene_names[:50]
    random_indices = [gene_to_idx[gene] for gene in random_genes]
    
    mock_ds_mapping = {
        "random": torch.tensor(random_indices, dtype=torch.long)
    }
    torch.save(mock_ds_mapping, os.path.join(mock_dir, "ds_emb_mapping_5120.pt"))
    
    # Create mock valid_genes_masks for the 'random' dataset
    mock_masks = {
        "random": torch.ones(50, dtype=torch.bool)  # All 50 genes are valid
    }
    torch.save(mock_masks, os.path.join(mock_dir, "valid_genes_masks.pt"))
    
    logger.info(f"✅ Created dataset embeddings using real ESM data")
    logger.info(f"✅ Dataset names: {list(mock_ds_mapping.keys())}")
    logger.info(f"✅ Gene mapping size: {mock_ds_mapping['random'].shape}")
    logger.info(f"✅ Using real gene names: {random_genes[:5]}...")

def train_se_model_with_graph():
    """Train the SE model with graph integration."""
    logger.info("🚀 Starting SE model training with graph integration...")
    
    try:
        # Create mock dataset embeddings
        create_mock_dataset_embeddings()
        
        # Create configuration
        config = create_training_config()
        
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
        
        logger.info("✅ Model created successfully")
        logger.info(f"✅ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(f"✅ Graph integration enabled: {model.use_graph_embeddings}")
        
        # Test gene embedding with graph
        test_genes = ["GAPDH", "ACTB", "TUBB", "HSP90AA1", "EEF1A1"]
        gene_embeddings = model.get_gene_embedding(test_genes)
        logger.info(f"✅ Gene embeddings shape: {gene_embeddings.shape}")
        
        # Test forward pass
        batch_size = 4
        seq_len = 256
        token_dim = 5120
        
        src = torch.randn(batch_size, seq_len, token_dim)
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        counts = torch.randn(batch_size, seq_len)
        
        model.eval()
        with torch.no_grad():
            gene_output, embedding, dataset_emb = model(src, mask, counts)
        
        logger.info(f"✅ Forward pass successful:")
        logger.info(f"   - Gene output: {gene_output.shape}")
        logger.info(f"   - Embedding: {embedding.shape}")
        
        # Test training step
        model.train()
        loss = model.shared_step({
            "src": src,
            "mask": mask,
            "counts": counts,
            "target": torch.randn(batch_size, seq_len, 128),
            "batch_weights": torch.ones(batch_size, seq_len)
        }, batch_idx=0)
        
        logger.info(f"✅ Training step successful, loss: {loss.item():.6f}")
        
        logger.info("🎉 SE model with graph integration is ready for training!")
        
        return model, config
        
    except Exception as e:
        logger.error(f"❌ Training setup failed: {e}")
        raise

def main():
    """Main training function."""
    logger.info("🚀 Starting SE model training with graph integration...")
    
    try:
        # Train the model
        model, config = train_se_model_with_graph()
        
        logger.info("✅ Training setup completed successfully!")
        logger.info("📝 Next steps:")
        logger.info("   1. Prepare your training data")
        logger.info("   2. Run: python -m state emb train --config examples/emb_graph_integration.yaml")
        logger.info("   3. Monitor training with graph integration")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    main() 
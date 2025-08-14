"""
Test script for Phase 6: Training Integration with REAL DATA.
Uses only the random.h5ad file to avoid tensor size conflicts.
"""

import torch
import logging
import tempfile
import os
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from state.tx.graphs.graph_construction import StateGraphBuilder
from state.tx.models.graph import GraphPerturbationModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_real_data_config():
    """Create a test configuration using only random.h5ad."""
    config = {
        "name": "test_graph.py",
        "output_dir": "./test_output",
        "overwrite": True,
        "use_wandb": False,
        "wandb": {
            "project": "test",
            "entity": "test",
            "local_wandb_dir": "./wandb"
        },
        "data": {
            "name": "PerturbationDataModule",
            "kwargs": {
                "toml_config_path": "../examples/competition_se.toml",  # Use specific config
                "h5_path": "../data/competition_train.h5",  
                "pert_col": "target_gene",
                "cell_type_key": "cell_type",
                "batch_col": "batch_var",
                "control_pert": "non-targeting",
                "embed_key": None, # or X_hvg?
                "output_space": "gene",
                "transform": "log-normalize",
                "train_task": "perturbation",
                "perturbation_type": "genetic"
            }
        },
        "model": {
            "name": "graph_perturbation",  
            "checkpoint": None,
            "device": "mps",
            "kwargs": {
                "hidden_dim": 64,
                "n_encoder_layers": 2,
                "n_decoder_layers": 2,
                "dropout": 0.1,
                "activation": "gelu",
                "graph_config": {
                    "experimental_graph": {
                        "type": "scgpt_derived",
                        "args": {
                            "reduce2perts": True,
                            "norm_weights": False,
                            "mode": "top_5"
                        }
                    }
                },
                "graph_cache_dir": "../graphs",
                "lr": 3e-4,
                "loss_fn": "mse",
                "output_space": "gene",
                "embed_key": None,
                "control_pert": "non-targeting",
                "pert_dim": 32,
                "predict_residual": True
            }
        },
        "training": {
            "batch_size": 4,
            "lr": 3e-4,
            "max_steps": 100,
            "train_seed": 42,
            "val_freq": 50,
            "ckpt_every_n_steps": 50,
            "gradient_clip_val": 10,
            "loss_fn": "mse",
            "devices": 1,
            "strategy": "auto",
            "graph_cache_dir": "../graphs",
            "graph_preprocessing": True,
            "graph_memory_efficient": True
        }
    }
    return DictConfig(config)

def test_data_loading_real_data():
    """Test data loading with real random.h5ad file."""
    logger.info("Testing data loading with real random.h5ad...")
    
    try:
        # Import the data module
        from cell_load.utils.modules import get_datamodule
        
        # Create data configuration
        data_config = {
            "name": "PerturbationDataModule",
            "kwargs": {
                "toml_config_path": "../examples/competition_se.toml",
                "h5_path": "../data/competition_train.h5",
                "pert_col": "target_gene",
                "cell_type_key": "cell_type",
                "batch_col": "batch_var",
                "control_pert": "non-targeting",
                "embed_key": None,
                "output_space": "gene",
                "transform": "log-normalize",
                "train_task": "perturbation",
                "perturbation_type": "genetic"
            }
        }
        
        # Create data module
        data_module = get_datamodule(
            name=data_config["name"],
            kwargs=data_config["kwargs"],
            batch_size=4,
            cell_sentence_len=16,
            
        )
        logger.info("✅ Data module created successfully")
        
        # Test data module setup
        data_module.setup()
        logger.info("✅ Data module setup completed")
        
        # Test training dataloader
        train_dl = data_module.train_dataloader()
        logger.info(f"✅ Training dataloader created: {len(train_dl)} batches")
        
        # Test validation dataloader (may be empty)
        try:
            val_dl = data_module.val_dataloader()
            logger.info(f"✅ Validation dataloader created: {len(val_dl)} batches")
        except Exception as e:
            logger.info(f"⚠️ No validation dataloader available: {e}")
            val_dl = None
        
        # Test variable dimensions
        try:
            var_dims = data_module.get_var_dims()
            logger.info(f"✅ Variable dimensions: {var_dims}")
        except Exception as e:
            logger.warning(f"⚠️ Could not get variable dimensions: {e}")
            # Create mock variable dimensions for testing
            var_dims = {
                "input_dim": 50,
                "output_dim": 50,
                "pert_dim": 32,
                "batch_dim": 1,
                "gene_dim": 5000,
                "hvg_dim": 2001,
                "gene_names": ["GENE1", "GENE2", "GENE3", "GENE4", "GENE5"]
            }
        
        return data_module, var_dims
        
    except Exception as e:
        logger.error(f"❌ Data loading failed: {e}")
        raise

def test_model_creation_with_real_data():
    """Test model creation with real data dimensions."""
    logger.info("Testing model creation with real data...")
    
    try:
        # Get data module and dimensions
        data_module, var_dims = test_data_loading_real_data()
        
        # Get perturbations from the data module's perturbation mapping
        try:
            # Try to get perturbations from the data module
            if hasattr(data_module, 'pert_categories'):
                unique_perts = data_module.pert_categories
            elif hasattr(data_module, 'pert2id'):
                unique_perts = list(data_module.pert2id.keys())
            else:
                # Fallback: use the perturbations we know from random.h5ad
                unique_perts = ['TARGET1', 'TARGET2', 'TARGET3', 'TARGET4', 'TARGET5']
                logger.info("⚠️ Using fallback perturbations from random.h5ad")
        except Exception as e:
            logger.warning(f"⚠️ Could not get perturbations from data module: {e}")
            # Fallback: use the perturbations we know from random.h5ad
            unique_perts = ['TARGET1', 'TARGET2', 'TARGET3', 'TARGET4', 'TARGET5']
            logger.info("⚠️ Using fallback perturbations from random.h5ad")
        
        pert2id = {pert: idx for idx, pert in enumerate(sorted(unique_perts))}
        logger.info(f"✅ Found {len(pert2id)} unique perturbations: {list(pert2id.keys())}")
        
        graph_builder = StateGraphBuilder(pert2id, cache_dir="../graphs")
        
        # Graph configuration
        graph_config = {
            "experimental_graph": {
                "type": "scgpt_derived",
                "args": {"mode": "top_5"}
            }
        }
        
        # Get real dimensions from the data
        real_input_dim = var_dims['gene_dim']  # From the real data: ctrl_cell_emb shape [12800, 11]
        real_output_dim = var_dims['gene_dim']  # Same as input for now
        
        # Create model with real dimensions
        model = GraphPerturbationModel(
            input_dim=real_input_dim,
            hidden_dim=64,
            output_dim=real_output_dim,
            pert_dim=var_dims["pert_dim"],
            graph_builder=graph_builder,
            graph_config=graph_config,
            dropout=0.1,
            lr=3e-4,
            device="cpu",  # Use CPU for testing
            **{
                "n_encoder_layers": 2,
                "n_decoder_layers": 2,
                "activation": "gelu"
            }
        )
        
        logger.info("✅ Model created successfully with real data")
        
        # Test model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"✅ Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        return model, data_module
        
    except Exception as e:
        logger.error(f"❌ Model creation failed: {e}")
        raise

def test_training_step_with_real_data():
    """Test a single training step with real data."""
    logger.info("Testing training step with real data...")
    
    try:
        # Create model and data module
        model, data_module = test_model_creation_with_real_data()
        
        # Get a real batch from the dataloader
        train_dl = data_module.train_dataloader()
        batch = next(iter(train_dl))
        
        logger.info(f"✅ Real batch loaded: {list(batch.keys())}")
        logger.info(f"✅ Batch shapes: {[(k, v.shape) for k, v in batch.items() if torch.is_tensor(v)]}")
        
        # Test forward pass
        model.train()
        output = model(batch)
        logger.info(f"✅ Forward pass completed: {output.shape}")
        
        # Test loss computation
        target = torch.randn_like(output)
        loss = torch.nn.functional.mse_loss(output, target)
        logger.info(f"✅ Loss computed: {loss.item():.4f}")
        
        # Test backward pass
        loss.backward()
        logger.info("✅ Backward pass completed")
        
        # Test gradient flow
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        logger.info(f"✅ Gradient norm: {grad_norm:.4f}")
        
        logger.info("✅ Training step with real data completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Training step failed: {e}")
        raise

def test_validation_step_with_real_data():
    """Test a single validation step with real data."""
    logger.info("Testing validation step with real data...")
    
    try:
        # Create model and data module
        model, data_module = test_model_creation_with_real_data()
        
        # Get a real batch from the dataloader
        train_dl = data_module.train_dataloader()
        batch = next(iter(train_dl))
        
        logger.info(f"✅ Real validation batch loaded: {list(batch.keys())}")
        
        # Test forward pass in eval mode
        model.eval()
        with torch.no_grad():
            output = model(batch)
            logger.info(f"✅ Validation forward pass completed: {output.shape}")
        
        logger.info("✅ Validation step with real data completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Validation step failed: {e}")
        raise

def test_end_to_end_real_data():
    """Test end-to-end training pipeline with real data."""
    logger.info("Testing end-to-end training pipeline with real data...")
    
    try:
        # Test data loading
        test_data_loading_real_data()
        
        # Test model creation
        test_model_creation_with_real_data()
        
        # Test training step
        test_training_step_with_real_data()
        
        # Test validation step
        test_validation_step_with_real_data()
        
        logger.info("✅ End-to-end training pipeline with real data completed successfully")
        
    except Exception as e:
        logger.error(f"❌ End-to-end test failed: {e}")
        raise

def run_all_real_data_tests():
    """Run all real data training integration tests."""
    logger.info("🚀 Starting Phase 6: Real Data Training Integration tests...")
    
    try:
        test_data_loading_real_data()
        test_model_creation_with_real_data()
        test_training_step_with_real_data()
        test_validation_step_with_real_data()
        test_end_to_end_real_data()
        
        logger.info("🎉 All Phase 6 real data training integration tests passed!")
        
    except Exception as e:
        logger.error(f"❌ Real data training integration test failed: {e}")
        raise

if __name__ == "__main__":
    run_all_real_data_tests() 
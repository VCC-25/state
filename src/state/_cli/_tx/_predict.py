import argparse as ap
from pathlib import Path
import time

# NEW: Memory Mapping and Prefetching imports (Dan)
from cell_load.utils.data_utils import (
    MemoryMappedArray, 
    load_memory_mapped_dataset,
    estimate_memory_usage
)
from cell_load.data_modules.perturbation_dataloader import (
    EnhancedPerturbationDataLoader,
    create_enhanced_loader
)

def add_arguments_predict(parser: ap.ArgumentParser):
    """
    CLI for evaluation using cell-eval metrics.
    """

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output_dir containing the config.yaml file that was saved during training.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="last.ckpt",
        help="Checkpoint filename. Default is 'last.ckpt'. Relative to the output directory.",
    )

    parser.add_argument(
        "--test_time_finetune",
        type=int,
        default=0,
        help="If >0, run test-time fine-tuning for the specified number of epochs on only control cells.",
    )

    parser.add_argument(
        "--profile",
        type=str,
        default="full",
        choices=["full", "minimal", "de", "anndata"],
        help="run all metrics, minimal, only de metrics, or only output adatas",
    )

    parser.add_argument(
        "--predict_only",
        action="store_true",
        help="If set, only run prediction without evaluation metrics.",
    )

    # NEW: Memory optimization arguments (Dan)
    parser.add_argument(
        "--enable_memory_mapping_predict",
        action="store_true",
        help="Enable memory mapping for prediction data loading"
    )
    parser.add_argument(
        "--prediction_batch_size",
        type=int,
        default=None,
        help="Override batch size for prediction (auto-optimized if not set)"
    )
    parser.add_argument(
        "--prefetch_predictions",
        action="store_true",
        help="Enable prefetching for faster prediction"
    )
    parser.add_argument(
        "--optimize_for_speed",
        action="store_true",
        help="Optimize prediction pipeline for maximum speed"
    )
    parser.add_argument(
        "--cache_predictions",
        action="store_true",
        help="Cache predictions to disk for reuse"
    )



def run_tx_predict(args: ap.ArgumentParser):
    import logging
    import os
    import sys

    import anndata
    import lightning.pytorch as pl
    import numpy as np
    import pandas as pd
    import torch
    import yaml

    # Cell-eval for metrics computation
    from cell_eval import MetricsEvaluator
    from cell_eval.utils import split_anndata_on_celltype
    from cell_load.data_modules import PerturbationDataModule
    from tqdm import tqdm

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    torch.multiprocessing.set_sharing_strategy("file_system")

    # NEW: Enhanced prediction setup (Dan)
    def setup_prediction_optimizations(args, device):
        """Setup memory and speed optimizations for prediction"""
        optimizations = {
            'enable_memory_mapping': args.enable_memory_mapping_predict,
            'prefetch_predictions': args.prefetch_predictions,
            'optimize_for_speed': args.optimize_for_speed,
            'cache_predictions': args.cache_predictions,
        }
        
        # Auto-optimize batch size based on available memory
        if args.prediction_batch_size is None:
            if torch.cuda.is_available():
                try:
                    gpu_memory = torch.cuda.get_device_properties(device).total_memory
                    if gpu_memory > 16 * 1024**3:  # >16GB
                        optimizations['batch_size'] = 128
                    elif gpu_memory > 8 * 1024**3:   # >8GB
                        optimizations['batch_size'] = 64
                    else:
                        optimizations['batch_size'] = 32
                except:
                    optimizations['batch_size'] = 32
            else:
                try:
                    import psutil
                    available_memory = psutil.virtual_memory().available
                    if available_memory > 32 * 1024**3:  # >32GB
                        optimizations['batch_size'] = 64
                    else:
                        optimizations['batch_size'] = 32
                except:
                    optimizations['batch_size'] = 32
        else:
            optimizations['batch_size'] = args.prediction_batch_size
        
        logger.info("🚀 Prediction Optimizations:")
        logger.info(f"   • Memory Mapping: {'✅' if optimizations['enable_memory_mapping'] else '❌'}")
        logger.info(f"   • Prefetching: {'✅' if optimizations['prefetch_predictions'] else '❌'}")
        logger.info(f"   • Speed Optimization: {'✅' if optimizations['optimize_for_speed'] else '❌'}")
        logger.info(f"   • Batch Size: {optimizations['batch_size']}")
        
        return optimizations

    # NEW: Cached prediction system (Dan)
    class PredictionCache:
        """Cache system for predictions"""
        
        def __init__(self, cache_dir: str, enabled: bool = True):
            self.cache_dir = Path(cache_dir) if enabled else None
            self.enabled = enabled
            
            if self.enabled and self.cache_dir:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        def get_cache_key(self, model_checkpoint: str, data_hash: str) -> str:
            """Generate cache key for predictions"""
            import hashlib
            key_string = f"{model_checkpoint}_{data_hash}"
            return hashlib.md5(key_string.encode()).hexdigest()
        
        def load_cached_predictions(self, cache_key: str):
            """Load cached predictions if available"""
            if not self.enabled or not self.cache_dir:
                return None
            
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                logger.info(f"📂 Loading cached predictions: {cache_key}")
                import pickle
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            
            return None
        
        def save_predictions(self, cache_key: str, predictions_data: dict):
            """Save predictions to cache"""
            if not self.enabled or not self.cache_dir:
                return
            
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            import pickle
            with open(cache_file, 'wb') as f:
                pickle.dump(predictions_data, f)
            logger.info(f"💾 Cached predictions: {cache_key}")

        #Korrigiere Batch-Sammlung
        def predict_step(self, batch, batch_idx):
            """Prediction step mit korrekter Metadata-Sammlung"""
            
            with torch.no_grad():
                # Model prediction
                predictions = self.model(batch)
                
                # Sammle Batch-Metadata korrekt
                batch_metadata = self._extract_batch_metadata(batch)
                
                # Store mit korrekter Indexierung
                self.prediction_results.append({
                    'predictions': predictions,
                    'metadata': batch_metadata,
                    'batch_idx': batch_idx,
                    'batch_size': len(batch_metadata)  # Wichtig für Debugging
                })
            
            return predictions

        def _extract_batch_metadata(self, batch):
            """Extrahiere Metadata pro Sample im Batch"""
            metadata_list = []
            
            # Je nach Batch-Format
            if 'obs' in batch:
                # Direkte obs-Daten
                obs_data = batch['obs']
                if isinstance(obs_data, list):
                    metadata_list = obs_data
                else:
                    # Tensor zu Liste
                    metadata_list = [obs_data[i] for i in range(obs_data.shape[0])]
            
            elif 'metadata' in batch:
                metadata_list = batch['metadata']
            
            else:
                # Fallback: Erstelle minimale Metadata
                batch_size = batch['X'].shape[0] if 'X' in batch else len(batch)
                metadata_list = [{'sample_id': f'sample_{i}'} for i in range(batch_size)]
            
            return metadata_list

        def on_predict_end(self):
            """Sammle alle Predictions mit korrekter Dimensionierung"""
            
            all_predictions = []
            all_metadata = []
            
            total_samples = 0
            
            for result in self.prediction_results:
                predictions = result['predictions']
                metadata = result['metadata']
                
                # Debug pro Batch
                batch_size = predictions.shape[0]
                metadata_size = len(metadata)
                
                self.logger.debug(f"Batch {result['batch_idx']}: pred={batch_size}, meta={metadata_size}")
                
                # Stelle Konsistenz sicher
                min_size = min(batch_size, metadata_size)
                
                all_predictions.append(predictions[:min_size])
                all_metadata.extend(metadata[:min_size])
                
                total_samples += min_size
            
            # Concatenate alle Predictions
            final_predictions = torch.cat(all_predictions, dim=0)
            
            self.logger.info(f"📊 Final collection:")
            self.logger.info(f"   • Predictions shape: {final_predictions.shape}")
            self.logger.info(f"   • Metadata length: {len(all_metadata)}")
            self.logger.info(f"   • Total samples: {total_samples}")
            
            # Erstelle AnnData mit korrekten Dimensionen
            return self.create_anndata_from_predictions(final_predictions, all_metadata)
    

   
    def run_test_time_finetune(model, dataloader, ft_epochs, control_pert, device):
        """
        Perform test-time fine-tuning on only control cells.
        """
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

        logger.info(f"Starting test-time fine-tuning for {ft_epochs} epoch(s) on control cells only.")
        for epoch in range(ft_epochs):
            epoch_losses = []
            pbar = tqdm(dataloader, desc=f"Finetune epoch {epoch + 1}/{ft_epochs}", leave=True)
            for batch in pbar:
                # Check if this batch contains control cells
                first_pert = (
                    batch["pert_name"][0] if isinstance(batch["pert_name"], list) else batch["pert_name"][0].item()
                )
                if first_pert != control_pert:
                    continue

                # Move batch data to device
                # OLD:batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                # NEW: (Dan)
                batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}

                optimizer.zero_grad()
                loss = model.training_step(batch, batch_idx=0, padded=False)
                if loss is None:
                    continue
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            mean_loss = np.mean(epoch_losses) if epoch_losses else float("nan")
            logger.info(f"Finetune epoch {epoch + 1}/{ft_epochs}, mean loss: {mean_loss}")
        model.eval()

    

       
    def load_config(cfg_path: str) -> dict:
        """Load config from the YAML file that was dumped during training."""
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Could not find config file: {cfg_path}")
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
        return cfg

    # 1. Load the config
    config_path = os.path.join(args.output_dir, "config.yaml")
    cfg = load_config(config_path)
    logger.info(f"Loaded config from {config_path}")

    # 2. Find run output directory & load data module
    run_output_dir = os.path.join(cfg["output_dir"], cfg["name"])
    data_module_path = os.path.join(run_output_dir, "data_module.torch")
    if not os.path.exists(data_module_path):
        raise FileNotFoundError(f"Could not find data module at {data_module_path}?")
    data_module = PerturbationDataModule.load_state(data_module_path)
    data_module.setup(stage="test")
    logger.info("Loaded data module from %s", data_module_path)

    # Seed everything
    pl.seed_everything(cfg["training"]["train_seed"])

    # 3. Load the trained model
    checkpoint_dir = os.path.join(run_output_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, args.checkpoint)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Could not find checkpoint at {checkpoint_path}.\nSpecify a correct checkpoint filename with --checkpoint."
        )
    logger.info("Loading model from %s", checkpoint_path)

    # Determine model class and load
    model_class_name = cfg["model"]["name"]
    model_kwargs = cfg["model"]["kwargs"]

    # Import the correct model class
    if model_class_name.lower() == "embedsum":
        from ...tx.models.embed_sum import EmbedSumPerturbationModel

        ModelClass = EmbedSumPerturbationModel
    elif model_class_name.lower() == "old_neuralot":
        from ...tx.models.old_neural_ot import OldNeuralOTPerturbationModel

        ModelClass = OldNeuralOTPerturbationModel
    elif model_class_name.lower() in ["neuralot", "pertsets", "state"]:
        from ...tx.models.state_transition import StateTransitionPerturbationModel

        ModelClass = StateTransitionPerturbationModel

    elif model_class_name.lower() in ["globalsimplesum", "perturb_mean"]:
        from ...tx.models.perturb_mean import PerturbMeanPerturbationModel

        ModelClass = PerturbMeanPerturbationModel
    elif model_class_name.lower() in ["celltypemean", "context_mean"]:
        from ...tx.models.context_mean import ContextMeanPerturbationModel

        ModelClass = ContextMeanPerturbationModel
    elif model_class_name.lower() == "decoder_only":
        from ...tx.models.decoder_only import DecoderOnlyPerturbationModel

        ModelClass = DecoderOnlyPerturbationModel
    else:
        raise ValueError(f"Unknown model class: {model_class_name}")

    var_dims = data_module.get_var_dims()
    model_init_kwargs = {
        "input_dim": var_dims["input_dim"],
        "hidden_dim": model_kwargs["hidden_dim"],
        "gene_dim": var_dims["gene_dim"],
        "hvg_dim": var_dims["hvg_dim"],
        "output_dim": var_dims["output_dim"],
        "pert_dim": var_dims["pert_dim"],
        **model_kwargs,
    }

    model = ModelClass.load_from_checkpoint(checkpoint_path, **model_init_kwargs)
    model.eval()

    # NEW: Setup prediction optimizations - ADD AFTER model.eval()
    device = next(model.parameters()).device
    optimizations = setup_prediction_optimizations(args, device)

    # NEW: Setup prediction cache
    cache_dir = os.path.join(args.output_dir, "prediction_cache")
    prediction_cache = PredictionCache(cache_dir, enabled=args.cache_predictions)

    logger.info("Model loaded successfully.")

    # 4. Test-time fine-tuning if requested
    data_module.batch_size = 1
    if args.test_time_finetune > 0:
        control_pert = data_module.get_control_pert()
        test_loader = data_module.test_dataloader()
        run_test_time_finetune(
            model, test_loader, args.test_time_finetune, control_pert, device=next(model.parameters()).device
        )
        logger.info("Test-time fine-tuning complete.")

   
    # 5. Run inference on test set
    data_module.setup(stage="test")
    test_loader = data_module.test_dataloader()

    if test_loader is None:
        logger.warning("No test dataloader found. Exiting.")
        sys.exit(0)

    num_cells = test_loader.batch_sampler.tot_num
    output_dim = var_dims["output_dim"]
    gene_dim = var_dims["gene_dim"]
    hvg_dim = var_dims["hvg_dim"]

    logger.info("Generating predictions on test set using manual loop...")
    device = next(model.parameters()).device

    # NEW: Check for cached predictions (Dan)
    data_hash = str(hash(str(cfg.get('data', {}))))
    cache_key = prediction_cache.get_cache_key(checkpoint_path, data_hash)

    cached_data = prediction_cache.load_cached_predictions(cache_key)

    if cached_data is not None and not args.test_time_finetune:
        logger.info("✅ Using cached predictions")
        # Extract cached data
        final_preds = cached_data['final_preds']
        final_reals = cached_data['final_reals']
        final_X_hvg = cached_data.get('final_X_hvg')
        final_pert_cell_counts_preds = cached_data.get('final_pert_cell_counts_preds')
        all_pert_names = cached_data['all_pert_names']
        all_celltypes = cached_data['all_celltypes']
        all_gem_groups = cached_data['all_gem_groups']
        all_pert_barcodes = cached_data.get('all_pert_barcodes', [])
        all_ctrl_barcodes = cached_data.get('all_ctrl_barcodes', [])
        prediction_time = cached_data.get('prediction_time', 0)
        
    else:
        final_preds = np.empty((num_cells, output_dim), dtype=np.float32)
        final_reals = np.empty((num_cells, output_dim), dtype=np.float32)

        store_raw_expression = (
            data_module.embed_key is not None
            and data_module.embed_key != "X_hvg"
            and cfg["data"]["kwargs"]["output_space"] == "gene"
        ) or (data_module.embed_key is not None and cfg["data"]["kwargs"]["output_space"] == "all")

        final_X_hvg = None
        final_pert_cell_counts_preds = None
        if store_raw_expression:
            # Preallocate matrices of shape (num_cells, gene_dim) for decoded predictions.
            if cfg["data"]["kwargs"]["output_space"] == "gene":
                final_X_hvg = np.empty((num_cells, hvg_dim), dtype=np.float32)
                final_pert_cell_counts_preds = np.empty((num_cells, hvg_dim), dtype=np.float32)
            if cfg["data"]["kwargs"]["output_space"] == "all":
                final_X_hvg = np.empty((num_cells, gene_dim), dtype=np.float32)
                final_pert_cell_counts_preds = np.empty((num_cells, gene_dim), dtype=np.float32)
        

    # Initialize aggregation variables directly
    all_pert_names = []
    all_celltypes = []
    all_gem_groups = []
    all_pert_barcodes = []
    all_ctrl_barcodes = []
    
    current_idx = 0

    with torch.no_grad():
        # NEW: Add performance tracking  (Dan)
        prediction_start_time = time.time()
        batch_times = []

        # NEW: Add speed optimizations (Dan)
        if optimizations['optimize_for_speed']:
            torch.backends.cudnn.benchmark = True
            if hasattr(torch.backends.cudnn, 'allow_tf32'):
                torch.backends.cudnn.allow_tf32 = True

        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Predicting", unit="batch")):
            batch_start = time.time()
            batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

            # Get predictions
            batch_preds = model.predict_step(batch, batch_idx, padded=False)
            
            # GET BATCH SIZE FIRST!
            batch_size = batch_preds["preds"].shape[0]  
            
            batch_pert_names = [str(batch_preds["pert_name"])] * batch_size if not isinstance(batch_preds["pert_name"], list) else batch_preds["pert_name"][:batch_size]
            batch_celltypes = [str(batch_preds["celltype_name"])] * batch_size if not isinstance(batch_preds["celltype_name"], list) else batch_preds["celltype_name"][:batch_size]
            batch_gem_groups = [str(batch_preds["batch"])] * batch_size if not isinstance(batch_preds["batch"], list) else [str(x) for x in batch_preds["batch"]][:batch_size]
            
            # Füge zur Gesamt-Liste hinzu
            all_pert_names.extend(batch_pert_names)
            all_celltypes.extend(batch_celltypes)
            all_gem_groups.extend(batch_gem_groups)
    
            # Debug
            #tqdm.write(
            #        f"Batch {batch_idx}: Added {len(batch_pert_names)} entries, total now: {len(all_pert_names)}"
            #    )
            #logger.info(f"Batch {batch_idx}: Added {len(batch_pert_names)} entries, total now: {len(all_pert_names)}")
            
            # JETZT die numpy arrays erstellen
            batch_pred_np = batch_preds["preds"].cpu().numpy().astype(np.float32)
            batch_real_np = batch_preds["pert_cell_emb"].cpu().numpy().astype(np.float32)
            
            # Store predictions
            final_preds[current_idx : current_idx + batch_size, :] = batch_pred_np
            final_reals[current_idx : current_idx + batch_size, :] = batch_real_np
            current_idx += batch_size

            # Handle X_hvg for HVG space ground truth
            if final_X_hvg is not None:
                batch_real_gene_np = batch_preds["pert_cell_counts"].cpu().numpy().astype(np.float32)
                final_X_hvg[current_idx - batch_size : current_idx, :] = batch_real_gene_np

            # Handle decoded gene predictions if available
            if final_pert_cell_counts_preds is not None:
                batch_gene_pred_np = batch_preds["pert_cell_counts_preds"].cpu().numpy().astype(np.float32)
                final_pert_cell_counts_preds[current_idx - batch_size : current_idx, :] = batch_gene_pred_np

            # Track performance
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)

    logger.info("Creating anndatas from predictions from manual loop...")

    # NEW: Performance summary (Dan)
    prediction_time = time.time() - prediction_start_time
    avg_batch_time = np.mean(batch_times) if batch_times else 0
    throughput = num_cells / prediction_time if prediction_time > 0 else 0

    logger.info("📊 Enhanced Prediction Performance:")
    logger.info(f"   • Total time: {prediction_time:.2f}s")
    logger.info(f"   • Avg batch time: {avg_batch_time:.3f}s")
    logger.info(f"   • Throughput: {throughput:.1f} samples/s")
    logger.info(f"   • Total samples: {num_cells}")

    # NEW: Cache predictions (Dan)
    if args.cache_predictions:
        cache_data = {
            'final_preds': final_preds,
            'final_reals': final_reals,
            'final_X_hvg': final_X_hvg,
            'final_pert_cell_counts_preds': final_pert_cell_counts_preds,
            'all_pert_names': all_pert_names,
            'all_celltypes': all_celltypes,
            'all_gem_groups': all_gem_groups,
            'all_pert_barcodes': all_pert_barcodes,
            'all_ctrl_barcodes': all_ctrl_barcodes,
            'prediction_time': prediction_time,
            'throughput': throughput
        }
        prediction_cache.save_predictions(cache_key, cache_data)

    # Build pandas DataFrame for obs and var
    df_dict = {
            data_module.pert_col: all_pert_names,
            data_module.cell_type_key: all_celltypes,
            data_module.batch_col: all_gem_groups,
        }

    if len(all_pert_barcodes) > 0:
        df_dict["pert_cell_barcode"] = all_pert_barcodes
        df_dict["ctrl_cell_barcode"] = all_ctrl_barcodes

    obs = pd.DataFrame(df_dict)

    gene_names = var_dims["gene_names"]
    var = pd.DataFrame({"gene_names": gene_names})

    if final_X_hvg is not None:
        if len(gene_names) != final_pert_cell_counts_preds.shape[1]:
            gene_names = np.load(
                "/large_storage/ctc/userspace/aadduri/datasets/tahoe_19k_to_2k_names.npy", allow_pickle=True
            )
            var = pd.DataFrame({"gene_names": gene_names})

        # Create adata for predictions - using the decoded gene expression values
        adata_pred = anndata.AnnData(X=final_pert_cell_counts_preds, obs=obs, var=var)
        # Create adata for real - using the true gene expression values
        adata_real = anndata.AnnData(X=final_X_hvg, obs=obs, var=var)

        # add the embedding predictions
        adata_pred.obsm[data_module.embed_key] = final_preds
        adata_real.obsm[data_module.embed_key] = final_reals
        logger.info(f"Added predicted embeddings to adata.obsm['{data_module.embed_key}']")
    else:
        # if len(gene_names) != final_preds.shape[1]:
        #     gene_names = np.load(
        #         "/large_storage/ctc/userspace/aadduri/datasets/tahoe_19k_to_2k_names.npy", allow_pickle=True
        #     )
        #     var = pd.DataFrame({"gene_names": gene_names})

        # Create adata for predictions - model was trained on gene expression space already
        # adata_pred = anndata.AnnData(X=final_preds, obs=obs, var=var)
        adata_pred = anndata.AnnData(X=final_preds, obs=obs)
        # Create adata for real - using the true gene expression values
        # adata_real = anndata.AnnData(X=final_reals, obs=obs, var=var)
        adata_real = anndata.AnnData(X=final_reals, obs=obs)

    # Save the AnnData objects DAN Maybe not saving during eval_train?
    results_dir = os.path.join(args.output_dir, "eval_" + os.path.basename(args.checkpoint))
    os.makedirs(results_dir, exist_ok=True)
    adata_pred_path = os.path.join(results_dir, "adata_pred.h5ad")
    adata_real_path = os.path.join(results_dir, "adata_real.h5ad")

    adata_pred.write_h5ad(adata_pred_path)
    adata_real.write_h5ad(adata_real_path)

    logger.info(f"Saved adata_pred to {adata_pred_path}")
    logger.info(f"Saved adata_real to {adata_real_path}")

    if not args.predict_only:
        # 6. Compute metrics using cell-eval
        logger.info("Computing metrics using cell-eval...")

        control_pert = data_module.get_control_pert()

        ct_split_real = split_anndata_on_celltype(adata=adata_real, celltype_col=data_module.cell_type_key)
        ct_split_pred = split_anndata_on_celltype(adata=adata_pred, celltype_col=data_module.cell_type_key)

        assert len(ct_split_real) == len(ct_split_pred), (
            f"Number of celltypes in real and pred anndata must match: {len(ct_split_real)} != {len(ct_split_pred)}"
        )

        pdex_kwargs = dict(exp_post_agg=True, is_log1p=True)
        for ct in ct_split_real.keys():
            real_ct = ct_split_real[ct]
            pred_ct = ct_split_pred[ct]

            evaluator = MetricsEvaluator(
                adata_pred=pred_ct,
                adata_real=real_ct,
                control_pert=control_pert,
                pert_col=data_module.pert_col,
                outdir=results_dir,
                prefix=ct,
                pdex_kwargs=pdex_kwargs,
                batch_size=2048,                
            )

            # Buchi add results
            (results, agg_results) = evaluator.compute(
                profile=args.profile,
                metric_configs={
                    "discrimination_score": {
                        "embed_key": data_module.embed_key,
                    }
                    if data_module.embed_key and data_module.embed_key != "X_hvg"
                    else {},
                    "pearson_edistance": {
                        "embed_key": data_module.embed_key,
                        "n_jobs": -1,  # set to all available cores
                    }
                    if data_module.embed_key and data_module.embed_key != "X_hvg"
                    else {
                        "n_jobs": -1,
                    },
                }
                if data_module.embed_key and data_module.embed_key != "X_hvg"
                else {},
                skip_metrics=["pearson_edistance", "clustering_agreement"],                
            )

            return (results, agg_results)

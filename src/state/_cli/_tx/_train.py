import argparse as ap
import os

from omegaconf import DictConfig, OmegaConf
from ...tx.callbacks import BatchSpeedMonitorCallback, ScheduledFinetuningCallback
from ...tx.callbacks.cell_eval_callback import CellEvalCallback
#from ._predict import run_tx_predict

def add_arguments_train(parser: ap.ArgumentParser):
    # Allow remaining args to be passed through to Hydra
    parser.add_argument("hydra_overrides", nargs="*", help="Hydra configuration overrides (e.g., data.batch_size=32)")
    # Add custom help handler 
    parser.add_argument("--help", "-h", action="store_true", help="Show configuration help with all parameters")
    
    # DAN:Hardware-spezifische Argumente - DAN
    parser.add_argument("--accelerator", type=str, 
                       choices=["auto", "gpu", "cpu", "mps"], 
                       default="auto",
                       help="Force specific accelerator (auto, gpu, cpu, mps)")
    parser.add_argument("--backend", type=str,
                       choices=["auto", "cuda", "rocm"],
                       default="auto", 
                       help="Force specific GPU backend (auto, cuda, rocm)")
    parser.add_argument("--mixed_precision", type=str,
                       choices=["16", "bf16", "32"],
                       default="32",
                       help="Mixed precision training")

    # DAN: Cell-eval args
    parser.add_argument("--eval_during_training", action="store_true", 
                       help="Enables cell evaluation during training")
    parser.add_argument("--eval_every_n_steps", type=int, default=500,
                       help="Cell-eval Evaluation every N steps (default: 500)")
    parser.add_argument("--pred_data_path", type=str,
                       help="Path to evaluation test data (.h5ad)")
    parser.add_argument("--real_data_path", type=str,
                       help="Path to the control template for evaluation (.h5ad)")
    parser.add_argument("--eval_metrics", nargs='+', 
                       default=['mse', 'pearson', 'spearman'],
                       help="Evaluation metrics for cell-eval")

def run_tx_train(cfg: DictConfig):
    import json
    import logging
    import os
    import pickle
    import shutil
    from os.path import exists, join
    from pathlib import Path

    import lightning.pytorch as pl
    import torch
    from cell_load.data_modules import PerturbationDataModule
    from cell_load.utils.modules import get_datamodule
    from lightning.pytorch.loggers import WandbLogger
    from lightning.pytorch.plugins.precision import MixedPrecision

    from ...tx.callbacks import BatchSpeedMonitorCallback
    from ...tx.utils import get_checkpoint_callbacks, get_lightning_module, get_loggers

    # DAN: Hardware-Setup-Funktionen
    def detect_accelerator():
        """Detect available accelerator with ROCm support"""
        if torch.cuda.is_available():
            # Check if it's ROCm or CUDA
            if torch.version.hip is not None:
                print(f"ROCm detected: {torch.version.hip}")
                return "gpu", "rocm"
            else:
                print(f"CUDA detected: {torch.version.cuda}")
                return "gpu", "cuda"
        elif torch.backends.mps.is_available():
            print("MPS (Apple Metal) detected")
            return "mps", "mps"
        else:
            print("Using CPU")
            return "cpu", "cpu"

    def setup_hardware_and_precision(cfg):
        """Setup hardware accelerator and precision plugins"""
        
        # Get user preferences from config
        force_accelerator = cfg.get("accelerator", "auto")
        force_backend = cfg.get("backend", "auto") 
        mixed_precision = cfg.get("mixed_precision", "32")
        
        # Detect or use forced accelerator
        if force_accelerator == "auto":
            accelerator, backend = detect_accelerator()
        else:
            accelerator = force_accelerator
            if accelerator == "gpu":
                if force_backend == "auto":
                    _, backend = detect_accelerator()
                else:
                    backend = force_backend
            else:
                backend = accelerator
        
        print(f"Using accelerator: {accelerator}, backend: {backend}")
        
        # Setup precision plugins
        plugins = []
        
        if accelerator == "gpu":
            device_str = "cuda"  # Both CUDA and ROCm use "cuda" device string
            
            if backend == "rocm":
                print("Configuring for ROCm...")
                # ROCm-specific optimizations
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                
            elif backend == "cuda":
                print("Configuring for CUDA...")
                
            # Mixed precision setup (same for both CUDA and ROCm)
            if mixed_precision == "bf16":
                plugins.append(MixedPrecision(precision="bf16-mixed", device=device_str))
            elif mixed_precision == "16":
                plugins.append(MixedPrecision(precision="16-mixed", device=device_str))
                
        return accelerator, backend, plugins
    
    logger = logging.getLogger(__name__)
    torch.set_float32_matmul_precision("medium")

    cfg_yaml = OmegaConf.to_yaml(cfg, resolve=True)
    cfg = OmegaConf.to_container(cfg, resolve=True)

    # Setup output directory
    run_output_dir = join(cfg["output_dir"], cfg["name"])
    if os.path.exists(run_output_dir) and cfg["overwrite"]:
        print(f"Output dir {run_output_dir} already exists, overwriting")
        shutil.rmtree(run_output_dir)
    os.makedirs(run_output_dir, exist_ok=True)

    # Set up wandb directory if needed
    if cfg["use_wandb"]:
        os.makedirs(cfg["wandb"]["local_wandb_dir"], exist_ok=True)

    with open(join(run_output_dir, "config.yaml"), "w") as f:
        f.write(cfg_yaml)

    # Set random seeds
    pl.seed_everything(cfg["training"]["train_seed"])

    # if the provided pert_col is drugname_drugconc, hard code the value of control pert
    # this is because it's surprisingly hard to specify a list of tuples in the config as a string
    if cfg["data"]["kwargs"]["pert_col"] == "drugname_drugconc":
        cfg["data"]["kwargs"]["control_pert"] = "[('DMSO_TF', 0.0, 'uM')]"

    # Initialize data module. this is backwards compatible with previous configs
    try:
        sentence_len = cfg["model"]["cell_set_len"]
    except KeyError:
        if cfg["model"]["name"].lower() in ["cpa", "scvi"] or cfg["model"]["name"].lower().startswith("scgpt"):
            if "cell_sentence_len" in cfg["model"]["kwargs"] and cfg["model"]["kwargs"]["cell_sentence_len"] > 1:
                sentence_len = cfg["model"]["kwargs"]["cell_sentence_len"]
                cfg["training"]["batch_size"] = 1
            else:
                sentence_len = 1
        else:
            try:
                sentence_len = cfg["model"]["kwargs"]["transformer_backbone_kwargs"]["n_positions"]
            except:
                sentence_len = cfg["model"]["kwargs"]["transformer_backbone_kwargs"]["max_position_embeddings"]

    if cfg["model"]["name"].lower().startswith("scgpt"):  # scGPT uses log-normalized expression
        cfg["data"]["kwargs"]["transform"] = "log-normalize"
        cfg["data"]["kwargs"]["hvg_names_uns_key"] = (
            "hvg_names" if cfg["data"]["kwargs"]["train_task"] != "replogle" else None
        )  # TODO: better to not hardcode this

        cfg["data"]["kwargs"]["dataset_cls"] = "scGPTPerturbationDataset"

        model_dir = Path(cfg["model"]["kwargs"]["pretrained_path"])

        vocab_file = model_dir / "vocab.json"

        vocab = json.load(open(vocab_file, "r"))
        cfg["model"]["kwargs"]["pad_token_id"] = vocab["<pad>"]
        for s in cfg["model"]["kwargs"]["special_tokens"]:
            if s not in vocab:
                vocab[s] = len(vocab)

        cfg["data"]["kwargs"]["vocab"] = vocab
        cfg["data"]["kwargs"]["perturbation_type"] = cfg["model"]["kwargs"]["perturbation_type"]
        cfg["model"]["kwargs"]["ntoken"] = len(vocab)
        cfg["model"]["kwargs"]["d_model"] = cfg["model"]["kwargs"]["embsize"]

        logger.info("Added vocab and hvg_names_uns_key to data kwargs for scGPT")

    elif cfg["model"]["name"].lower() == "cpa" and cfg["model"]["kwargs"]["recon_loss"] == "gauss":
        cfg["data"]["kwargs"]["transform"] = "log-normalize"
    elif cfg["model"]["name"].lower() == "scvi":
        cfg["data"]["kwargs"]["transform"] = None

    data_module: PerturbationDataModule = get_datamodule(
        cfg["data"]["name"],
        cfg["data"]["kwargs"],
        batch_size=cfg["training"]["batch_size"],
        cell_sentence_len=sentence_len,
        exclude_datasets=["competition_val_template"],
    )

    with open(join(run_output_dir, "data_module.torch"), "wb") as f:
        # TODO-Abhi: only save necessary data
        data_module.save_state(f)
    data_module.setup(stage="fit")
    dl = data_module.train_dataloader()
    print("num_workers:", dl.num_workers)
    print("batch size:", dl.batch_size)

    var_dims = data_module.get_var_dims()  # {"gene_dim": …, "hvg_dim": …}
    if cfg["data"]["kwargs"]["output_space"] == "gene":
        gene_dim = var_dims.get("hvg_dim", 2000)  # fallback if key missing
    else:
        gene_dim = var_dims.get("gene_dim", 2000)  # fallback if key missing
    latent_dim = var_dims["output_dim"]  # same as model.output_dim
    hidden_dims = cfg["model"]["kwargs"].get("decoder_hidden_dims", [1024, 1024, 512])

    decoder_cfg = dict(
        latent_dim=latent_dim,
        gene_dim=gene_dim,
        hidden_dims=hidden_dims,
        dropout=cfg["model"]["kwargs"].get("decoder_dropout", 0.1),
        residual_decoder=cfg["model"]["kwargs"].get("residual_decoder", False),
    )

    # tuck it into the kwargs that will reach the LightningModule
    cfg["model"]["kwargs"]["decoder_cfg"] = decoder_cfg

    # Save the onehot maps as pickle files instead of storing in config
    cell_type_onehot_map_path = join(run_output_dir, "cell_type_onehot_map.pkl")
    pert_onehot_map_path = join(run_output_dir, "pert_onehot_map.pt")
    batch_onehot_map_path = join(run_output_dir, "batch_onehot_map.pkl")
    var_dims_path = join(run_output_dir, "var_dims.pkl")

    with open(cell_type_onehot_map_path, "wb") as f:
        pickle.dump(data_module.cell_type_onehot_map, f)
    torch.save(data_module.pert_onehot_map, pert_onehot_map_path)
    with open(batch_onehot_map_path, "wb") as f:
        pickle.dump(data_module.batch_onehot_map, f)
    with open(var_dims_path, "wb") as f:
        pickle.dump(var_dims, f)

    if cfg["model"]["name"].lower() in ["cpa", "scvi"] or cfg["model"]["name"].lower().startswith("scgpt"):
        cfg["model"]["kwargs"]["n_cell_types"] = len(data_module.celltype_onehot_map)
        cfg["model"]["kwargs"]["n_perts"] = len(data_module.pert_onehot_map)
        cfg["model"]["kwargs"]["n_batches"] = len(data_module.batch_onehot_map)

    # Create model
    model = get_lightning_module(
        cfg["model"]["name"],
        cfg["data"]["kwargs"],
        cfg["model"]["kwargs"],
        cfg["training"],
        data_module.get_var_dims(),
    )

    print(
        f"Model created. Estimated params size: {sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3:.2f} GB"
    )
    loggers = get_loggers(
        output_dir=cfg["output_dir"],
        name=cfg["name"],
        wandb_project=cfg["wandb"]["project"],
        wandb_entity=cfg["wandb"]["entity"],
        local_wandb_dir=cfg["wandb"]["local_wandb_dir"],
        use_wandb=cfg["use_wandb"],
        cfg=cfg,
    )

    # If using wandb, store the run path in a text file for eval
    # that matches the old train_lightning.py logic
    for lg in loggers:
        if isinstance(lg, WandbLogger):
            wandb_info_path = os.path.join(run_output_dir, "wandb_path.txt")
            with open(wandb_info_path, "w") as f:
                f.write(lg.experiment.path)
            break

    # Set up callbacks
    ckpt_callbacks = get_checkpoint_callbacks(
        cfg["output_dir"],
        cfg["name"],
        cfg["training"]["val_freq"],
        cfg["training"].get("ckpt_every_n_steps", 4000),
    )
    # Add BatchSpeedMonitorCallback to log batches per second to wandb
    batch_speed_monitor = BatchSpeedMonitorCallback()
    callbacks = ckpt_callbacks + [batch_speed_monitor]
    
    # DAN eval start
    # Create arguments for run_tx_predict
    class MockArgs:
        def __init__(self, output_dir, checkpoint, profile="minimal", predict_only=False):
            self.output_dir = output_dir
            if checkpoint is None:
                checkpoint = cfg["model"]["kwargs"].get("init_from", None)
            self.checkpoint = checkpoint # only filename
            self.test_time_finetune = 0  # No fine-tuning during training
            self.profile = profile
            self.predict_only = predict_only
    # NEU: Cell-eval Callback hinzufügen (falls konfiguriert)
    if cfg.get("cell_eval", {}).get("enabled", False):
    #if cfg["cell_eval"]["enabled"]:
        logger.info("set CellEvalCallback with baseline.")
        cell_eval_callback = CellEvalCallback(
            eval_every_n_steps=cfg["cell_eval"].get("eval_every_n_steps", 500),
            #pred_data_path=cfg["cell_eval"]["pred_data_path"],
            #real_data_path=cfg["cell_eval"]["real_data_path"],
            control_pert=cfg["data"]["kwargs"]["control_pert"],
            pert_col=cfg["data"]["kwargs"]["pert_col"],
            #eval_metrics=cfg["cell_eval"].get("eval_metrics", ['overlap_at_N', 'mae', 'discrimination_score_l1']),
            output_dir=run_output_dir,
            profile="minimal", 
            save_predictions=cfg["cell_eval"].get("save_predictions", True),
            log_to_wandb=cfg["use_wandb"],
            primary_metric="discrimination_score_l1",  # Hauptmetrik
            metric_weights={
                "discrimination_score_l1": 1.0,  # 2.0 Doppeltes Gewicht
                "overlap_at_N": 1.0,  # 1.5
                "mae": 1.0,
                "avg_score": 1.0
            },
            improvement_threshold=0.001,  # Mindestverbesserung
            save_best_checkpoint=True,
            baseline_comparison=True,  # from_baseline Modus
            verbose=cfg["cell_eval"].get("verbose", True),            
        )
        #callbacks.append(cell_eval_callback)
        callbacks = ckpt_callbacks + [cell_eval_callback]
        logger.info("Cell-eval Callback added")
        

    # DAN eval end
    
    # Add ScheduledFinetuningCallback if finetuning schedule is specified in the config
    finetuning_schedule = cfg["training"].get("finetuning_schedule", None)
    if finetuning_schedule and finetuning_schedule.get("enable", False):
        logger.info("Calling ScheduledFinetuningCallback.")
        finetune_steps = finetuning_schedule.get("finetune_steps", 0)
        modules_to_unfreeze = finetuning_schedule.get("modules_to_unfreeze", [])
        
        if finetune_steps > 0 and modules_to_unfreeze:
            scheduled_finetuning_callback = ScheduledFinetuningCallback(
                finetune_steps=finetune_steps,
                modules_to_unfreeze=modules_to_unfreeze,
            )
            callbacks.append(scheduled_finetuning_callback)
        else:
            logger.warning("Finetuning schedule is enabled but 'finetune_steps' or 'modules_to_unfreeze' are not set. Skipping.")

    logger.info("Loggers and callbacks set up.")

    # DAN replaced
    """if cfg["model"]["name"].lower().startswith("scgpt"):
        plugins = [
            MixedPrecision(
                precision="bf16-mixed",
                device="cuda",
            )
        ]
    else:
        plugins = []

    if torch.cuda.is_available():
        accelerator = "gpu"
    elif torch.backends.mps.is_available():
        accelerator = "mps"
    else:
        accelerator = "cpu"""
    
    # DAN: Hardware-Setup with ROCm-Support
    accelerator, backend, plugins = setup_hardware_and_precision(cfg)
    
    # Spezial scGPT treatment (if not already covered by mixed_precision)
    if cfg["model"]["name"].lower().startswith("scgpt") and not plugins:
        if accelerator == "gpu":
            device_str = "cuda"  # Both CUDA and ROCm use "cuda"
            plugins = [MixedPrecision(precision="bf16-mixed", device=device_str)]

    # Decide on trainer params
    trainer_kwargs = dict(
        accelerator=accelerator,
        devices=1,
        max_steps=cfg["training"]["max_steps"],  # for normal models
        check_val_every_n_epoch=None,
        val_check_interval=cfg["training"]["val_freq"],
        logger=loggers,
        plugins=plugins,
        callbacks=callbacks,
        gradient_clip_val=cfg["training"]["gradient_clip_val"] if cfg["model"]["name"].lower() != "cpa" else None,
    )

    # If it's SimpleSum, override to do exactly 1 epoch, ignoring `max_steps`.
    if cfg["model"]["name"].lower() == "celltypemean" or cfg["model"]["name"].lower() == "globalsimplesum":
        trainer_kwargs["max_epochs"] = 1  # do exactly one epoch
        # delete max_steps to avoid conflicts
        del trainer_kwargs["max_steps"]

    # Build trainer
    print(f"Building trainer with kwargs: {trainer_kwargs}")
    trainer = pl.Trainer(**trainer_kwargs)
    print("Trainer built successfully")

    # Load checkpoint if exists
    checkpoint_path = join(ckpt_callbacks[0].dirpath, "last.ckpt")
    if not exists(checkpoint_path):
        checkpoint_path = None
    else:
        logging.info(f"!! Resuming training from {checkpoint_path} !!")
    # print(f"DAN, why model=cpu: {next(model.parameters()).device}")
    # DAN replaced
    """if torch.mps.is_available():
        print(f"Model device: {next(model.parameters()).device}")
        print(f"METAL memory allocated: {torch.mps.memory_allocated() / 1024**3:.2f} GB")
        print(f"METAL memory reserved: {torch.mps.memory_reserved() / 1024**3:.2f} GB")

    else:    
        print(f"Model device: {next(model.parameters()).device}")
        print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"CUDA memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    """
    # DAN Hardware-Info ausgeben
    print(f"Model device: {next(model.parameters()).device}")
    
    if accelerator == "gpu":
        device_name = torch.cuda.get_device_name() if torch.cuda.is_available() else "Unknown GPU"
        print(f"GPU Device: {device_name}")
        print(f"GPU Backend: {backend.upper()}")
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU Memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        
        if backend == "rocm":
            print("ROCm-specific optimizations enabled")
        elif backend == "cuda":
            print("CUDA-specific optimizations enabled")
            
    elif accelerator == "mps":
        print(f"MPS Memory allocated: {torch.mps.memory_allocated() / 1024**3:.2f} GB")
        print(f"MPS Memory reserved: {torch.mps.memory_reserved() / 1024**3:.2f} GB")
    else:
        print("Using CPU - no GPU acceleration")

    
    logger.info("Starting trainer fit.")

    # if a checkpoint does not exist, start with the provided checkpoint
    # this is mainly used for pretrain -> finetune workflows
    manual_init = cfg["model"]["kwargs"].get("init_from", None)
    if checkpoint_path is None and manual_init is not None:
        print(f"Loading manual checkpoint from {manual_init}")
        checkpoint_path = manual_init
        # DAN replace: device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        # DAN Use the detected accelerator for device mapping
        if accelerator == "gpu":
            device = torch.device("cuda")
        elif accelerator == "mps":
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model_state = model.state_dict()
        checkpoint_state = checkpoint["state_dict"]

        # Check if output_space differs between current config and checkpoint
        checkpoint_output_space = checkpoint.get("hyper_parameters", {}).get("output_space", "gene")
        current_output_space = cfg["data"]["kwargs"]["output_space"]
        
        if checkpoint_output_space != current_output_space:
            print(f"Output space mismatch: checkpoint has '{checkpoint_output_space}', current config has '{current_output_space}'")
            print("Creating new decoder for the specified output space...")

            if not cfg["model"]["kwargs"].get("gene_decoder_bool", True):
                model._decoder_externally_configured = False
            else:
                # Override the decoder_cfg to match the new output_space
                if current_output_space == "gene":
                    new_gene_dim = var_dims.get("hvg_dim", 2000)
                else:  # output_space == "all"
                    new_gene_dim = var_dims.get("gene_dim", 2000)
                
                logger.info(f"New gene dimension for output_space='{current_output_space}': {new_gene_dim}")
                new_decoder_cfg = dict(
                    latent_dim=var_dims["output_dim"],
                    gene_dim=new_gene_dim,
                    hidden_dims=cfg["model"]["kwargs"].get("decoder_hidden_dims", [1024, 1024, 512]),
                    dropout=cfg["model"]["kwargs"].get("decoder_dropout", 0.1),
                    residual_decoder=cfg["model"]["kwargs"].get("residual_decoder", False),
                )
                
                # Update the model's decoder_cfg and rebuild decoder
                model.decoder_cfg = new_decoder_cfg
                model._build_decoder()
                model._decoder_externally_configured = True  # Mark that decoder was configured externally
                print(f"Created new decoder for output_space='{current_output_space}' with gene_dim={new_gene_dim}")

        pert_encoder_weight_key = "pert_encoder.0.weight"
        if pert_encoder_weight_key in checkpoint_state:
            checkpoint_pert_dim = checkpoint_state[pert_encoder_weight_key].shape[1]
            if checkpoint_pert_dim != model.pert_dim:
                print(
                    f"pert_encoder input dimension mismatch: model.pert_dim = {model.pert_dim} but checkpoint expects {checkpoint_pert_dim}. Overriding model's pert_dim and rebuilding pert_encoder."
                )
                # Rebuild the pert_encoder with the new pert input dimension
                from ...tx.models.utils import build_mlp

                model.pert_encoder = build_mlp(
                    in_dim=model.pert_dim,
                    out_dim=model.hidden_dim,
                    hidden_dim=model.hidden_dim,
                    n_layers=model.n_encoder_layers,
                    dropout=model.dropout,
                    activation=model.activation_class,
                )

        # Filter out mismatched size parameters
        filtered_state = {}
        for name, param in checkpoint_state.items():
            if name in model_state:
                if param.shape == model_state[name].shape:
                    filtered_state[name] = param
                else:
                    print(
                        f"Skipping parameter {name} due to shape mismatch: checkpoint={param.shape}, model={model_state[name].shape}"
                    )
            else:
                print(f"Skipping parameter {name} as it doesn't exist in the current model")

        # Load the filtered state dict
        model.load_state_dict(filtered_state, strict=False)
        print("About to call trainer.fit() with manual checkpoint...")

        # Train - for clarity we pass None
        trainer.fit(
            model,
            datamodule=data_module,
            ckpt_path=None,
        )
        print("trainer.fit() completed with manual checkpoint")
    else:
        print(f"About to call trainer.fit() with checkpoint_path={checkpoint_path}")
        # Train
        trainer.fit(
            model,
            datamodule=data_module,
            ckpt_path=checkpoint_path,
        )
        print("trainer.fit() completed")

    print("Training completed, saving final checkpoint...")

    # at this point if checkpoint_path does not exist, manually create one
    checkpoint_path = join(ckpt_callbacks[0].dirpath, "final.ckpt")
    if not exists(checkpoint_path):
        trainer.save_checkpoint(checkpoint_path)

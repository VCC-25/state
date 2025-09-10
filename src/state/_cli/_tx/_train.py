import argparse as ap
import os

import time
import threading
from queue import Queue
import psutil
import torch

from omegaconf import DictConfig, OmegaConf
from ...tx.callbacks import BatchSpeedMonitorCallback, ScheduledFinetuningCallback
from ...tx.callbacks.cell_eval_callback import CellEvalCallback
#from ._predict import run_tx_predict

# NEW: Memory Mapping and Prefetching imports (Dan)
from cell_load.utils.data_utils import (
    MemoryMappedArray, 
    create_memory_mapped_dataset,
    estimate_memory_usage
)
from cell_load.data_modules.perturbation_dataloader import (
    EnhancedPerturbationDataLoader,
    create_enhanced_loader
)
from cell_load.mapping_strategies.batch import (
    create_memory_mapped_batch_strategy,
    AdaptiveMemoryMappedBatchStrategy
)

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

    # Cell-eval args (Dan)
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
    
    # NEW: Distributed training specific memory optimization arguments (Dan)
    parser.add_argument("--distributed_memory_mapping", action="store_true",
                       help="Enable memory mapping optimized for distributed training")
    parser.add_argument("--shared_mmap_cache", action="store_true",
                       help="Use shared memory-mapped cache across processes")
    parser.add_argument("--per_gpu_prefetch_factor", type=int, default=2,
                       help="Prefetch factor per GPU in distributed training")
    parser.add_argument("--distributed_batch_strategy", type=str,
                       choices=["sequential", "random", "stratified", "distributed_adaptive"],
                       default="distributed_adaptive",
                       help="Batch strategy optimized for distributed training")
    parser.add_argument("--sync_batch_loading", action="store_true",
                       help="Synchronize batch loading across distributed processes")

    parser.add_argument(
            "--prefetch-predictions", 
            action="store_true", 
            default=True,
            help="Enable prefetching for faster GPU utilization"
        )
        
    parser.add_argument(
        "--optimize-dataloader-memory", 
        action="store_true", 
        default=True,
        help="Optimize DataLoader for better memory usage"
    )
    
    parser.add_argument(
        "--force-single-worker", 
        action="store_true", 
        default=False,
        help="Force single-worker DataLoader to avoid shared memory issues"
    )
    
    parser.add_argument(
        "--max-dataloader-workers", 
        type=int, 
        default=4,
        help="Maximum number of DataLoader workers"
    )

class PrefetchDataLoader:
    """
    DataLoader-Wrapper für asynchrones GPU-Prefetching mit Shared Memory Optimierung
    """
    def __init__(self, loader, device, logger=None, optimize_memory=True):
        self.loader = loader
        self.device = device
        self.logger = logger
        self.stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        self.optimize_memory = optimize_memory
        
        # Shared Memory Optimierung
        if self.optimize_memory:
            self._optimize_dataloader_memory()
        
        if self.logger:
            self.logger.info(f"🚀 PrefetchDataLoader initialized for device: {device}")
            if self.stream:
                self.logger.info("✅ CUDA stream created for async prefetching")
    
    def _optimize_dataloader_memory(self):
        """Optimiert DataLoader für bessere Memory-Nutzung"""
        if hasattr(self.loader, 'num_workers') and self.loader.num_workers > 0:
            # Reduziere num_workers falls zu hoch
            available_memory = psutil.virtual_memory().available / (1024**3)  # GB
            
            if available_memory < 8:  # < 8GB RAM
                if self.loader.num_workers > 2:
                    if self.logger:
                        self.logger.warning(f"⚠️ Low memory detected ({available_memory:.1f}GB), reducing num_workers from {self.loader.num_workers} to 2")
                    self.loader.num_workers = 2
            elif available_memory < 16:  # < 16GB RAM
                if self.loader.num_workers > 4:
                    if self.logger:
                        self.logger.warning(f"⚠️ Medium memory detected ({available_memory:.1f}GB), reducing num_workers from {self.loader.num_workers} to 4")
                    self.loader.num_workers = 4
            
            # Aktiviere Pin Memory falls verfügbar
            if torch.cuda.is_available() and not getattr(self.loader, 'pin_memory', False):
                self.loader.pin_memory = True
                if self.logger:
                    self.logger.info("📌 Enabled pin_memory for faster GPU transfers")
    
    def __len__(self):
        return len(self.loader)
    
    def __iter__(self):
        batch_iter = iter(self.loader)
        preloaded_batch = None
        
        # Lade ersten Batch
        try:
            preloaded_batch = next(batch_iter)
            preloaded_batch = self._transfer_to_device(preloaded_batch)
        except StopIteration:
            return
        
        # Iteriere durch restliche Batches
        for batch in batch_iter:
            # Starte GPU-Transfer für nächsten Batch asynchron
            if self.stream and torch.cuda.is_available():
                with torch.cuda.stream(self.stream):
                    next_batch = self._transfer_to_device(batch)
            else:
                next_batch = self._transfer_to_device(batch)
            
            # Gib aktuellen Batch zurück (während nächster lädt)
            yield preloaded_batch
            
            # Warte auf Stream-Completion falls CUDA verwendet wird
            if self.stream and torch.cuda.is_available():
                torch.cuda.current_stream().wait_stream(self.stream)
            
            preloaded_batch = next_batch
        
        # Gib letzten Batch zurück
        if preloaded_batch is not None:
            yield preloaded_batch
    
    def _transfer_to_device(self, batch):
        """Transferiert Batch-Daten auf das Zielgerät"""
        if isinstance(batch, dict):
            return {k: (v.to(self.device, non_blocking=True) 
                       if torch.is_tensor(v) else v) 
                   for k, v in batch.items()}
        elif isinstance(batch, (list, tuple)):
            return type(batch)(v.to(self.device, non_blocking=True) 
                             if torch.is_tensor(v) else v 
                             for v in batch)
        elif torch.is_tensor(batch):
            return batch.to(self.device, non_blocking=True)
        else:
            return batch


class SharedMemoryOptimizedDataLoader:
    """
    DataLoader-Wrapper speziell für Shared Memory Probleme
    """
    def __init__(self, loader, device, logger=None):
        self.original_loader = loader
        self.device = device
        self.logger = logger
        
        # Erstelle optimierten DataLoader
        self.loader = self._create_optimized_loader()
        
        if self.logger:
            self.logger.info("🔧 SharedMemoryOptimizedDataLoader created")
    
    def _create_optimized_loader(self):
        """Erstellt einen memory-optimierten DataLoader"""
        from torch.utils.data import DataLoader
        
        # Kopiere alle Attribute vom Original-Loader
        dataset = self.original_loader.dataset
        batch_size = getattr(self.original_loader, 'batch_size', 1)
        shuffle = getattr(self.original_loader, 'shuffle', False)
        sampler = getattr(self.original_loader, 'sampler', None)
        batch_sampler = getattr(self.original_loader, 'batch_sampler', None)
        collate_fn = getattr(self.original_loader, 'collate_fn', None)
        drop_last = getattr(self.original_loader, 'drop_last', False)
        timeout = getattr(self.original_loader, 'timeout', 0)
        worker_init_fn = getattr(self.original_loader, 'worker_init_fn', None)
        
        # Memory-optimierte Einstellungen
        available_memory = psutil.virtual_memory().available / (1024**3)  # GB
        
        if available_memory < 8:  # < 8GB
            num_workers = 0  # Kein Multiprocessing
            pin_memory = False
            persistent_workers = False
        elif available_memory < 16:  # < 16GB
            num_workers = min(2, getattr(self.original_loader, 'num_workers', 4))
            pin_memory = torch.cuda.is_available()
            persistent_workers = False
        else:  # >= 16GB
            num_workers = min(4, getattr(self.original_loader, 'num_workers', 8))
            pin_memory = torch.cuda.is_available()
            persistent_workers = True
        
        if self.logger:
            self.logger.info(f"🔧 Optimized DataLoader settings:")
            self.logger.info(f"   • num_workers: {num_workers} (was: {getattr(self.original_loader, 'num_workers', 'unknown')})")
            self.logger.info(f"   • pin_memory: {pin_memory}")
            self.logger.info(f"   • persistent_workers: {persistent_workers}")
            self.logger.info(f"   • Available RAM: {available_memory:.1f}GB")
        
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            persistent_workers=persistent_workers,
            prefetch_factor=2 if num_workers > 0 else None,
        )
    
    def __len__(self):
        return len(self.loader)
    
    def __iter__(self):
        return iter(self.loader)


def optimize_dataloader_for_memory(loader, logger=None):
    """
    Optimiert einen DataLoader für bessere Memory-Nutzung
    """
    if logger:
        logger.info("🔧 Optimizing DataLoader for memory usage...")
    
    # Prüfe verfügbaren Speicher
    try:
        available_memory = psutil.virtual_memory().available / (1024**3)  # GB
        total_memory = psutil.virtual_memory().total / (1024**3)  # GB
        
        if logger:
            logger.info(f"💾 System Memory: {total_memory:.1f}GB total, {available_memory:.1f}GB available")
        
        # Entscheide basierend auf verfügbarem Speicher
        if available_memory < 4:  # Sehr wenig Speicher
            if logger:
                logger.warning("⚠️ Very low memory detected - using single-threaded DataLoader")
            return SharedMemoryOptimizedDataLoader(loader, None, logger)
        
        elif available_memory < 8:  # Wenig Speicher
            if logger:
                logger.warning("⚠️ Low memory detected - using memory-optimized DataLoader")
            return SharedMemoryOptimizedDataLoader(loader, None, logger)
        
        else:  # Genug Speicher für normale Optimierung
            if logger:
                logger.info("✅ Sufficient memory - using standard optimizations")
            return loader
            
    except Exception as e:
        if logger:
            logger.warning(f"⚠️ Memory detection failed: {e} - using conservative settings")
        return SharedMemoryOptimizedDataLoader(loader, None, logger)

# NEW: DataLoader Memory-Optimierung (Dan)
'''def optimize_datamodule_loaders(data_module, cfg, logger):
    """Optimiert DataModule DataLoader für bessere Memory-Nutzung"""
    
    # Prüfe ob Optimierung gewünscht ist
    optimize_memory = cfg.get("optimize_dataloader_memory", True)
    force_single_worker = cfg.get("force_single_worker", False)
    max_workers = cfg.get("max_dataloader_workers", 4)
    
    if not optimize_memory and not force_single_worker:
        logger.info("ℹ️ DataLoader optimization disabled")
        return data_module
    
    logger.info("🔧 Optimizing DataModule DataLoaders...")
    
    # Optimiere Train DataLoader
    if hasattr(data_module, 'train_dataloader'):
        original_train_method = data_module.train_dataloader
        
        def optimized_train_dataloader():
            loader = original_train_method()
            
            if force_single_worker:
                loader.num_workers = 0
                loader.pin_memory = False
                logger.info("🔧 Train DataLoader: Forced single-worker mode")
            else:
                # Memory-basierte Optimierung
                available_memory = psutil.virtual_memory().available / (1024**3)
                
                if available_memory < 8:
                    loader.num_workers = 0
                    loader.pin_memory = False
                    logger.info(f"🔧 Train DataLoader: Single-worker (low memory: {available_memory:.1f}GB)")
                else:
                    loader.num_workers = min(max_workers, loader.num_workers)
                    loader.pin_memory = torch.cuda.is_available()
                    logger.info(f"🔧 Train DataLoader: {loader.num_workers} workers, pin_memory={loader.pin_memory}")
            
            return loader
        
        data_module.train_dataloader = optimized_train_dataloader
    
    # Optimiere Val DataLoader
    if hasattr(data_module, 'val_dataloader'):
        original_val_method = data_module.val_dataloader
        
        def optimized_val_dataloader():
            loader = original_val_method()
            
            if force_single_worker:
                loader.num_workers = 0
                loader.pin_memory = False
                logger.info("🔧 Val DataLoader: Forced single-worker mode")
            else:
                # Validation braucht weniger Workers
                available_memory = psutil.virtual_memory().available / (1024**3)
                
                if available_memory < 8:
                    loader.num_workers = 0
                    loader.pin_memory = False
                else:
                    loader.num_workers = min(2, max_workers // 2)  # Weniger Workers für Validation
                    loader.pin_memory = torch.cuda.is_available()
                
                logger.info(f"🔧 Val DataLoader: {loader.num_workers} workers, pin_memory={loader.pin_memory}")
            
            return loader
        
        data_module.val_dataloader = optimized_val_dataloader
    
    logger.info("✅ DataModule DataLoaders optimized")
    return data_module
'''

def optimize_datamodule_loaders(data_module, cfg, logger):
    """Optimiert DataModule DataLoader für bessere Memory-Nutzung"""
    
    # Prüfe ob Optimierung gewünscht ist
    optimize_memory = cfg.get("optimize_dataloader_memory", True)
    force_single_worker = cfg.get("force_single_worker", False)
    max_workers = cfg.get("max_dataloader_workers", 4)
    
    if not optimize_memory and not force_single_worker:
        logger.info("ℹ️ DataLoader optimization disabled")
        return data_module
    
    logger.info("🔧 Optimizing DataModule DataLoaders...")
    
    # Bestimme optimale Worker-Anzahl
    def get_optimal_workers(is_validation=False):
        if force_single_worker:
            return 0, False, "forced single-worker"
        
        available_memory = psutil.virtual_memory().available / (1024**3)
        
        # Sehr konservative Einstellungen
        if available_memory < 4:
            return 0, False, f"very low memory ({available_memory:.1f}GB)"
        elif available_memory < 8:
            return 0, False, f"low memory ({available_memory:.1f}GB)"
        elif available_memory < 16:
            workers = 1 if is_validation else 2
            return workers, torch.cuda.is_available(), f"medium memory ({available_memory:.1f}GB)"
        else:
            workers = min(2 if is_validation else max_workers, 4)  # Max 4 workers
            return workers, torch.cuda.is_available(), f"sufficient memory ({available_memory:.1f}GB)"
    
    def create_optimized_dataloader(original_loader, is_validation=False):
        """Erstellt einen neuen optimierten DataLoader"""
        from torch.utils.data import DataLoader
        
        # Debug Original-DataLoader
        debug_dataloader_params(original_loader, logger)
        
        # Hole optimale Einstellungen
        num_workers, pin_memory, reason = get_optimal_workers(is_validation)
        
        # Extrahiere alle relevanten Attribute vom Original-DataLoader
        dataset = original_loader.dataset
        batch_size = getattr(original_loader, 'batch_size', None)
        shuffle = getattr(original_loader, 'shuffle', False)
        sampler = getattr(original_loader, 'sampler', None)
        batch_sampler = getattr(original_loader, 'batch_sampler', None)
        collate_fn = getattr(original_loader, 'collate_fn', None)
        drop_last = getattr(original_loader, 'drop_last', False)
        timeout = getattr(original_loader, 'timeout', 0)
        worker_init_fn = getattr(original_loader, 'worker_init_fn', None)
        multiprocessing_context = getattr(original_loader, 'multiprocessing_context', None)
        generator = getattr(original_loader, 'generator', None)
        
        # Bestimme DataLoader-Parameter basierend auf batch_sampler
        if batch_sampler is not None:
            # Wenn batch_sampler vorhanden ist, dürfen batch_size, shuffle, sampler, drop_last nicht gesetzt werden
            dataloader_kwargs = {
                'dataset': dataset,
                'batch_sampler': batch_sampler,
                'num_workers': num_workers,
                'collate_fn': collate_fn,
                'pin_memory': pin_memory,
                'timeout': timeout,
                'worker_init_fn': worker_init_fn,
                'persistent_workers': num_workers > 0,
            }
            
            # Optionale Parameter nur wenn nicht None
            if multiprocessing_context is not None:
                dataloader_kwargs['multiprocessing_context'] = multiprocessing_context
            if generator is not None:
                dataloader_kwargs['generator'] = generator
            if num_workers > 0:
                dataloader_kwargs['prefetch_factor'] = 2
            
            logger.info(f"🔧 {'Val' if is_validation else 'Train'} DataLoader using batch_sampler: {num_workers} workers, pin_memory={pin_memory} ({reason})")
            
        else:
            # Standard DataLoader ohne batch_sampler
            dataloader_kwargs = {
                'dataset': dataset,
                'batch_size': batch_size,
                'shuffle': shuffle,
                'sampler': sampler,
                'num_workers': num_workers,
                'collate_fn': collate_fn,
                'pin_memory': pin_memory,
                'drop_last': drop_last,
                'timeout': timeout,
                'worker_init_fn': worker_init_fn,
                'persistent_workers': num_workers > 0,
            }
            
            # Optionale Parameter nur wenn nicht None
            if multiprocessing_context is not None:
                dataloader_kwargs['multiprocessing_context'] = multiprocessing_context
            if generator is not None:
                dataloader_kwargs['generator'] = generator
            if num_workers > 0:
                dataloader_kwargs['prefetch_factor'] = 2
            
            logger.info(f"🔧 {'Val' if is_validation else 'Train'} DataLoader using standard params: {num_workers} workers, pin_memory={pin_memory} ({reason})")
        
        # Erstelle neuen DataLoader
        try:
            new_loader = DataLoader(**dataloader_kwargs)
            logger.info("✅ New DataLoader created successfully")
            return new_loader
        except Exception as e:
            logger.error(f"❌ Failed to create optimized DataLoader: {e}")
            logger.warning("⚠️ Falling back to original DataLoader")
            return original_loader
    
    # Optimiere Train DataLoader
    if hasattr(data_module, 'train_dataloader'):
        original_train_method = data_module.train_dataloader
        
        def optimized_train_dataloader():
            # Hole Original-DataLoader
            original_loader = original_train_method()
            # Erstelle optimierten DataLoader
            return create_optimized_dataloader(original_loader, is_validation=False)
        
        data_module.train_dataloader = optimized_train_dataloader
    
    # Optimiere Val DataLoader
    if hasattr(data_module, 'val_dataloader'):
        original_val_method = data_module.val_dataloader
        
        def optimized_val_dataloader():
            # Hole Original-DataLoader
            original_loader = original_val_method()
            # Erstelle optimierten DataLoader
            return create_optimized_dataloader(original_loader, is_validation=True)
        
        data_module.val_dataloader = optimized_val_dataloader
    
    logger.info("✅ DataModule DataLoaders optimized")
    return data_module

def diagnose_shared_memory_issues(logger):
    """Diagnostiziert potentielle Shared Memory Probleme"""
    logger.info("🔍 Diagnosing shared memory configuration...")
    
    force_single_worker = False
    
    try:
        import subprocess
        
        # Prüfe /dev/shm Größe
        result = subprocess.run(['df', '-h', '/dev/shm'], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                shm_info = lines[1].split()
                shm_size = shm_info[1] if len(shm_info) > 1 else "unknown"
                shm_used = shm_info[2] if len(shm_info) > 2 else "unknown"
                shm_available = shm_info[3] if len(shm_info) > 3 else "unknown"
                
                logger.info(f"📊 Shared Memory (/dev/shm):")
                logger.info(f"   • Size: {shm_size}")
                logger.info(f"   • Used: {shm_used}")
                logger.info(f"   • Available: {shm_available}")
                
                # Erweiterte Prüfung für verschiedene Größenangaben
                available_numeric = 0
                if 'G' in shm_available:
                    available_numeric = float(shm_available.replace('G', '')) * 1024  # Convert to MB
                elif 'M' in shm_available:
                    available_numeric = float(shm_available.replace('M', ''))
                elif 'K' in shm_available:
                    available_numeric = float(shm_available.replace('K', '')) / 1024  # Convert to MB
                
                # Konservative Entscheidung: Weniger als 2GB Shared Memory = Single Worker
                if available_numeric < 2048:  # < 2GB
                    logger.warning(f"⚠️ Shared memory ({shm_available}) might be insufficient for multi-worker DataLoader")
                    force_single_worker = True
        
        # Zusätzliche Prüfungen
        # 1. Prüfe ob wir in einem Container sind
        try:
            with open('/proc/1/cgroup', 'r') as f:
                cgroup_content = f.read()
                if 'docker' in cgroup_content or 'containerd' in cgroup_content:
                    logger.warning("⚠️ Running in container - shared memory might be limited")
                    force_single_worker = True
        except:
            pass
        
        # 2. Prüfe verfügbare Prozesse/Threads
        try:
            import resource
            max_processes = resource.getrlimit(resource.RLIMIT_NPROC)[0]
            if max_processes < 1024:
                logger.warning(f"⚠️ Low process limit ({max_processes}) - using single worker")
                force_single_worker = True
        except:
            pass
        
        # 3. Prüfe System Load
        try:
            load_avg = os.getloadavg()[0]  # 1-minute load average
            cpu_count = os.cpu_count()
            if load_avg > cpu_count * 2:
                logger.warning(f"⚠️ High system load ({load_avg:.1f}) - using single worker")
                force_single_worker = True
        except:
            pass
        
    except Exception as e:
        logger.warning(f"⚠️ Shared memory diagnosis failed: {e}")
        force_single_worker = True  # Be conservative on error
    
    if force_single_worker:
        logger.warning("🔧 Forcing single-worker DataLoader due to system constraints")
    else:
        logger.info("✅ Multi-worker DataLoader should be safe")
    
    return force_single_worker

def safe_fallback_dataloader(original_method, logger):
    """Sicherer Fallback auf Original-DataLoader mit minimalen Änderungen"""
    
    def fallback_dataloader():
        try:
            # Versuche Original-DataLoader zu erstellen
            original_loader = original_method()
            
            # Nur num_workers auf 0 setzen, wenn möglich
            if hasattr(original_loader, '_num_workers'):
                original_loader._num_workers = 0
            
            logger.warning("⚠️ Using original DataLoader with single worker fallback")
            return original_loader
            
        except Exception as e:
            logger.error(f"❌ Even fallback DataLoader failed: {e}")
            raise e
    
    return fallback_dataloader

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

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    os.environ['OMP_NUM_THREADS'] = '1'  # Reduziere Threading-Konflikte

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
    
    # NEW: Memory optimization setup (Dan)
    def setup_memory_optimizations(cfg, run_output_dir):
        """Setup memory optimizations with graceful fallback"""
        import logging
        logger = logging.getLogger(__name__)
        
        optimizations = {
            'enable_memory_mapping': cfg.get('enable_memory_mapping', True),
            'enable_gradient_checkpointing': cfg.get('enable_gradient_checkpointing', True),
            'mixed_precision': cfg.get('mixed_precision', True),
            'optimize_dataloading': cfg.get('optimize_dataloading', True),
        }
        
        # Memory estimation with fallback
        try:
            import psutil
            available_memory = psutil.virtual_memory().available
            total_memory = psutil.virtual_memory().total
            
            logger.info(f"💾 System Memory: {total_memory / (1024**3):.1f}GB total, {available_memory / (1024**3):.1f}GB available")
            
            # Auto-optimize based on available memory
            if available_memory > 32 * 1024**3:  # >32GB
                optimizations['suggested_batch_size'] = 64
                optimizations['num_workers'] = 8
            elif available_memory > 16 * 1024**3:  # >16GB
                optimizations['suggested_batch_size'] = 32
                optimizations['num_workers'] = 4
            else:  # <16GB
                optimizations['suggested_batch_size'] = 16
                optimizations['num_workers'] = 2
                optimizations['enable_gradient_checkpointing'] = True
                logger.warning("⚠️  Low memory detected, enabling gradient checkpointing")
            
            budget_bytes = int(available_memory * 0.7)  # Use 70% of available memory
            
        except ImportError:
            logger.warning("⚠️  psutil not available, using conservative memory settings")
            # Conservative fallback settings
            optimizations['suggested_batch_size'] = 16
            optimizations['num_workers'] = 2
            optimizations['enable_gradient_checkpointing'] = True
            budget_bytes = 8 * 1024**3  # Assume 8GB available
            
        except Exception as e:
            logger.warning(f"⚠️  Memory detection failed: {e}, using conservative settings")
            optimizations['suggested_batch_size'] = 16
            optimizations['num_workers'] = 2
            budget_bytes = 8 * 1024**3
        
        # GPU memory check with fallback
        try:
            import torch
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                logger.info(f"🎮 GPU Memory: {gpu_memory / (1024**3):.1f}GB")
                
                if gpu_memory > 16 * 1024**3:  # >16GB GPU
                    optimizations['gpu_batch_multiplier'] = 2
                elif gpu_memory < 8 * 1024**3:   # <8GB GPU
                    optimizations['enable_gradient_checkpointing'] = True
                    logger.warning("⚠️  Low GPU memory, enabling gradient checkpointing")
        except Exception as e:
            logger.warning(f"⚠️  GPU memory detection failed: {e}")
        
        logger.info("🚀 Memory Optimizations:")
        for key, value in optimizations.items():
            if isinstance(value, bool):
                logger.info(f"   • {key}: {'✅' if value else '❌'}")
            else:
                logger.info(f"   • {key}: {value}")
        
        return budget_bytes


    # NEW: Enhanced DataModule creation (Dan)
    def create_enhanced_datamodule(cfg, budget_bytes):
        """Create enhanced DataModule with memory optimizations"""
        logger = logging.getLogger(__name__)
        
        # Get base datamodule
        base_datamodule = get_datamodule(cfg["data"]["name"],
        cfg["data"]["kwargs"],
        batch_size=cfg["training"]["batch_size"],
        )
        
        # Apply memory optimizations if enabled
        if cfg.get('enable_memory_mapping') or cfg.get('prefetch_factor', 0) > 0:
            logger.info("🔧 Applying memory optimizations to DataModule...")
            
            # Create enhanced data loaders
            enhanced_params = {
                'prefetch_factor': cfg.get('prefetch_factor', 2),
                'use_memory_mapping': cfg.get('enable_memory_mapping', False),
                'mmap_cache_dir': cfg.get('mmap_cache_dir'),
                'adaptive_prefetching': cfg.get('adaptive_prefetching', False),
            }
            
            # Replace train dataloader
            if hasattr(base_datamodule, 'train_dataloader'):
                original_train_loader = base_datamodule.train_dataloader()
                base_datamodule._enhanced_train_loader = create_enhanced_loader(
                    dataset=getattr(original_train_loader, 'dataset', None),
                    batch_size=getattr(original_train_loader, 'batch_size', 32),
                    **enhanced_params
                )
                
                # Override train_dataloader method
                base_datamodule.train_dataloader = lambda: base_datamodule._enhanced_train_loader
            
            # Replace val dataloader
            if hasattr(base_datamodule, 'val_dataloader'):
                original_val_loader = base_datamodule.val_dataloader()
                base_datamodule._enhanced_val_loader = create_enhanced_loader(
                    dataset=getattr(original_val_loader, 'dataset', None),
                    batch_size=getattr(original_val_loader, 'batch_size', 32),
                    prefetch_factor=max(1, cfg.get('prefetch_factor', 2) // 2),  # Less prefetching for validation
                    **{k: v for k, v in enhanced_params.items() if k != 'prefetch_factor'}
                )
                
                # Override val_dataloader method
                base_datamodule.val_dataloader = lambda: base_datamodule._enhanced_val_loader
            
            logger.info("✅ DataModule enhanced with memory optimizations")
        
        return base_datamodule

    # NEW: Performance monitoring callback (Dan)
    class MemoryOptimizationCallback(pl.Callback):
        """Callback to monitor memory optimization performance"""
        
        def __init__(self, log_every_n_steps=100):
            self.log_every_n_steps = log_every_n_steps
            self.step_count = 0
            
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            self.step_count += 1
            
            if self.step_count % self.log_every_n_steps == 0:
                # Log memory usage
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated() / (1024**3)
                    memory_reserved = torch.cuda.memory_reserved() / (1024**3)
                    
                    pl_module.log("memory/gpu_allocated_gb", memory_allocated)
                    pl_module.log("memory/gpu_reserved_gb", memory_reserved)
                
                # Log batch processing time
                if hasattr(trainer.datamodule, '_enhanced_train_loader'):
                    loader = trainer.datamodule._enhanced_train_loader
                    if hasattr(loader, 'batch_times') and loader.batch_times:
                        avg_batch_time = sum(loader.batch_times[-10:]) / min(10, len(loader.batch_times))
                        pl_module.log("performance/avg_batch_time", avg_batch_time)


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

    # NEW: Setup hardware and memory optimizations (Dan)
    #accelerator, backend = setup_hardware_optimizations(cfg)
    budget_bytes = setup_memory_optimizations(cfg, run_output_dir)

    # NEW: Create enhanced datamodule (Dan)
    datamodule = create_enhanced_datamodule(cfg, budget_bytes)

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
        cell_sentence_len=sentence_len        
    )

    with open(join(run_output_dir, "data_module.torch"), "wb") as f:
        # TODO-Abhi: only save necessary data
        data_module.save_state(f)

    data_module.setup(stage="fit")
    dl = data_module.train_dataloader()

    # NEW: Shared Memory und DataLoader Optimierung (Dan)
    logger.info("🔧 Starting DataLoader optimization...")

    # 1. Diagnose Shared Memory
    force_single_worker_due_to_shm = diagnose_shared_memory_issues(logger)
    if force_single_worker_due_to_shm:
        cfg["force_single_worker"] = True

    # 2. Optimiere DataModule
    data_module = optimize_datamodule_loaders(data_module, cfg, logger)

    # 3. Einfacher Test ohne Rekursion
    logger.info("🧪 Testing optimized DataLoader...")

    test_success = False
    max_attempts = 3

    for attempt in range(1, max_attempts + 1):
        try:
            # Test Train DataLoader
            logger.info(f"🧪 Testing Train DataLoader (attempt {attempt})...")
            dl = data_module.train_dataloader()
            
            logger.info(f"✅ Train DataLoader created successfully:")
            logger.info(f"   • num_workers: {dl.num_workers}")
            logger.info(f"   • batch_size: {getattr(dl, 'batch_size', 'batch_sampler_controlled')}")
            logger.info(f"   • pin_memory: {getattr(dl, 'pin_memory', False)}")
            logger.info(f"   • persistent_workers: {getattr(dl, 'persistent_workers', False)}")
            logger.info(f"   • batch_sampler: {type(getattr(dl, 'batch_sampler', None)).__name__ if getattr(dl, 'batch_sampler', None) else 'None'}")
            
            # Vorsichtiger Batch-Test
            logger.info("🧪 Testing batch loading...")
            
            try:
                batch_iter = iter(dl)
                test_batch = next(batch_iter)
                logger.info("✅ Test batch loaded successfully")
                
                # Cleanup
                del test_batch
                del batch_iter
                del dl
                
                test_success = True
                break
                
            except Exception as batch_error:
                logger.error(f"❌ Batch loading failed: {batch_error}")
                try:
                    del dl
                except:
                    pass
                raise batch_error
                
        except Exception as e:
            logger.error(f"❌ DataLoader test failed (attempt {attempt}): {e}")
            
            if attempt < max_attempts:
                if attempt == 1:
                    logger.warning("🔧 Fallback 1: Forcing single worker...")
                    cfg["force_single_worker"] = True
                    cfg["max_dataloader_workers"] = 0
                    data_module = optimize_datamodule_loaders(data_module, cfg, logger)
                elif attempt == 2:
                    logger.warning("🔧 Fallback 2: Using original DataLoader with minimal changes...")
                    # Ersetze mit sicherem Fallback
                    if hasattr(data_module, 'train_dataloader'):
                        original_train_method = data_module.train_dataloader
                        data_module.train_dataloader = safe_fallback_dataloader(original_train_method, logger)
                    if hasattr(data_module, 'val_dataloader'):
                        original_val_method = data_module.val_dataloader
                        data_module.val_dataloader = safe_fallback_dataloader(original_val_method, logger)
            else:
                logger.error("❌ All attempts failed")

    if test_success:
        logger.info("✅ DataLoader optimization and testing complete")
    else:
        logger.error("❌ DataLoader optimization failed - proceeding with original settings")

    logger.info("✅ DataLoader setup complete - proceeding with training")

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

    # NEW: Performance summary
    logger.info("📊 Training completed! Performance Summary:")
    if cfg.get('enable_memory_mapping'):
        logger.info("   • Memory mapping was enabled")
    if cfg.get('prefetch_factor', 0) > 0:
        logger.info(f"   • Prefetching factor: {cfg['prefetch_factor']}")
    if cfg.get('adaptive_prefetching'):
        logger.info("   • Adaptive prefetching was enabled")

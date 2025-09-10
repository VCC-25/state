from datetime import datetime
import json
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from ..._cli._tx._predict import run_tx_predict, add_arguments_predict
from cell_eval import score_agg_metrics
import argparse
import logging
import os
import tempfile
from pathlib import Path
import pandas as pd
from typing import Dict, List, Optional, Union

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import Callback
from .metrics_plotter import MetricsPlotter

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)

class CellEvalCallback(Callback):
    """
    Lightning callback for cell-eval evaluation during training
    """
    
    def __init__(
        self,
        eval_every_n_steps: int = 500,  # Evaluate every 500 steps (equivalent to ~5 epochs at 100 steps/epoch)
        plot_every_n_evals: int = 1,  # Plot after every 5. evaluation
        # pred_data_path: Optional[str] = None,
        # real_data_path: Optional[str] = None,
        control_pert: Optional[str] = None,
        pert_col: Optional[str] = None,
        eval_metrics: List[str] = None,
        output_dir: Optional[str] = None,
        save_predictions: bool = True,
        log_to_wandb: bool = True,
        verbose: bool = True,
        temp_checkpoint_name: str = "temp_eval_checkpoint.ckpt",
        profile: str = "minimal",  # For faster evaluation
        # Multi-Metrik Best Checkpoint Tracking
        primary_metric: str = "discrimination_score_l1",  # HMain metric for final decision
        metric_weights: Optional[Dict[str, float]] = None,  # Weighting of metrics
        metric_modes: Optional[Dict[str, str]] = None,  # “max” or “min” per metric
        improvement_threshold: float = 0.001,  # Minimum improvement for primary metric
        composite_score_method: str = "weighted_average",  # "weighted_average", "rank_based", "pareto"
        save_best_checkpoint: bool = True,
        track_all_metrics: bool = True,
        baseline_comparison: bool = True,  # Use from_baseline values     
    ):
        super().__init__()
        
        self.eval_every_n_steps = eval_every_n_steps
        self.plot_every_n_evals = plot_every_n_evals
        #print(f"pred: {pred_data_path}")
        #self.pred_data_path = pred_data_path
        #print(f"real: {real_data_path}")
        #self.real_data_path = real_data_path
        self.control_pert = control_pert
        self.pert_col = pert_col
        #self.eval_metrics = ['overlap_at_N', 'mae', 'discrimination_score_l1']
        self.output_dir = Path(output_dir) if output_dir else None
        self.save_predictions = save_predictions
        self.log_to_wandb = log_to_wandb
        self.verbose = verbose
        self.temp_checkpoint_name = temp_checkpoint_name
        self.profile = profile
        self.agg_baseline = None
        
        # Evaluation history
        self.results_history = []  # list of all results DataFrames
        self.agg_results_history = []  # list of all agg_results DataFrames
        self.eval_metadata = []  # Metadata of all evaluation
        self.eval_counter = 0
        
        # Paths for persistent storage
        self.results_save_path = os.path.join(output_dir, "eval_results_history.pkl")
        self.agg_results_save_path = os.path.join(output_dir, "eval_agg_results_history.pkl")
        self.metadata_save_path = os.path.join(output_dir, "eval_metadata.json")
        self.plots_dir = os.path.join(output_dir, "evaluation_plots")
        
        # Create plots directory
        os.makedirs(self.plots_dir, exist_ok=True)

        # Multi-Metrik Tracking
        self.primary_metric = primary_metric
        self.metric_weights = metric_weights or {}
        self.metric_modes = metric_modes or {}
        self.improvement_threshold = improvement_threshold
        self.composite_score_method = composite_score_method
        self.save_best_checkpoint = save_best_checkpoint
        self.track_all_metrics = track_all_metrics
        self.baseline_comparison = baseline_comparison
        
        # All metrics Tracking 
        self.all_metrics_history = []  # List of all metric dictionaries per step
        self.best_scores_per_metric = {}  # Best values per metric
        self.best_steps_per_metric = {}  # Best steps per metric
        self.best_composite_score = None
        self.best_composite_step = None
        self.best_checkpoint_path = None
        
        # Automatic metric detection
        self.discovered_metrics = set()
        self.metric_statistics = {}  # Min, Max, Mean, Std per metric
        
        # Default metric modes - adjusted for from_baseline values
        self.default_metric_modes = {
            'discrimination_score_l1': 'max',  # Higher is better
            'overlap_at_N': 'max',             # Higher is better
            'mae': 'min',                      # Lower is better 
            'avg_score': 'max',                # Higher is better
            'mse': 'min',                      # Lower is better
            'rmse': 'min',                     # Lower is better 
            'pearson_delta': 'max',            # Higher is better
            'r2': 'max',                       # Higher is better
            'correlation': 'max',              # Higher is better
        }
        
        # Paths
        self.scores_dir = output_dir
        self.best_scores_file = os.path.join(output_dir, "best_scores_all_metrics.json")
        self.metrics_analysis_file = os.path.join(output_dir, "metrics_analysis.json")
        
        # Load previous data
        self._load_all_metrics_history()
        
        # Initialize plotter
        self.plotter = MetricsPlotter(
            plots_dir=self.plots_dir,
            primary_metric=self.primary_metric,
            dpi=300,
            results_history=self.results_history,
            agg_results_history=self.agg_results_history,
            all_metrics_history=self.all_metrics_history,
        )

        # Cell-eval Setup
        self._setup_cell_eval()

    def create_anndata_from_predictions(self, predictions, metadata):
        """Sichere AnnData-Erstellung mit korrekten Dimensionen"""
        
        # Debug-Info
        n_predictions = len(predictions)
        n_metadata = len(metadata) if hasattr(metadata, '__len__') else metadata.shape[0]
        
        self.logger.info(f"🔍 Dimension Check:")
        self.logger.info(f"   • Predictions: {n_predictions}")
        self.logger.info(f"   • Metadata: {n_metadata}")
        
        # Stelle sicher, dass Dimensionen übereinstimmen
        if n_predictions != n_metadata:
            self.logger.warning(f"⚠️  Dimension mismatch detected!")
            
            # Option 1: Schneide Metadata ab
            if n_metadata > n_predictions:
                metadata = metadata[:n_predictions]
                self.logger.info(f"✂️  Trimmed metadata to {n_predictions} rows")
            
            # Option 2: Fülle Predictions auf (falls nötig)
            elif n_predictions > n_metadata:
                # Das sollte nicht passieren, aber zur Sicherheit
                predictions = predictions[:n_metadata]
                self.logger.info(f"✂️  Trimmed predictions to {n_metadata} rows")
        
        # Erstelle AnnData mit korrekten Dimensionen
        try:
            import anndata as ad
            import pandas as pd
            
            # Konvertiere Predictions zu numpy array
            if hasattr(predictions, 'cpu'):
                X = predictions.cpu().numpy()
            else:
                X = np.array(predictions)
            
            # Erstelle obs DataFrame mit korrekter Länge
            if isinstance(metadata, pd.DataFrame):
                obs = metadata.iloc[:X.shape[0]].copy()
            else:
                obs = pd.DataFrame(metadata[:X.shape[0]])
            
            # Validiere finale Dimensionen
            assert X.shape[0] == obs.shape[0], f"Final mismatch: X={X.shape[0]}, obs={obs.shape[0]}"
            
            # Erstelle AnnData
            adata = ad.AnnData(X=X, obs=obs)
            
            self.logger.info(f"✅ AnnData created: {adata.shape}")
            return adata
            
        except Exception as e:
            self.logger.error(f"❌ AnnData creation failed: {e}")
            raise    
    def _setup_cell_eval(self):
        """Initializes cell-eval integration"""
        try:
            # Dynamic import of cell-eval
            #from cell_eval import OptimizedMetricsEvaluator
            #from cell_eval.data import build_random_anndata, downsample_cells

            self.cell_eval_available = True
            
             
        
            #if self.pred_data_path and self.real_data_path:
            #    self.eval_enabled = True
            #    logger.info(f"Cell-eval callback enabled - Evaluation every{self.eval_every_n_steps} steps")
            #else:
            #    self.eval_enabled = False
            #    logger.warning("Cell callback: pred_data_path or real_data_path not set")
                
        except ImportError:
            self.cell_eval_available = False
            self.eval_enabled = False
            logger.warning("cell-eval not available - callback disabled")
    
    def _collect_evaluation_results(self, trainer, results, agg_results):
        """Collects and stores evaluation results"""
        
        # Metadata for this evaluation
        eval_metadata = {
            'eval_step': self.eval_counter,
            'global_step': trainer.global_step,
            'epoch': trainer.current_epoch,
            'timestamp': datetime.now().isoformat(),
            'learning_rate': trainer.optimizers[0].param_groups[0]['lr'] if trainer.optimizers else None
        }
        # Add data to history
        if results is not None and not results.is_empty():
            # Extend DataFrame with step information
            #print("Buchi results:", type(results))
            results_with_step = results.to_pandas() #pd.DataFrame(results, columns=list(results.name))
            #print("Buchi results_with_step:", results_with_step)
            results_with_step['eval_step'] = self.eval_counter
            results_with_step['global_step'] = trainer.global_step
            results_with_step['epoch'] = trainer.current_epoch
            self.results_history.append(results_with_step)
        if agg_results is not None and not agg_results.is_empty():
            # Extend DataFrame with step information
            #print("Buchi agg_results:", type(agg_results))
            #agg_results_with_step = pd.DataFrame(agg_results, columns=list(agg_results[0].keys()))
            agg_results_with_step = agg_results.to_pandas() #pd.DataFrame(agg_results, columns=agg_results.name)
            #print("Buchi agg_results_with_step:", agg_results_with_step)
            agg_results_with_step['eval_step'] = self.eval_counter
            agg_results_with_step['global_step'] = trainer.global_step
            agg_results_with_step['epoch'] = trainer.current_epoch
            self.agg_results_history.append(agg_results_with_step)
        
        self.eval_metadata.append(eval_metadata)
        self.eval_counter += 1
        
        # Store data persistently
        self._save_data()
        if self.verbose:
            logger.info(f"Collected evaluation results for step {trainer.global_step} "
                    f"(eval #{self.eval_counter})")
    def _load_all_metrics_history(self):
        """Lädt die komplette Metrik-Historie"""
        try:
            if os.path.exists(self.best_scores_file):
                with open(self.best_scores_file, 'r') as f:
                    data = json.load(f)
                    self.best_scores_per_metric = data.get('best_scores_per_metric', {})
                    self.best_steps_per_metric = data.get('best_steps_per_metric', {})
                    self.best_composite_score = data.get('best_composite_score')
                    self.best_composite_step = data.get('best_composite_step')
                    self.best_checkpoint_path = data.get('best_checkpoint_path')
                    self.discovered_metrics = set(data.get('discovered_metrics', []))
                    self.metric_statistics = data.get('metric_statistics', {})
                    
                    logger.info(f"Loaded metrics history: {len(self.discovered_metrics)} metrics tracked")
                    
            if os.path.exists(self.metrics_analysis_file):
                with open(self.metrics_analysis_file, 'r') as f:
                    self.all_metrics_history = json.load(f)
                    logger.info(f"Loaded {len(self.all_metrics_history)} historical metric records")
                    
        except Exception as e:
            logger.warning(f"Could not load metrics history: {e}")
            self._initialize_empty_tracking()

    def _initialize_empty_tracking(self):
        """Initialisiert leere Tracking-Strukturen"""
        self.best_scores_per_metric = {}
        self.best_steps_per_metric = {}
        self.best_composite_score = None
        self.best_composite_step = None
        self.best_checkpoint_path = None
        self.discovered_metrics = set()
        self.metric_statistics = {}
        self.all_metrics_history = []


    def _save_data(self):
        """Stores all collected data persistently"""
        try:
            # Save results history
            if self.results_history:
                with open(self.results_save_path, 'wb') as f:
                    pickle.dump(self.results_history, f)
            
            # Save Agg results
            if self.agg_results_history:
                with open(self.agg_results_save_path, 'wb') as f:
                    pickle.dump(self.agg_results_history, f)
            
            # Save metadata
            with open(self.metadata_save_path, 'w') as f:
                json.dump(self.eval_metadata, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Could not save evaluation history: {e}")

    def _load_existing_data(self):
        """Loads existing data at callback start"""
        try:
            if os.path.exists(self.results_save_path):
                with open(self.results_save_path, 'rb') as f:
                    self.results_history = pickle.load(f)
            
            if os.path.exists(self.agg_results_save_path):
                with open(self.agg_results_save_path, 'rb') as f:
                    self.agg_results_history = pickle.load(f)
            
            if os.path.exists(self.metadata_save_path):
                with open(self.metadata_save_path, 'r') as f:
                    self.eval_metadata = json.load(f)
                    
            self.eval_counter = len(self.eval_metadata)
            
            if self.eval_counter > 0:
                logger.info(f"Loaded {self.eval_counter} previous evaluations")
                
        except Exception as e:
            logger.warning(f"Could not load existing evaluation history: {e}")

    def _read_all_metrics_from_score_file(self, step):
        """Reads ALL metrics from the score_step_X.csv file (metric,from_baseline format)"""
        score_file = os.path.join(self.scores_dir, f"score_step_{step}.csv")
        
        if not os.path.exists(score_file):
            logger.warning(f"Score file not found: {score_file}")
            return None
        
        try:
            df = pd.read_csv(score_file)
            
            # Check expected columns
            if 'metric' not in df.columns or 'from_baseline' not in df.columns:
                logger.error(f"Expected columns 'metric' and 'from_baseline' not found in {score_file}")
                logger.info(f"Available columns: {list(df.columns)}")
                return None
            
            # Convert to dictionary
            metrics_dict = {}
            
            for _, row in df.iterrows():
                metric_name = row['metric']
                from_baseline_value = float(row['from_baseline'])
                
                metrics_dict[metric_name] = from_baseline_value
            
            # Add metadata
            metrics_dict['_step'] = step
            metrics_dict['_timestamp'] = datetime.now().isoformat()
            metrics_dict['_num_metrics'] = len(df)
            
            # Update discovered metrics
            new_metrics = set(metrics_dict.keys()) - {'_step', '_timestamp', '_num_metrics'}
            self.discovered_metrics.update(new_metrics)
            
            logger.info(f"Read {len(new_metrics)} metrics from step {step}: {list(new_metrics)}")
            logger.info(f"Metric values: {[(k, f'{v:.4f}') for k, v in metrics_dict.items() if not k.startswith('_')]}")
            
            return metrics_dict
            
        except Exception as e:
            logger.error(f"Error reading score file {score_file}: {e}")
            import traceback
            traceback.print_exc()
            return None
    def _update_metric_statistics(self, metrics_dict):
        """Updated statistics for all metrics"""
        for metric, value in metrics_dict.items():
            if metric.startswith('_'):  # Skip Metadaten
                continue
                
            if not isinstance(value, (int, float)):
                continue
                
            if metric not in self.metric_statistics:
                self.metric_statistics[metric] = {
                    'values': [],
                    'min': value,
                    'max': value,
                    'count': 0
                }
            
            stats = self.metric_statistics[metric]
            stats['values'].append(value)
            stats['min'] = min(stats['min'], value)
            stats['max'] = max(stats['max'], value)
            stats['count'] += 1
            
            # Calculate running statistics
            values = stats['values']
            stats['mean'] = np.mean(values)
            stats['std'] = np.std(values) if len(values) > 1 else 0.0
            stats['median'] = np.median(values)
            
            # Only keep the last 100 values for performance
            if len(stats['values']) > 100:
                stats['values'] = stats['values'][-100:]

    def _determine_metric_mode(self, metric_name):
        """Automatically determines whether higher or lower is better for from_baseline values"""
        if metric_name in self.metric_modes:
            return self.metric_modes[metric_name]
        
        # Search in standard modes
        for pattern, mode in self.default_metric_modes.items():
            if pattern.lower() in metric_name.lower():
                return mode
        
        # For from_baseline values: By default, higher is better
        # (positive values indicate improvement over baseline)
        logger.info(f"Unknown metric mode for '{metric_name}', defaulting to 'max' (from_baseline logic)")
        return 'max'

    def _calculate_normalized_score(self, metric_name, value):
        """Normalizes a from_baseline value to [0, 1]"""
        if metric_name not in self.metric_statistics:
            # For from_baseline values: 0 is neutral, positive values are good
            if value >= 0:
                return 0.5 + min(value * 0.1, 0.5)  # Scale positive values to [0.5, 1.0]
            else:
                return 0.5 - min(abs(value) * 0.1, 0.5)  # Scale negative values to [0.0, 0.5]
        
        stats = self.metric_statistics[metric_name]
        min_val = stats['min']
        max_val = stats['max']
        
        if min_val == max_val:
            return 0.5  # Neutral if no variation
        
        # Normalisierung to [0, 1]
        normalized = (value - min_val) / (max_val - min_val)
        
        # For from_baseline values, higher is usually better
        # (unless explicitly configured otherwise)
        mode = self._determine_metric_mode(metric_name)
        if mode == 'min':
            normalized = 1.0 - normalized
        
        return np.clip(normalized, 0.0, 1.0)
    def _calculate_composite_score(self, metrics_dict):
        """Calculates a composite score from all metrics"""
        if self.composite_score_method == "weighted_average":
            return self._calculate_weighted_average_score(metrics_dict)
        elif self.composite_score_method == "rank_based":
            return self._calculate_rank_based_score(metrics_dict)
        elif self.composite_score_method == "pareto":
            return self._calculate_pareto_score(metrics_dict)
        else:
            return self._calculate_weighted_average_score(metrics_dict)

    def _calculate_composite_score(self, metrics_dict):
        """Calculates a composite score from all metrics"""
        if self.composite_score_method == "weighted_average":
            return self._calculate_weighted_average_score(metrics_dict)
        elif self.composite_score_method == "rank_based":
            return self._calculate_rank_based_score(metrics_dict)
        elif self.composite_score_method == "pareto":
            return self._calculate_pareto_score(metrics_dict)
        else:
            return self._calculate_weighted_average_score(metrics_dict)

    def _calculate_weighted_average_score(self, metrics_dict):
        """Weighted average of all normalized metrics"""
        total_score = 0.0
        total_weight = 0.0
        
        for metric, value in metrics_dict.items():
            if metric.startswith('_') or not isinstance(value, (int, float)):
                continue
            
            # Normalized score
            normalized_score = value #self._calculate_normalized_score(metric, value)
            
            # Weighting
            weight = self.metric_weights.get(metric, 1.0)
            
            # Primary metric gets extra weight
            if metric == self.primary_metric:
                weight *= 2.0
            
            total_score += normalized_score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0

    def _calculate_rank_based_score(self, metrics_dict):
        """Rank-based score (how good is each metric compared to history)"""
        ranks = []
        
        for metric, value in metrics_dict.items():
            if metric.startswith('_') or not isinstance(value, (int, float)):
                continue
                
            if metric not in self.metric_statistics:
                continue
            
            # Calculate rank based on historical values
            historical_values = self.metric_statistics[metric]['values']
            mode = self._determine_metric_mode(metric)
            
            if mode == 'max':
                # Higher is better
                rank = sum(1 for v in historical_values if value >= v) / len(historical_values)
            else:
                # Lower is better
                rank = sum(1 for v in historical_values if value <= v) / len(historical_values)
            
            # Apply weighting
            weight = self.metric_weights.get(metric, 1.0)
            if metric == self.primary_metric:
                weight *= 2.0
                
            ranks.append(rank * weight)
        
        return np.mean(ranks) if ranks else 0.0

    def _update_best_scores(self, metrics_dict, current_step):
        """Updates the best scores for all metrics"""
        improvements = {}
        
        for metric, value in metrics_dict.items():
            if metric.startswith('_') or not isinstance(value, (int, float)):
                continue
            
            mode = self._determine_metric_mode(metric)
            
            # Check if there is an improvement
            is_improvement = False

            if metric not in self.best_scores_per_metric:
            # First measurement
                is_improvement = True
            else:
                old_value = self.best_scores_per_metric[metric]
                
                if mode == 'max':
                    is_improvement = value > old_value + self.improvement_threshold
                else:
                    is_improvement = value < old_value - self.improvement_threshold
            
            if is_improvement:
                old_value = self.best_scores_per_metric.get(metric, None)
                self.best_scores_per_metric[metric] = value
                self.best_steps_per_metric[metric] = current_step
                
                improvement_amount = value - old_value if old_value is not None else value
                improvements[metric] = {
                    'old_value': old_value,
                    'new_value': value,
                    'improvement': improvement_amount,
                    'mode': mode
                }
        
        return improvements

    def _is_composite_score_improved(self, current_composite_score):
        """Checks whether the composite score represents an improvement"""
        if self.best_composite_score is None:
            return True
        
        return current_composite_score > self.best_composite_score + self.improvement_threshold

    def _save_all_metrics_history(self):
        """Stores the complete metric history"""
        try:
            # Best Scores Summary
            best_data = {
                'best_scores_per_metric': self.best_scores_per_metric,
                'best_steps_per_metric': self.best_steps_per_metric,
                'best_composite_score': self.best_composite_score,
                'best_composite_step': self.best_composite_step,
                'best_checkpoint_path': self.best_checkpoint_path,
                'discovered_metrics': list(self.discovered_metrics),
                'metric_statistics': self.metric_statistics,
                'primary_metric': self.primary_metric,
                'composite_score_method': self.composite_score_method,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.best_scores_file, 'w') as f:
                json.dump(best_data, f, indent=2)
            
            # comnplete history
            with open(self.metrics_analysis_file, 'w') as f:
                json.dump(self.all_metrics_history, f, indent=2)
                
            logger.info(f"Saved metrics history: {len(self.discovered_metrics)} metrics, "
                       f"{len(self.all_metrics_history)} records")
                
        except Exception as e:
            logger.warning(f"Could not save metrics history: {e}")

    def _save_best_checkpoint_multi_metric(self, trainer, current_step, metrics_dict, composite_score, improvements):
        """Stores checkpoint with multi-metric information"""
        try:
            # Path for the best checkpoint
            best_checkpoint_path = os.path.join(self.output_dir, "best_checkpoint.ckpt")
            
            # saves checkpoint
            trainer.save_checkpoint(best_checkpoint_path)
            
            # Update composite tracking
            old_composite = self.best_composite_score
            self.best_composite_score = composite_score
            self.best_composite_step = current_step
            self.best_checkpoint_path = best_checkpoint_path
            
            # saves all data
            self._save_all_metrics_history()
            
            # Detailed logging
            logger.info(f"🎉 NEW BEST CHECKPOINT SAVED! Step {current_step}")
            logger.info(f"Composite Score: {old_composite:.4f} → {composite_score:.4f} "
                       f"(+{composite_score - (old_composite or 0):.4f})")
            
            logger.info(f"Individual metric improvements:")
            for metric, improvement_data in improvements.items():
                old_val = improvement_data['old_value'] if improvement_data['old_value'] is not None else -1.0
                new_val = improvement_data['new_value']
                mode = improvement_data['mode']
                
                if old_val is not None:
                    logger.info(f"  • {metric}: {old_val:.4f} → {new_val:.4f} "
                               f"({'↗' if mode == 'max' else '↘'})")
                else:
                    logger.info(f"  • {metric}: {new_val:.4f} (first measurement)")
            
            # Also save step-specific copy
            #step_best_path = os.path.join(self.output_dir, f"best_checkpoint_step_{current_step}.ckpt")
            #trainer.save_checkpoint(step_best_path)
            
            # WandB Logging
            if self.log_to_wandb and WANDB_AVAILABLE:
                try:
                    wandb_log = {
                        "best_composite_score": composite_score,
                        "best_step": trainer.global_step,
                        "checkpoint_saved": 1
                    }
                    
                    # logs all best metrics
                    for metric, value in self.best_scores_per_metric.items():
                        wandb_log[f"best_{metric}"] = value
                    
                    wandb.log(wandb_log, step=trainer.global_step)
                except:
                    pass
            
            # Create detailed summary
            self._create_comprehensive_checkpoint_summary(trainer.global_step, metrics_dict, composite_score, improvements)
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving best checkpoint: {e}")
            return False
    
    def _create_comprehensive_checkpoint_summary(self, current_step, metrics_dict, composite_score, improvements):
        """Creates a comprehensive summary of all metrics"""
        summary_file = os.path.join(self.output_dir, "comprehensive_best_checkpoint_summary.txt")
        
        try:
            with open(summary_file, 'w') as f:
                f.write("COMPREHENSIVE BEST CHECKPOINT SUMMARY\n")
                f.write("=" * 60 + "\n\n")
                
                f.write(f"Best Checkpoint Step: {current_step}\n")
                f.write(f"Composite Score: {composite_score:.6f}\n")
                f.write(f"Composite Method: {self.composite_score_method}\n")
                f.write(f"Primary Metric: {self.primary_metric}\n")
                f.write(f"Total Metrics Tracked: {len(self.discovered_metrics)}\n")
                f.write(f"Last Updated: {datetime.now().isoformat()}\n\n")
                
                f.write("CURRENT STEP METRICS:\n")
                f.write("-" * 30 + "\n")
                for metric, value in sorted(metrics_dict.items()):
                    if not metric.startswith('_'):
                        mode = self._determine_metric_mode(metric)
                        normalized = value # self._calculate_normalized_score(metric, value)
                        f.write(f"{metric:30s}: {value:10.6f} (norm: {normalized:.3f}, mode: {mode})\n")
                
                f.write("\nIMPROVEMENTS IN THIS STEP:\n")
                f.write("-" * 30 + "\n")
                if improvements:
                    for metric, imp_data in improvements.items():
                        old_val = imp_data['old_value']
                        new_val = imp_data['new_value']
                        improvement = imp_data['improvement']
                        mode = imp_data['mode']
                        
                        if old_val is not None:
                            f.write(f"{metric:30s}: {old_val:.6f} → {new_val:.6f} "
                                   f"({improvement:+.6f}) {'↗' if mode == 'max' else '↘'}\n")
                        else:
                            f.write(f"{metric:30s}: {new_val:.6f} (first measurement)\n")
                else:
                    f.write("No individual metric improvements.\n")
                
                f.write("\nALL-TIME BEST SCORES:\n")
                f.write("-" * 30 + "\n")
                for metric in sorted(self.best_scores_per_metric.keys()):
                    best_score = self.best_scores_per_metric[metric]
                    best_step = self.best_steps_per_metric[metric]
                    mode = self._determine_metric_mode(metric)
                    current_val = metrics_dict.get(metric, 'N/A')
                    
                    status = "🎯 CURRENT BEST" if best_step == current_step else f"(step {best_step})"
                    f.write(f"{metric:30s}: {best_score:10.6f} {status} (mode: {mode})\n")
                
                f.write("\nMETRIC STATISTICS:\n")
                f.write("-" * 30 + "\n")
                for metric in sorted(self.metric_statistics.keys()):
                    stats = self.metric_statistics[metric]
                    f.write(f"{metric:30s}: count={stats['count']:3d}, "
                           f"min={stats['min']:8.4f}, max={stats['max']:8.4f}, "
                           f"mean={stats['mean']:8.4f}, std={stats['std']:8.4f}\n")
                
                f.write(f"\nScore File Content (step {current_step}):\n")
                f.write("-" * 30 + "\n")
                score_file = os.path.join(self.scores_dir, f"score_step_{current_step}.csv")
                if os.path.exists(score_file):
                    df = pd.read_csv(score_file)
                    f.write(df.to_string(index=False))
                
            logger.info(f"Comprehensive summary saved to: {summary_file}")
            
        except Exception as e:
            logger.warning(f"Could not create comprehensive summary: {e}")

    def _is_score_improved(self, metric_name, current_value):
        """Checks whether the current from_baseline value represents an improvement."""
        if metric_name not in self.best_scores_per_metric:
            return True  # The first value is always an “improvement.”
        
        best_value = self.best_scores_per_metric[metric_name]
        mode = self._determine_metric_mode(metric_name)
        
        if mode == 'max':
            # Higher is better 
            improvement = current_value - best_value
            return improvement > self.improvement_threshold
        else:
            # Lower is better
            improvement = best_value - current_value
            return improvement > self.improvement_threshold

    def _update_best_scores(self, metrics_dict, current_step):
        """Updates the best scores for all from_baseline metrics"""
        improvements = {}
        
        for metric, value in metrics_dict.items():
            if metric.startswith('_') or not isinstance(value, (int, float)):
                continue
            
            # Check whether there is any improvement
            if self._is_score_improved(metric, value):
                old_value = self.best_scores_per_metric.get(metric, None)
                self.best_scores_per_metric[metric] = value
                self.best_steps_per_metric[metric] = current_step
                
                improvement_amount = value - old_value if old_value is not None else value
                improvements[metric] = {
                    'old_value': old_value,
                    'new_value': value,
                    'improvement': improvement_amount,
                    'mode': self._determine_metric_mode(metric)
                }
        
        return improvements
    def _check_and_save_best_checkpoint_multi_metric(self, trainer):
        """Main function: Multi-metric checkpoint verification"""
        if not self.save_best_checkpoint:
            return
        
        current_step = trainer.global_step
        
        # read all metrics
        metrics_dict = self._read_all_metrics_from_score_file(current_step)
        
        if metrics_dict is None:
            logger.warning(f"Could not read metrics for step {current_step}")
            return
        
        # Add to history
        self.all_metrics_history.append(metrics_dict)
        
        # Update statistics
        self._update_metric_statistics(metrics_dict)
        
        # Calculate composite score
        composite_score = self._calculate_composite_score(metrics_dict)
        
        # Check individual metric improvements
        improvements = self._update_best_scores(metrics_dict, current_step)
        
        # Check if composite score has improved
        composite_improved = self._is_composite_score_improved(composite_score)
        
        # Save checkpoint if improvement is present
        should_save = composite_improved or len(improvements) > 0
        
        if should_save:
            success = self._save_best_checkpoint_multi_metric(
                trainer, current_step, metrics_dict, composite_score, improvements
            )
            
            if success:
                # Create advanced visualizations
                self._create_multi_metric_analysis_plots(current_step)
        else:
            if self.verbose:
                logger.info(f"Step {current_step}: Composite score {composite_score:.4f} "
                           f"(best: {self.best_composite_score:.4f}), no improvements")
        
        # Save history regularly
        self._save_all_metrics_history()
    def _create_multi_metric_analysis_plots(self, current_step):
        """Delegiert Plot-Erstellung an MetricsPlotter"""
        
        if self.eval_counter % self.plot_every_n_evals != 0:
            return
        #print(f"Dan: all_metrics_history: {self.all_metrics_history}")

        self.plotter.create_all_plots(
            all_metrics_history=self.all_metrics_history,
            best_scores_per_metric=self.best_scores_per_metric,
            best_steps_per_metric=self.best_steps_per_metric,
            metric_statistics=self.metric_statistics,
            current_step=current_step,
            composite_score_method=self.composite_score_method,
            metric_weights=self.metric_weights,
            best_composite_score=self.best_composite_score,
            best_composite_step=self.best_composite_step
        )
    
    def _call_run_tx_predict(self, checkpoint_path):
        """Call run_tx_predict with the correct arguments"""
        
        # Create arguments for run_tx_predict
        class MockArgs:
            def __init__(self, output_dir, checkpoint, profile="minimal", predict_only=False,
                        prediction_batch_size=64,
                        enable_memory_mapping_predict=True,
                        prefetch_predictions=True,
                        optimize_for_speed=True,
                        cache_predictions=True,
                        advanced_prefetch=True,
                        prefetch_factor=2):
                self.output_dir = output_dir
                self.checkpoint = os.path.basename(checkpoint)  # only filename
                self.test_time_finetune = 0  # No fine-tuning during training
                self.profile = profile
                self.predict_only = predict_only
                self.prediction_batch_size=prediction_batch_size,
                self.enable_memory_mapping_predict=enable_memory_mapping_predict,
                self.prefetch_predictions=prefetch_predictions,
                self.advanced_prefetch=advanced_prefetch
                self.prefetch_factor=prefetch_factor
                self.optimize_for_speed=optimize_for_speed,
                self.cache_predictions=cache_predictions
        
        # Move checkpoint to the expected directory
        expected_checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        os.makedirs(expected_checkpoint_dir, exist_ok=True)
        expected_checkpoint_path = os.path.join(expected_checkpoint_dir, self.temp_checkpoint_name)
        
        # copy checkpoint
        import shutil
        shutil.copy2(checkpoint_path, expected_checkpoint_path)
        
        try:
            # call cell_eval baseline
            if self.agg_baseline is None:                
                logger.info("call Cell-eval baseline.")
                # create mock arguments
                args = MockArgs(
                    output_dir=self.output_dir,
                    checkpoint=expected_checkpoint_path,                
                    profile="vcc",
                    predict_only=False,
                    prediction_batch_size=64,
                    enable_memory_mapping_predict=True,
                    prefetch_predictions=True,
                    optimize_for_speed=True,
                    cache_predictions=True,
                    advanced_prefetch=True,
                    prefetch_factor=2
                )
                res_baseline, self.agg_baseline = run_tx_predict(args)
                logger.info(f"Cell-eval baseline results: {self.agg_baseline}")
            # create mock arguments
            args = MockArgs(
                output_dir=self.output_dir,
                checkpoint=expected_checkpoint_path,                
                profile="vcc",
                predict_only=False,
                prediction_batch_size=64,
                enable_memory_mapping_predict=True,
                prefetch_predictions=True,
                optimize_for_speed=True,
                cache_predictions=True,
                advanced_prefetch=True,
                prefetch_factor=2
            )
            
            # run_tx_predict aufrufen
            (results, agg_results) = run_tx_predict(args)
            #print("Buchi after run_tx_predict in _call_run_tx_predict")
            return (results, agg_results)
            
        finally:
            # Delete temporary checkpoint from checkpoints/
            if os.path.exists(expected_checkpoint_path):
                os.remove(expected_checkpoint_path)
            #print(f"Buchi no need to remove expected checkpoint path: {checkpoint_path}")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % self.eval_every_n_steps == 0 and trainer.global_step > 0:
            logger.info(f"Running dynamic evaluation at step {trainer.global_step}")
            
            temp_checkpoint_path = self._save_temp_checkpoint(trainer)
            
            try:
                # Call run_tx_predict and collect results
                results, agg_results = self._call_run_tx_predict(temp_checkpoint_path)
                
                # Collect and store results
                self._collect_evaluation_results(trainer, results, agg_results)
                
                # log metrics
                #if agg_results is not None and self.log_to_wandb:
                #    self._log_metrics(trainer, agg_results)
                                
                # run score
                score_filename = "score_step_" + str(trainer.global_step) + ".csv"
                score_agg_metrics(
                    results_user=agg_results,
                    results_base=self.agg_baseline,                    
                    output=os.path.join(self.output_dir, score_filename)
                )
    
                # From-baseline Multi-Metrik Checkpoint-Prüfung
                self._check_and_save_best_checkpoint_multi_metric(trainer)

                # Create plots (all N evaluations)
                if self.eval_counter % self.plot_every_n_evals == 0:
                    logger.info(f"Creating evaluation plots (eval #{self.eval_counter})")
                    self._create_multi_metric_analysis_plots(trainer.global_step) #self._create_evaluation_plots(trainer)
                    
            except Exception as e:
                logger.error(f"Error during dynamic evaluation: {e}")
            finally:
                self._cleanup_temp_files(temp_checkpoint_path)

    """def get_combined_dataframes(self):
        Returns the combined DataFrames for external analysis
        combined_results = None
        combined_agg_results = None
        
        if self.results_history:
            combined_results = pd.concat(self.results_history, ignore_index=True)
        
        if self.agg_results_history:
            combined_agg_results = pd.concat(self.agg_results_history, ignore_index=True)
        
        return combined_results, combined_agg_results"""
    def _save_temp_checkpoint(self, trainer):
        """Saves temporary checkpoint for run_tx_predict"""
        temp_dir = os.path.join(self.output_dir, "temp_eval")
        os.makedirs(temp_dir, exist_ok=True)
        temp_checkpoint_path = os.path.join(temp_dir, self.temp_checkpoint_name)
        
        trainer.save_checkpoint(temp_checkpoint_path)
        return temp_checkpoint_path

    def _cleanup_temp_files(self, checkpoint_path):
        """Cleanup of temporary files"""
        try:
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
            temp_dir = os.path.dirname(checkpoint_path)
            if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                os.rmdir(temp_dir)
        except Exception as e:
            logger.warning(f"Could not cleanup temp files: {e}")

    def _log_metrics(self, trainer, metrics):
        """Logs the metrics from run_tx_predict"""
        if not metrics:
            return
            
        # Forward metrics to Lightning Logger
        for metric_name, metric_value in metrics.items():
            trainer.logger.log_metrics(
                {f"eval_{metric_name}": metric_value}, 
                step=trainer.global_step
            )
        
        if self.verbose:
            logger.info(f"Step {trainer.global_step} - Evaluation metrics: {metrics}")
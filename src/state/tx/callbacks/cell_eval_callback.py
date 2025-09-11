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
        
    def _setup_cell_eval(self):
        """Initializes cell-eval integration"""
        try:
            # Dynamic import of cell-eval
            from cell_eval import MetricsEvaluator
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

    '''def _create_multi_metric_analysis_plots(self, current_step):
        """Creates comprehensive multi-metric analysis plots"""
        
        if len(self.all_metrics_history) < 2:
            return
        
        # Convert history to DataFrame
        df_history = pd.DataFrame(self.all_metrics_history)
        
        # 1. All metrics over time (normalised)
        self._plot_all_metrics_normalized(df_history, current_step)
        
        # 2. metric correlations
        self._plot_metric_correlations(df_history, current_step)
        
        # 3. Best Scores Dashboard
        self._plot_best_scores_dashboard(current_step)
        
        # 4. Composite Score Evolution
        self._plot_composite_score_evolution(df_history, current_step)
        
        # 5. Metric distributions
        self._plot_metric_distributions(df_history, current_step)

    def _plot_all_metrics_normalized(self, df_history, current_step):
        """Plot of all metrics normalised over time"""
        
        metrics = [col for col in df_history.columns if not col.startswith('_')]
        
        if not metrics:
            return
        
        try:
            fig, ax = plt.subplots(figsize=(15, 10))
            
            colors = plt.cm.tab20(np.linspace(0, 1, len(metrics)))
            
            for metric, color in zip(metrics, colors):
                if metric in df_history.columns:
                    values = df_history[metric].dropna()
                    steps = df_history.loc[values.index, '_step']
                    
                    # Normalize values for better comparability
                    normalized_values = []
                    for val in values:
                        norm_val = val #self._calculate_normalized_score(metric, val)
                        normalized_values.append(norm_val)
                    
                    ax.plot(steps, normalized_values, marker='o', color=color, 
                        label=metric, linewidth=2, markersize=4, alpha=0.8)
                    
                    # Mark best value
                    if metric in self.best_scores_per_metric:
                        best_step = self.best_steps_per_metric[metric]
                        if best_step in steps.values:
                            best_idx = steps[steps == best_step].index[0]
                            if best_idx < len(normalized_values):
                                ax.scatter(best_step, normalized_values[best_idx], 
                                        color=color, s=100, marker='*', 
                                        edgecolors='black', linewidth=1, zorder=5)
            
            ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Neutral (0.5)')
            ax.set_title(f'All Metrics Normalized Over Time (Step: {current_step})', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Normalized Score [0-1]')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.05, 1.05)
            
            plt.tight_layout()
            
            save_path = os.path.join(self.plots_dir, f'all_metrics_normalized_step_{current_step}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved normalized metrics plot: {save_path}")
            
        except Exception as e:
            logger.error(f"Error creating normalized metrics plot: {e}")
            import traceback
            traceback.print_exc()

    def _plot_metric_correlations(self, df_history, current_step):
        """Plot of correlations between metrics"""
        
        metrics = [col for col in df_history.columns if not col.startswith('_')]
        
        if len(metrics) < 2:
            return
        
        try:
            # Erstelle Korrelationsmatrix
            metric_data = df_history[metrics].dropna()
            
            if len(metric_data) < 2:
                logger.warning("Not enough data points for correlation analysis")
                return
            
            correlation_matrix = metric_data.corr()
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Heatmap der Korrelationen
            im1 = ax1.imshow(correlation_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1)
            ax1.set_title('Metric Correlations Heatmap', fontsize=14, fontweight='bold')
            ax1.set_xticks(range(len(metrics)))
            ax1.set_yticks(range(len(metrics)))
            ax1.set_xticklabels(metrics, rotation=45, ha='right')
            ax1.set_yticklabels(metrics)
            
            # Füge Korrelationswerte hinzu
            for i in range(len(metrics)):
                for j in range(len(metrics)):
                    text = ax1.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                                ha="center", va="center", color="black" if abs(correlation_matrix.iloc[i, j]) < 0.5 else "white")
            
            cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
            cbar1.set_label('Correlation Coefficient')
            
            # Scatter plot der stärksten Korrelationen
            # Finde die stärkste positive und negative Korrelation (außer Diagonale)
            mask = np.triu(np.ones_like(correlation_matrix.values, dtype=bool), k=1)
            masked_corr = correlation_matrix.values.copy()
            masked_corr[~mask] = 0
            
            max_corr_idx = np.unravel_index(np.argmax(np.abs(masked_corr)), masked_corr.shape)
            
            if max_corr_idx[0] != max_corr_idx[1]:  # Nicht die Diagonale
                metric1 = metrics[max_corr_idx[0]]
                metric2 = metrics[max_corr_idx[1]]
                corr_value = correlation_matrix.iloc[max_corr_idx[0], max_corr_idx[1]]
                
                x_data = metric_data[metric1]
                y_data = metric_data[metric2]
                
                ax2.scatter(x_data, y_data, alpha=0.6, s=50)
                ax2.set_xlabel(f'{metric1} (from baseline)')
                ax2.set_ylabel(f'{metric2} (from baseline)')
                ax2.set_title(f'Strongest Correlation: {metric1} vs {metric2}\n(r = {corr_value:.3f})')
                ax2.grid(True, alpha=0.3)
                
                # Trendlinie
                try:
                    z = np.polyfit(x_data, y_data, 1)
                    p = np.poly1d(z)
                    ax2.plot(x_data, p(x_data), "r--", alpha=0.8, linewidth=2)
                except:
                    pass
            
            plt.tight_layout()
            
            save_path = os.path.join(self.plots_dir, f'metric_correlations_step_{current_step}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved correlation plot: {save_path}")
            
        except Exception as e:
            logger.error(f"Error creating correlation plot: {e}")
            import traceback
            traceback.print_exc()

    def _plot_best_scores_dashboard(self, current_step):
        """Dashboard mit allen besten Scores"""
        
        if not self.best_scores_per_metric:
            return
        
        try:
            metrics = list(self.best_scores_per_metric.keys())
            best_scores = list(self.best_scores_per_metric.values())
            best_steps = [self.best_steps_per_metric[m] for m in metrics]
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Best Scores Dashboard (Current Step: {current_step})', 
                        fontsize=16, fontweight='bold')
            
            # 1. Best Scores Bar Chart
            colors = ['green' if score > 0 else 'red' if score < 0 else 'gray' for score in best_scores]
            bars = ax1.bar(range(len(metrics)), best_scores, color=colors, alpha=0.7)
            
            ax1.axhline(y=0, color='black', linestyle='-', alpha=0.8)
            ax1.set_title('Best From-Baseline Scores')
            ax1.set_xlabel('Metrics')
            ax1.set_ylabel('Best From-Baseline Value')
            ax1.set_xticks(range(len(metrics)))
            ax1.set_xticklabels(metrics, rotation=45, ha='right')
            ax1.grid(True, alpha=0.3)
            
            # Werte auf Balken
            for bar, value in zip(bars, best_scores):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + (0.001 if height >= 0 else -0.001),
                        f'{value:.4f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)
            
            # 2. Steps wo beste Scores erreicht wurden
            ax2.bar(range(len(metrics)), best_steps, alpha=0.7, color='skyblue')
            ax2.set_title('Steps of Best Scores')
            ax2.set_xlabel('Metrics')
            ax2.set_ylabel('Training Step')
            ax2.set_xticks(range(len(metrics)))
            ax2.set_xticklabels(metrics, rotation=45, ha='right')
            ax2.grid(True, alpha=0.3)
            
            # Aktuelle Step-Linie
            ax2.axhline(y=current_step, color='red', linestyle='--', alpha=0.8, label=f'Current Step: {current_step}')
            ax2.legend()
            
            # 3. Normalized Best Scores
            normalized_scores = [score # self._calculate_normalized_score(metric, score) 
                            for metric, score in zip(metrics, best_scores)]
            
            ax3.bar(range(len(metrics)), normalized_scores, alpha=0.7, color='orange')
            ax3.set_title('Normalized Best Scores [0-1]')
            ax3.set_xlabel('Metrics')
            ax3.set_ylabel('Normalized Score')
            ax3.set_xticks(range(len(metrics)))
            ax3.set_xticklabels(metrics, rotation=45, ha='right')
            ax3.set_ylim(0, 1)
            ax3.grid(True, alpha=0.3)
            
            # 4. Improvement Timeline
            if len(self.all_metrics_history) > 1:
                # Zeige wann jede Metrik zuletzt verbessert wurde
                steps_since_improvement = []
                for metric in metrics:
                    best_step = self.best_steps_per_metric[metric]
                    steps_since = current_step - best_step
                    steps_since_improvement.append(steps_since)
                
                colors = ['green' if steps == 0 else 'yellow' if steps < 1000 else 'red' 
                        for steps in steps_since_improvement]
                
                ax4.bar(range(len(metrics)), steps_since_improvement, color=colors, alpha=0.7)
                ax4.set_title('Steps Since Last Improvement')
                ax4.set_xlabel('Metrics')
                ax4.set_ylabel('Steps Since Best')
                ax4.set_xticks(range(len(metrics)))
                ax4.set_xticklabels(metrics, rotation=45, ha='right')
                ax4.grid(True, alpha=0.3)
                
                # Werte anzeigen
                for i, steps in enumerate(steps_since_improvement):
                    ax4.text(i, steps + max(steps_since_improvement) * 0.01, 
                            f'{steps}', ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            
            save_path = os.path.join(self.plots_dir, f'best_scores_dashboard_step_{current_step}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved best scores dashboard: {save_path}")
            
        except Exception as e:
            logger.error(f"Error creating best scores dashboard: {e}")
            import traceback
            traceback.print_exc()

    def _plot_composite_score_evolution(self, df_history, current_step):
        """Plot der Composite Score Evolution"""
        
        if len(self.all_metrics_history) < 2:
            return
        
        try:
            # Berechne Composite Scores für alle historischen Punkte
            composite_scores = []
            steps = []
            
            for record in self.all_metrics_history:
                score = self._calculate_composite_score(record)
                composite_scores.append(score)
                steps.append(record['_step'])
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # 1. Composite Score über Zeit
            ax1.plot(steps, composite_scores, 'b-o', linewidth=2, markersize=6, alpha=0.8)
            ax1.set_title(f'Composite Score Evolution ({self.composite_score_method})')
            ax1.set_xlabel('Training Step')
            ax1.set_ylabel('Composite Score')
            ax1.grid(True, alpha=0.3)
            
            # Markiere besten Score
            if self.best_composite_score is not None:
                best_idx = composite_scores.index(max(composite_scores))
                ax1.scatter(steps[best_idx], composite_scores[best_idx], 
                        color='red', s=150, marker='*', 
                        label=f'Best: {max(composite_scores):.4f} (Step {steps[best_idx]})', 
                        zorder=5, edgecolors='black', linewidth=1)
                ax1.legend()
            
            # Trend-Analyse
            if len(composite_scores) > 5:
                # Gleitender Durchschnitt
                window_size = min(5, len(composite_scores) // 2)
                moving_avg = pd.Series(composite_scores).rolling(window=window_size, center=True).mean()
                ax1.plot(steps, moving_avg, 'r--', alpha=0.7, linewidth=2, 
                        label=f'Moving Average (window={window_size})')
                ax1.legend()
            
            # 2. Composite Score Komponenten (letzter Wert)
            if self.all_metrics_history:
                last_record = self.all_metrics_history[-1]
                component_scores = {}
                
                for metric, value in last_record.items():
                    if not metric.startswith('_') and isinstance(value, (int, float)):
                        normalized = value #self._calculate_normalized_score(metric, value)
                        weight = self.metric_weights.get(metric, 1.0)
                        if metric == self.primary_metric:
                            weight *= 2.0
                        component_scores[metric] = normalized * weight
                
                if component_scores:
                    metrics = list(component_scores.keys())
                    scores = list(component_scores.values())
                    
                    # Sortiere nach Beitrag
                    sorted_items = sorted(zip(metrics, scores), key=lambda x: x[1], reverse=True)
                    metrics, scores = zip(*sorted_items)
                    
                    colors = plt.cm.viridis(np.linspace(0, 1, len(metrics)))
                    bars = ax2.bar(range(len(metrics)), scores, color=colors, alpha=0.8)
                    
                    ax2.set_title(f'Composite Score Components (Step {current_step})')
                    ax2.set_xlabel('Metrics')
                    ax2.set_ylabel('Weighted Normalized Score')
                    ax2.set_xticks(range(len(metrics)))
                    ax2.set_xticklabels(metrics, rotation=45, ha='right')
                    ax2.grid(True, alpha=0.3)
                    
                    # Werte auf Balken
                    for bar, value in zip(bars, scores):
                        height = bar.get_height()
                        ax2.text(bar.get_x() + bar.get_width()/2., height + max(scores) * 0.01,
                                f'{value:.3f}', ha='center', va='bottom', fontsize=8)
                    
                    # Markiere primäre Metrik
                    if self.primary_metric in metrics:
                        primary_idx = metrics.index(self.primary_metric)
                        bars[primary_idx].set_edgecolor('red')
                        bars[primary_idx].set_linewidth(3)
            
            plt.tight_layout()
            
            save_path = os.path.join(self.plots_dir, f'composite_score_evolution_step_{current_step}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved composite score evolution plot: {save_path}")
            
        except Exception as e:
            logger.error(f"Error creating composite score evolution plot: {e}")
            import traceback
            traceback.print_exc()

    def _plot_metric_distributions(self, df_history, current_step):
        """Plot der Metrik-Verteilungen"""
        
        metrics = [col for col in df_history.columns if not col.startswith('_')]
        
        if not metrics:
            return
        
        try:
            # Bestimme Layout
            n_metrics = len(metrics)
            n_cols = 3
            n_rows = (n_metrics + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
            fig.suptitle(f'Metric Distributions (Step: {current_step})', fontsize=16, fontweight='bold')
            
            if n_rows == 1:
                axes = [axes] if n_cols == 1 else axes
            else:
                axes = axes.flatten()
            
            for i, metric in enumerate(metrics):
                ax = axes[i]
                
                values = df_history[metric].dropna()
                
                if len(values) < 2:
                    ax.text(0.5, 0.5, f'Not enough data\nfor {metric}', 
                        ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(metric)
                    continue
                
                # Histogram
                ax.hist(values, bins=min(20, len(values)), alpha=0.7, color='skyblue', edgecolor='black')
                
                # Statistiken
                mean_val = values.mean()
                std_val = values.std()
                current_val = values.iloc[-1] if len(values) > 0 else None
                best_val = self.best_scores_per_metric.get(metric, None)
                
                # Vertikale Linien für wichtige Werte
                ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.4f}')
                
                if current_val is not None:
                    ax.axvline(current_val, color='blue', linestyle='-', alpha=0.8, 
                            label=f'Current: {current_val:.4f}')
                
                if best_val is not None:
                    ax.axvline(best_val, color='green', linestyle='-', alpha=0.8, linewidth=2,
                            label=f'Best: {best_val:.4f}')
                
                # Baseline-Linie
                ax.axvline(0, color='black', linestyle=':', alpha=0.5, label='Baseline')
                
                ax.set_title(f'{metric}\n(μ={mean_val:.4f}, σ={std_val:.4f})')
                ax.set_xlabel('From Baseline Value')
                ax.set_ylabel('Frequency')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
            
            # Verstecke leere Subplots
            for i in range(n_metrics, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            
            save_path = os.path.join(self.plots_dir, f'metric_distributions_step_{current_step}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved metric distributions plot: {save_path}")
            
        except Exception as e:
            logger.error(f"Error creating metric distributions plot: {e}")
            import traceback
            traceback.print_exc()
    
    def _create_comprehensive_checkpoint_summary(self, current_step, metrics_dict, composite_score, improvements):
        """Erstellt eine umfassende Zusammenfassung aller from_baseline Metriken"""
        summary_file = os.path.join(self.output_dir, "comprehensive_best_checkpoint_summary.txt")
        
        try:
            with open(summary_file, 'w') as f:
                f.write("COMPREHENSIVE BEST CHECKPOINT SUMMARY (FROM_BASELINE METRICS)\n")
                f.write("=" * 70 + "\n\n")
                
                f.write(f"Best Checkpoint Step: {current_step}\n")
                f.write(f"Composite Score: {composite_score:.6f}\n")
                f.write(f"Composite Method: {self.composite_score_method}\n")
                f.write(f"Primary Metric: {self.primary_metric}\n")
                f.write(f"Total Metrics Tracked: {len(self.discovered_metrics)}\n")
                f.write(f"Baseline Comparison Mode: {self.baseline_comparison}\n")
                f.write(f"Last Updated: {datetime.now().isoformat()}\n\n")
                
                f.write("CURRENT STEP FROM_BASELINE METRICS:\n")
                f.write("-" * 40 + "\n")
                for metric, value in sorted(metrics_dict.items()):
                    if not metric.startswith('_'):
                        mode = self._determine_metric_mode(metric)
                        normalized = value #self._calculate_normalized_score(metric, value)
                        
                        # Interpretation der from_baseline Werte
                        if value > 0:
                            interpretation = "📈 Better than baseline"
                        elif value < 0:
                            interpretation = "📉 Worse than baseline"
                        else:
                            interpretation = "➡️  Same as baseline"
                        
                        f.write(f"{metric:25s}: {value:10.6f} (norm: {normalized:.3f}) {interpretation}\n")
                
                f.write("\nIMPROVEMENTS IN THIS STEP:\n")
                f.write("-" * 40 + "\n")
                if improvements:
                    for metric, imp_data in improvements.items():
                        old_val = imp_data['old_value']
                        new_val = imp_data['new_value']
                        improvement = imp_data['improvement']
                        mode = imp_data['mode']
                        
                        if old_val is not None:
                            f.write(f"{metric:25s}: {old_val:8.6f} → {new_val:8.6f} "
                                   f"({improvement:+8.6f}) {'📈' if improvement > 0 else '📉'}\n")
                        else:
                            f.write(f"{metric:25s}: {new_val:8.6f} (first measurement)\n")
                else:
                    f.write("No individual metric improvements in this step.\n")
                
                f.write("\nALL-TIME BEST FROM_BASELINE SCORES:\n")
                f.write("-" * 40 + "\n")
                for metric in sorted(self.best_scores_per_metric.keys()):
                    best_score = self.best_scores_per_metric[metric]
                    best_step = self.best_steps_per_metric[metric]
                    current_val = metrics_dict.get(metric, 'N/A')
                    
                    status = "🎯 CURRENT BEST" if best_step == current_step else f"(step {best_step})"
                    
                    # Interpretation
                    if isinstance(best_score, (int, float)):
                        if best_score > 0:
                            interpretation = "📈"
                        elif best_score < 0:
                            interpretation = "📉"
                        else:
                            interpretation = "➡️"
                    else:
                        interpretation = ""
                    
                    f.write(f"{metric:25s}: {best_score:10.6f} {interpretation} {status}\n")
                
                f.write("\nMETRIC STATISTICS (FROM_BASELINE VALUES):\n")
                f.write("-" * 40 + "\n")
                for metric in sorted(self.metric_statistics.keys()):
                    stats = self.metric_statistics[metric]
                    f.write(f"{metric:25s}: count={stats['count']:3d}, "
                           f"min={stats['min']:8.4f}, max={stats['max']:8.4f}, "
                           f"mean={stats['mean']:8.4f}, std={stats['std']:8.4f}\n")
                
                f.write(f"\nOriginal Score File Content (step {current_step}):\n")
                f.write("-" * 40 + "\n")
                score_file = os.path.join(self.scores_dir, f"score_step_{current_step}.csv")
                if os.path.exists(score_file):
                    df = pd.read_csv(score_file)
                    f.write(df.to_string(index=False))
                    f.write("\n\n")
                    
                    # Zusätzliche Analyse
                    f.write("FROM_BASELINE ANALYSIS:\n")
                    f.write("-" * 20 + "\n")
                    positive_metrics = df[df['from_baseline'] > 0]['metric'].tolist()
                    negative_metrics = df[df['from_baseline'] < 0]['metric'].tolist()
                    zero_metrics = df[df['from_baseline'] == 0]['metric'].tolist()
                    
                    f.write(f"Metrics better than baseline ({len(positive_metrics)}): {positive_metrics}\n")
                    f.write(f"Metrics worse than baseline ({len(negative_metrics)}): {negative_metrics}\n")
                    f.write(f"Metrics same as baseline ({len(zero_metrics)}): {zero_metrics}\n")
                
            logger.info(f"Comprehensive summary saved to: {summary_file}")
            
        except Exception as e:
            logger.warning(f"Could not create comprehensive summary: {e}")
            import traceback
            traceback.print_exc()

    def _create_from_baseline_analysis_plots(self, current_step):
        """Erstellt spezielle Plots für from_baseline Metriken"""
        
        if len(self.all_metrics_history) < 2:
            logger.info("Need at least 2 evaluations for from_baseline plots")
            return
        
        try:
            # Konvertiere Historie zu DataFrame
            df_history = pd.DataFrame(self.all_metrics_history)
            metrics = [col for col in df_history.columns if not col.startswith('_')]
            
            if not metrics:
                return
            
            # 1. From_baseline Progression Plot
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'From-Baseline Metrics Analysis (Step: {current_step})', 
                        fontsize=16, fontweight='bold')
            
            # Plot 1: Alle Metriken über Zeit
            ax1 = axes[0, 0]
            for metric in metrics:
                if metric in df_history.columns:
                    values = df_history[metric].dropna()
                    steps = df_history.loc[values.index, '_step']
                    ax1.plot(steps, values, marker='o', label=metric, linewidth=2)
            
            ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Baseline')
            ax1.set_title('All Metrics vs Baseline Over Time')
            ax1.set_xlabel('Training Step')
            ax1.set_ylabel('From Baseline Value')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Aktuelle Werte als Bar Chart
            ax2 = axes[0, 1]
            current_metrics = self.all_metrics_history[-1]
            metric_names = [k for k in current_metrics.keys() if not k.startswith('_')]
            metric_values = [current_metrics[k] for k in metric_names]
            
            colors = ['green' if v > 0 else 'red' if v < 0 else 'gray' for v in metric_values]
            bars = ax2.bar(range(len(metric_names)), metric_values, color=colors, alpha=0.7)
            
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.8)
            ax2.set_title(f'Current From-Baseline Values (Step {current_step})')
            ax2.set_xlabel('Metrics')
            ax2.set_ylabel('From Baseline Value')
            ax2.set_xticks(range(len(metric_names)))
            ax2.set_xticklabels(metric_names, rotation=45, ha='right')
            ax2.grid(True, alpha=0.3)
            
            # Werte auf Balken anzeigen
            for bar, value in zip(bars, metric_values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.01),
                        f'{value:.3f}', ha='center', va='bottom' if height >= 0 else 'top')
            
            # Plot 3: Improvement Heatmap
            ax3 = axes[1, 0]
            
            # Erstelle Heatmap-Daten (letzte 10 Steps oder alle verfügbaren)
            recent_history = self.all_metrics_history[-10:] if len(self.all_metrics_history) > 10 else self.all_metrics_history
            
            heatmap_data = []
            step_labels = []
            
            for record in recent_history:
                step_labels.append(f"Step {record['_step']}")
                row_data = [record.get(metric, 0) for metric in metrics]
                heatmap_data.append(row_data)
            
            heatmap_data = np.array(heatmap_data)
            
            im = ax3.imshow(heatmap_data.T, cmap='RdYlGn', aspect='auto', vmin=-0.1, vmax=0.1)
            ax3.set_title('From-Baseline Heatmap (Recent Steps)')
            ax3.set_xlabel('Training Steps')
            ax3.set_ylabel('Metrics')
            ax3.set_xticks(range(len(step_labels)))
            ax3.set_xticklabels(step_labels, rotation=45, ha='right')
            ax3.set_yticks(range(len(metrics)))
            ax3.set_yticklabels(metrics)
            
            # Colorbar
            cbar = plt.colorbar(im, ax=ax3, shrink=0.8)
            cbar.set_label('From Baseline Value')
            
            # Plot 4: Composite Score Evolution
            ax4 = axes[1, 1]
            
            if len(self.all_metrics_history) > 1:
                composite_scores = []
                steps = []
                
                for record in self.all_metrics_history:
                    score = self._calculate_composite_score(record)
                    composite_scores.append(score)
                    steps.append(record['_step'])
                
                ax4.plot(steps, composite_scores, 'b-o', linewidth=2, markersize=6)
                ax4.set_title('Composite Score Evolution')
                ax4.set_xlabel('Training Step')
                ax4.set_ylabel('Composite Score')
                ax4.grid(True, alpha=0.3)
                
                # Markiere besten Score
                if self.best_composite_score is not None:
                    best_idx = composite_scores.index(max(composite_scores))
                    ax4.scatter(steps[best_idx], composite_scores[best_idx], 
                              color='red', s=100, label=f'Best: {max(composite_scores):.3f}', zorder=5)
                    ax4.legend()
            
            plt.tight_layout()
            
            save_path = os.path.join(self.plots_dir, f'from_baseline_analysis_step_{current_step}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved from-baseline analysis plot: {save_path}")
            
        except Exception as e:
            logger.error(f"Error creating from-baseline analysis plots: {e}")
            import traceback
            traceback.print_exc()

    def _create_evaluation_plots(self, trainer):
        """Creates robust plots based on available data"""
        
        if len(self.agg_results_history) < 2:
            logger.info("Need at least 2 evaluations for meaningful plots")
            return
        
        try:
            # Kombinierte DataFrames erstellen
            combined_agg_results = pd.concat(self.agg_results_history, ignore_index=True)
            combined_results = None
            
            if self.results_history:
                combined_results = pd.concat(self.results_history, ignore_index=True)
            
            logger.info(f"Creating comprehensive evaluation plots for step {trainer.global_step}")
            logger.info(f"Available agg_results columns: {list(combined_agg_results.columns)}")
            if combined_results is not None:
                logger.info(f"Available results columns: {list(combined_results.columns)}")
            
            # Plot-Stil setzen
            plt.style.use('default')
            
            # === KERN-ANALYSEN ===
            # 1. Basis-Metriken über Zeit
            self._plot_aggregated_metrics_over_time(combined_agg_results, trainer.global_step)
            
            # 2. Detaillierte Metrik-Analyse
            self._plot_current_metrics_analysis(combined_agg_results, trainer.global_step)
            
            # 3. Performance-Trends mit Vorhersagen
            self._plot_performance_trends(combined_agg_results, trainer.global_step)
            
            # 4. Training-Dynamik und Learning-Curves
            self._plot_training_dynamics(combined_agg_results, trainer.global_step)
            
            # === PERTURBATION-ANALYSEN ===
            if combined_results is not None:
                # 5. Perturbation-Ranking und Konsistenz
                self._plot_perturbation_ranking_analysis(combined_results, trainer.global_step)
                
                # 6. Korrelations- und Clustering-Analyse
                self._plot_correlation_and_clustering(combined_results, trainer.global_step)
                
                # 7. Perturbation Deep-Dive
                self._plot_perturbation_deep_dive(combined_results, trainer.global_step)
                
                # 8. Statistische Analyse
                self._plot_statistical_analysis(combined_results, trainer.global_step)
            
            # === DASHBOARD ===
            # 9. Summary Dashboard (Übersicht aller Key Metrics)
            #self._create_summary_dashboard(combined_agg_results, combined_results, trainer.global_step)
            
            # === ZUSÄTZLICHE SPEZIAL-PLOTS ===
            # 10. Performance Heatmap über Zeit (falls genug Daten)
            if combined_results is not None and len(combined_results['eval_step'].unique()) > 3:
                self._plot_performance_heatmap_over_time(combined_results, trainer.global_step)
            
            # 11. Model Comparison (falls mehrere Modelle)
            if hasattr(self, 'model_comparison_data') and self.model_comparison_data:
                self._plot_model_comparison(trainer.global_step)
            
            # From-baseline spezifische Plots
            self._create_from_baseline_analysis_plots(trainer.global_step)
            
            # Multi-Metrik Analyse Plots
            if len(self.all_metrics_history) >= 2:
                self._create_multi_metric_analysis_plots(trainer.global_step)

            logger.info(f"Successfully created comprehensive evaluation plots for step {trainer.global_step}")
            logger.info(f"Plots saved in: {self.plots_dir}")
            
            # Plot-Zusammenfassung erstellen
            self._create_plot_summary(trainer.global_step)
            
        except Exception as e:
            logger.error(f"Error creating evaluation plots: {e}")
            import traceback
            traceback.print_exc()

    def _plot_performance_heatmap_over_time(self, combined_results, current_step):
        """Performance Heatmap über Zeit für Top Perturbations"""
        
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Top 20 Perturbations aus der letzten Evaluation
        latest_eval = combined_results[combined_results['eval_step'] == combined_results['eval_step'].max()]
        
        if 'discrimination_score_l1' in latest_eval.columns:
            top_20_perts = latest_eval.nlargest(20, 'discrimination_score_l1')['perturbation'].tolist()
            
            eval_steps = sorted(combined_results['eval_step'].unique())
            
            # Heatmap-Daten erstellen
            heatmap_data = []
            
            for pert in top_20_perts:
                pert_scores = []
                for eval_step in eval_steps:
                    eval_data = combined_results[
                        (combined_results['eval_step'] == eval_step) & 
                        (combined_results['perturbation'] == pert)
                    ]
                    
                    if len(eval_data) > 0 and 'discrimination_score_l1' in eval_data.columns:
                        score = eval_data['discrimination_score_l1'].iloc[0]
                        pert_scores.append(score)
                    else:
                        pert_scores.append(np.nan)
                
                heatmap_data.append(pert_scores)
            
            heatmap_data = np.array(heatmap_data)
            
            # Heatmap erstellen
            im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', interpolation='nearest')
            
            # Achsen-Labels
            ax.set_xticks(range(len(eval_steps)))
            ax.set_xticklabels([f'Step {step}' for step in eval_steps], rotation=45)
            ax.set_yticks(range(len(top_20_perts)))
            ax.set_yticklabels([pert[:40] + '...' if len(pert) > 40 else pert 
                            for pert in top_20_perts], fontsize=8)
            
            ax.set_xlabel('Evaluation Steps')
            ax.set_ylabel('Top Perturbations')
            ax.set_title(f'Performance Heatmap Over Time (Step: {current_step})')
            
            # Colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Discrimination Score L1')
            
            plt.tight_layout()
            save_path = os.path.join(self.plots_dir, f'performance_heatmap_step_{current_step}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved performance heatmap: {save_path}")

    """def _create_plot_summary(self, current_step):
        Erstellt eine Zusammenfassung aller erstellten Plots
        
        summary_file = os.path.join(self.plots_dir, f'plot_summary_step_{current_step}.txt')
        
        plot_files = [f for f in os.listdir(self.plots_dir) 
                    if f.endswith('.png') and f'step_{current_step}' in f]
        
        with open(summary_file, 'w') as f:
            f.write(f"Evaluation Plots Summary - Step {current_step}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated {len(plot_files)} plots:\n\n")
            
            for plot_file in sorted(plot_files):
                f.write(f"- {plot_file}\n")
            
            f.write(f"\nPlots directory: {self.plots_dir}\n")
            f.write(f"Generated at: {datetime.now().isoformat()}\n")
        
        logger.info(f"Created plot summary: {summary_file}")         """                        
    def _plot_aggregated_metrics_over_time(self, combined_agg_results, current_step):
        """Plot der aggregierten Metriken über die Zeit - angepasst für verfügbare Spalten"""
        #print("Buchi _plot_aggregated_metrics_over_time")
        # Identify available metrics
        available_metrics = []
        possible_metrics = ['overlap_at_N', 'mae', 'discrimination_score_l1', 'pearson_delta', 'mse']
        
        #print(f"Buchi _plot_aggregated_metrics_over_time combined_agg_results: {combined_agg_results}")
        for metric in possible_metrics:
            if metric in combined_agg_results.columns:
                available_metrics.append(metric)
        
        if not available_metrics:
            logger.warning("No metrics available for plotting")
            return
        
        # Dynamic plot size based on available metrics
        n_metrics = len(available_metrics)
        n_cols = min(2, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 5*n_rows))
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(f'Available Metrics Over Training Steps (Current: {current_step})', 
                    fontsize=16, fontweight='bold')
        
        for idx, metric in enumerate(available_metrics):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            # Line plot with markers
            sns.lineplot(data=combined_agg_results, x='global_step', y=metric, 
                        ax=ax, marker='o', linewidth=2, markersize=6)
            
            ax.set_title(f'{metric.replace("_", " ").title()}', fontweight='bold')
            ax.set_xlabel('Global Training Step')
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.grid(True, alpha=0.3)
            
            # Add trend line
            if len(combined_agg_results) > 2:
                try:
                    z = np.polyfit(combined_agg_results['global_step'], 
                                combined_agg_results[metric], 1)
                    p = np.poly1d(z)
                    ax.plot(combined_agg_results['global_step'], 
                        p(combined_agg_results['global_step']), 
                        "--", alpha=0.7, color='red', label='Trend')
                    ax.legend()
                except:
                    pass  # If Polyfit fails
        
        # Hide empty subplots
        if n_metrics < n_rows * n_cols:
            for idx in range(n_metrics, n_rows * n_cols):
                row = idx // n_cols
                col = idx % n_cols
                ax = axes[row, col] if n_rows > 1 else axes[col]
                ax.set_visible(False)
        
        plt.tight_layout()
        
        save_path = os.path.join(self.plots_dir, f'available_metrics_step_{current_step}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved metrics plot with {n_metrics} available metrics: {save_path}")

    def _plot_current_metrics_analysis(self, combined_agg_results, current_step):
        """Special analysis for overlap_at_N, mae, discrimination_score_l1"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Detailed Metrics Analysis (Step: {current_step})', 
                    fontsize=16, fontweight='bold')
        
        # 1. All three metrics normalized in one plot
        ax1 = axes[0, 0]
        
        #print(f"Buchi _plot_current_metrics_analysis combined_agg_results: {combined_agg_results}")
        # Normalize data for comparability
        metrics_to_normalize = ['overlap_at_N', 'mae', 'discrimination_score_l1']
        available_metrics = [m for m in metrics_to_normalize if m in combined_agg_results.columns]
        
        for metric in available_metrics:
            # Min-Max Normalization
            values = combined_agg_results[metric]
            print(F"Dan values: {values}")
            normalized_values = (values - values.min()) / (values.max() - values.min()) if values.max() != values.min() else values
            print(F"Dan normalized_values: {normalized_values}")
            ax1.plot(combined_agg_results['global_step'], values, #normalized_values, 
                    marker='o', label=f'{metric} (normalized)', linewidth=2)
        
        ax1.set_title('Normalized Metrics Comparison')
        ax1.set_xlabel('Global Training Step')
        ax1.set_ylabel('Normalized Value [0-1]')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. MAE detailed analysis
        ax2 = axes[0, 1]
        if 'mae' in combined_agg_results.columns:
            mae_values = combined_agg_results['mae']
            ax2.plot(combined_agg_results['global_step'], mae_values, 
                    'b-o', linewidth=2, markersize=6)
            ax2.set_title('MAE Progression')
            ax2.set_xlabel('Global Training Step')
            ax2.set_ylabel('Mean Absolute Error')
            ax2.grid(True, alpha=0.3)
            
            # Mark best and worst MAE
            best_idx = mae_values.idxmin()
            worst_idx = mae_values.idxmax()
            ax2.scatter(combined_agg_results.loc[best_idx, 'global_step'], 
                    mae_values.iloc[best_idx], color='green', s=100, 
                    label=f'Best: {mae_values.iloc[best_idx]:.4f}', zorder=5)
            ax2.scatter(combined_agg_results.loc[worst_idx, 'global_step'], 
                    mae_values.iloc[worst_idx], color='red', s=100, 
                    label=f'Worst: {mae_values.iloc[worst_idx]:.4f}', zorder=5)
            ax2.legend()
        
        # 3. Discrimination score analysis
        ax3 = axes[1, 0]
        if 'discrimination_score_l1' in combined_agg_results.columns:
            disc_values = combined_agg_results['discrimination_score_l1']
            ax3.plot(combined_agg_results['global_step'], disc_values, 
                    'g-o', linewidth=2, markersize=6)
            ax3.set_title('Discrimination Score L1 Progression')
            ax3.set_xlabel('Global Training Step')
            ax3.set_ylabel('Discrimination Score L1')
            ax3.grid(True, alpha=0.3)
            
            # average line
            mean_score = disc_values.mean()
            ax3.axhline(y=mean_score, color='orange', linestyle='--', 
                    label=f'Mean: {mean_score:.3f}')
            ax3.legend()
        
        # 4. Overlap at N analysis
        ax4 = axes[1, 1]
        if 'overlap_at_N' in combined_agg_results.columns:
            overlap_values = combined_agg_results['overlap_at_N']
            ax4.plot(combined_agg_results['global_step'], overlap_values, 
                    'r-o', linewidth=2, markersize=6)
            ax4.set_title('Overlap at N Progression')
            ax4.set_xlabel('Global Training Step')
            ax4.set_ylabel('Overlap at N')
            ax4.grid(True, alpha=0.3)
            
            # Also show the distribution as a histogram in the inset
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes
            inset_ax = inset_axes(ax4, width="30%", height="30%", loc='upper right')
            inset_ax.hist(overlap_values, bins=10, alpha=0.7, color='red')
            inset_ax.set_title('Distribution', fontsize=8)
        
        plt.tight_layout()
        
        save_path = os.path.join(self.plots_dir, f'detailed_analysis_step_{current_step}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved detailed analysis plot: {save_path}")

    def _plot_perturbation_metrics_robust(self, combined_results, current_step):
        """Robust perturbation analysis for available columns"""
        #print(f"Buchi _plot_perturbation_metrics_robust combined_results: {combined_results}")
        
        if combined_results is None or len(combined_results) == 0:
            logger.warning("No perturbation results available for plotting")
            return
        
        # Last evaluation
        latest_eval = combined_results[combined_results['eval_step'] == combined_results['eval_step'].max()]
        
        # Identify available metrics
        available_metrics = []
        possible_metrics = ['overlap_at_N', 'mae', 'discrimination_score_l1']
        
        for metric in possible_metrics:
            if metric in latest_eval.columns:
                available_metrics.append(metric)
        
        if not available_metrics:
            logger.warning("No perturbation metrics available for plotting")
            return
        
        n_metrics = len(available_metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 8))
        if n_metrics == 1:
            axes = [axes]
        
        fig.suptitle(f'Top Perturbations Analysis (Step: {current_step})', 
                    fontsize=16, fontweight='bold')
        
        for idx, metric in enumerate(available_metrics):
            ax = axes[idx]
            
            # Top 20 Perturbations for Better Visibility
            if metric in ['discrimination_score_l1', 'overlap_at_N']:
                # Higher values are better
                top_perts = latest_eval.nlargest(20, metric)
                title_suffix = "(Higher is Better)"
            else:
                # Lower values are better (mae)
                top_perts = latest_eval.nsmallest(20, metric)
                title_suffix = "(Lower is Better)"
            
            # Horizontal bar plot
            y_pos = np.arange(len(top_perts))
            bars = ax.barh(y_pos, top_perts[metric])
            
            # Set perturbation names
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_perts['perturbation'], fontsize=8)
            ax.set_xlabel(f'{metric.replace("_", " ").title()}')
            ax.set_title(f'Top 20 Perturbations\n{metric.replace("_", " ").title()} {title_suffix}')
            ax.grid(True, alpha=0.3, axis='x')
            
            # color gradient
            colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            # Add values as text
            for i, (bar, value) in enumerate(zip(bars, top_perts[metric])):
                ax.text(bar.get_width() + bar.get_width()*0.01, bar.get_y() + bar.get_height()/2, 
                    f'{value:.3f}', ha='left', va='center', fontsize=7)
        
        plt.tight_layout()
        
        save_path = os.path.join(self.plots_dir, f'perturbation_analysis_robust_step_{current_step}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved robust perturbation analysis: {save_path}")

    def _plot_perturbation_deep_dive(self, combined_results, current_step):
        """Tiefgehende Analyse spezifischer Perturbations"""
        
        if combined_results is None or len(combined_results) == 0:
            return
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        fig.suptitle(f'Perturbation Deep Dive Analysis (Step: {current_step})', 
                    fontsize=16, fontweight='bold')
        
        latest_eval = combined_results[combined_results['eval_step'] == combined_results['eval_step'].max()]
        
        # 1. Performance Distribution by Perturbation Type
        ax1 = axes[0, 0]
        
        # Versuche Perturbation-Typen zu extrahieren (z.B. aus Namen)
        def extract_perturbation_type(pert_name):
            """Extrahiert Perturbation-Typ aus dem Namen"""
            if 'noise' in pert_name.lower():
                return 'Noise'
            elif 'blur' in pert_name.lower():
                return 'Blur'
            elif 'brightness' in pert_name.lower():
                return 'Brightness'
            elif 'contrast' in pert_name.lower():
                return 'Contrast'
            elif 'rotation' in pert_name.lower():
                return 'Rotation'
            elif 'scale' in pert_name.lower():
                return 'Scale'
            else:
                return 'Other'
        
        if 'perturbation' in latest_eval.columns:
            latest_eval['pert_type'] = latest_eval['perturbation'].apply(extract_perturbation_type)
            
            if 'discrimination_score_l1' in latest_eval.columns:
                pert_types = latest_eval['pert_type'].unique()
                
                box_data = []
                box_labels = []
                
                for pert_type in pert_types:
                    type_data = latest_eval[latest_eval['pert_type'] == pert_type]['discrimination_score_l1']
                    if len(type_data) > 0:
                        box_data.append(type_data)
                        box_labels.append(f'{pert_type}\n(n={len(type_data)})')
                
                if box_data:
                    bp = ax1.boxplot(box_data, labels=box_labels, patch_artist=True)
                    
                    # Verschiedene Farben für jeden Typ
                    colors = plt.cm.Set3(np.linspace(0, 1, len(bp['boxes'])))
                    for patch, color in zip(bp['boxes'], colors):
                        patch.set_facecolor(color)
                    
                    ax1.set_title('Performance by Perturbation Type')
                    ax1.set_ylabel('Discrimination Score L1')
                    ax1.tick_params(axis='x', rotation=45)
                    ax1.grid(True, alpha=0.3)
        
        # 2. Top vs Bottom Performers Comparison
        ax2 = axes[0, 1]
        
        if 'discrimination_score_l1' in latest_eval.columns and len(latest_eval) >= 10:
            # Top 10 und Bottom 10
            top_10 = latest_eval.nlargest(10, 'discrimination_score_l1')
            bottom_10 = latest_eval.nsmallest(10, 'discrimination_score_l1')
            
            # Vergleiche andere Metriken
            other_metrics = [col for col in ['overlap_at_N', 'mae'] if col in latest_eval.columns]
            
            if other_metrics:
                metric = other_metrics[0]
                
                top_values = top_10[metric]
                bottom_values = bottom_10[metric]
                
                # Violin Plot
                data_to_plot = [top_values, bottom_values]
                parts = ax2.violinplot(data_to_plot, positions=[1, 2], showmeans=True, showmedians=True)
                
                # Farben
                colors = ['lightgreen', 'lightcoral']
                for pc, color in zip(parts['bodies'], colors):
                    pc.set_facecolor(color)
                    pc.set_alpha(0.7)
                
                ax2.set_xticks([1, 2])
                ax2.set_xticklabels(['Top 10\nPerformers', 'Bottom 10\nPerformers'])
                ax2.set_ylabel(metric.replace('_', ' ').title())
                ax2.set_title(f'{metric} Distribution:\nTop vs Bottom Performers')
                ax2.grid(True, alpha=0.3)
        
        # 3. Performance Evolution Over Time (für ausgewählte Perturbations)
        ax3 = axes[1, 0]
        
        if len(combined_results['eval_step'].unique()) > 1:
            # Wähle Top 5 Perturbations aus der letzten Evaluation
            if 'discrimination_score_l1' in latest_eval.columns:
                top_5_perts = latest_eval.nlargest(5, 'discrimination_score_l1')['perturbation'].tolist()
                
                eval_steps = sorted(combined_results['eval_step'].unique())
                
                for pert in top_5_perts:
                    pert_data = combined_results[combined_results['perturbation'] == pert]
                    if len(pert_data) > 1:
                        ax3.plot(pert_data['eval_step'], pert_data['discrimination_score_l1'], 
                                'o-', label=pert[:20] + '...' if len(pert) > 20 else pert)
                
                ax3.set_xlabel('Evaluation Step')
                ax3.set_ylabel('Discrimination Score L1')
                ax3.set_title('Top 5 Perturbations Evolution')
                ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax3.grid(True, alpha=0.3)
        
        # 4. Metric Relationships Scatter Plot
        ax4 = axes[1, 1]
        
        available_metrics = [col for col in ['overlap_at_N', 'mae', 'discrimination_score_l1'] 
                            if col in latest_eval.columns]
        
        if len(available_metrics) >= 2:
            x_metric = available_metrics[0]
            y_metric = available_metrics[1]
            
            # Scatter plot mit Perturbation-Typ als Farbe
            if 'pert_type' in latest_eval.columns:
                pert_types = latest_eval['pert_type'].unique()
                colors = plt.cm.tab10(np.linspace(0, 1, len(pert_types)))
                
                for i, pert_type in enumerate(pert_types):
                    type_data = latest_eval[latest_eval['pert_type'] == pert_type]
                    ax4.scatter(type_data[x_metric], type_data[y_metric], 
                            c=[colors[i]], label=pert_type, alpha=0.7, s=50)
            else:
                ax4.scatter(latest_eval[x_metric], latest_eval[y_metric], alpha=0.7, s=50)
            
            ax4.set_xlabel(x_metric.replace('_', ' ').title())
            ax4.set_ylabel(y_metric.replace('_', ' ').title())
            ax4.set_title(f'{x_metric} vs {y_metric}\nby Perturbation Type')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # 5. Performance Consistency Analysis
        ax5 = axes[2, 0]
        
        if len(combined_results['eval_step'].unique()) > 2:
            # Berechne Konsistenz für jede Perturbation
            consistency_data = []
            pert_names = []
            
            for pert in combined_results['perturbation'].unique():
                pert_data = combined_results[combined_results['perturbation'] == pert]
                
                if len(pert_data) >= 3 and 'discrimination_score_l1' in pert_data.columns:
                    # Konsistenz = 1 / (Standardabweichung + kleine Konstante)
                    std_dev = pert_data['discrimination_score_l1'].std()
                    consistency = 1 / (std_dev + 0.001)  # Kleine Konstante für numerische Stabilität
                    
                    consistency_data.append(consistency)
                    pert_names.append(pert)
            
            if consistency_data:
                # Sortiere nach Konsistenz
                sorted_indices = np.argsort(consistency_data)[::-1][:15]  # Top 15
                
                top_consistency = [consistency_data[i] for i in sorted_indices]
                top_names = [pert_names[i] for i in sorted_indices]
                
                y_pos = np.arange(len(top_consistency))
                bars = ax5.barh(y_pos, top_consistency)
                
                ax5.set_yticks(y_pos)
                ax5.set_yticklabels([name[:25] + '...' if len(name) > 25 else name 
                                    for name in top_names], fontsize=8)
                ax5.set_xlabel('Consistency Score')
                ax5.set_title('Most Consistent Perturbations\n(Low Variance)')
                ax5.grid(True, alpha=0.3, axis='x')
                
                # Farbverlauf
                colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
                for bar, color in zip(bars, colors):
                    bar.set_color(color)
        
        # 6. Anomaly Detection
        ax6 = axes[2, 1]
        
        if 'discrimination_score_l1' in latest_eval.columns and len(latest_eval) >= 10:
            from sklearn.ensemble import IsolationForest
            
            # Verwende alle verfügbaren numerischen Metriken
            numeric_data = latest_eval[available_metrics].dropna()
            
            if len(numeric_data) >= 10:
                # Isolation Forest für Anomalie-Erkennung
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                anomaly_labels = iso_forest.fit_predict(numeric_data)
                
                # Anomalien markieren
                normal_mask = anomaly_labels == 1
                anomaly_mask = anomaly_labels == -1
                
                if len(available_metrics) >= 2:
                    x_vals = numeric_data[available_metrics[0]]
                    y_vals = numeric_data[available_metrics[1]]
                    
                    ax6.scatter(x_vals[normal_mask], y_vals[normal_mask], 
                            c='blue', alpha=0.6, label='Normal', s=50)
                    ax6.scatter(x_vals[anomaly_mask], y_vals[anomaly_mask], 
                            c='red', alpha=0.8, label='Anomaly', s=100, marker='^')
                    
                    ax6.set_xlabel(available_metrics[0].replace('_', ' ').title())
                    ax6.set_ylabel(available_metrics[1].replace('_', ' ').title())
                    ax6.set_title(f'Anomaly Detection\n({anomaly_mask.sum()} anomalies found)')
                    ax6.legend()
                    ax6.grid(True, alpha=0.3)
                    
                    # Anomalie-Namen anzeigen
                    if anomaly_mask.sum() > 0 and anomaly_mask.sum() <= 5:
                        anomaly_perts = latest_eval.loc[numeric_data.index[anomaly_mask], 'perturbation']
                        anomaly_text = '\n'.join([p[:20] + '...' if len(p) > 20 else p 
                                                for p in anomaly_perts])
                        ax6.text(0.02, 0.98, f'Anomalies:\n{anomaly_text}', 
                                transform=ax6.transAxes, fontsize=8, 
                                verticalalignment='top',
                                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        plt.tight_layout()
        save_path = os.path.join(self.plots_dir, f'perturbation_deep_dive_step_{current_step}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved perturbation deep dive analysis: {save_path}")

    """def _create_summary_dashboard(self, combined_agg_results, combined_results, current_step):
        Erstellt ein zusammenfassendes Dashboard mit Key Metrics
        
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        fig.suptitle(f'Training Summary Dashboard - Step {current_step}', 
                    fontsize=20, fontweight='bold')
        
        # 1. Key Metrics Cards (Top Row)
        self._create_metrics_cards(fig, gs, combined_agg_results, current_step)
        
        # 2. Main Performance Chart
        ax_main = fig.add_subplot(gs[1, :2])
        self._plot_main_performance_chart(ax_main, combined_agg_results)
        
        # 3. Training Progress
        ax_progress = fig.add_subplot(gs[1, 2:])
        self._plot_training_progress(ax_progress, combined_agg_results)
        
        # 4. Top Performers
        ax_top = fig.add_subplot(gs[2, :2])
        if combined_results is not None:
            self._plot_top_performers_summary(ax_top, combined_results)
        
        # 5. Recent Trends
        ax_trends = fig.add_subplot(gs[2, 2:])
        self._plot_recent_trends(ax_trends, combined_agg_results)
        
        # 6. Performance Distribution
        ax_dist = fig.add_subplot(gs[3, :2])
        if combined_results is not None:
            self._plot_performance_distribution_summary(ax_dist, combined_results)
        
        # 7. Training Health
        ax_health = fig.add_subplot(gs[3, 2:])
        self._plot_training_health(ax_health, combined_agg_results)
        
        # Speichern
        save_path = os.path.join(self.plots_dir, f'summary_dashboard_step_{current_step}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved summary dashboard: {save_path}")
"""
    def _create_plot_summary(self, current_step):
        """Erstellt eine Zusammenfassung aller erstellten Plots als Dashboard"""
        
        # Finde alle Plot-Dateien für diesen Step
        plot_files = [f for f in os.listdir(self.plots_dir) 
                    if f.endswith('.png') and f'step_{current_step}' in f]
        
        if not plot_files:
            logger.warning(f"No plot files found for step {current_step}")
            return
        
        # Sortiere die Plots nach Typ für bessere Anordnung
        plot_types = {
            'summary_dashboard': [],
            'performance_trends': [],
            'top_performers': [],
            'score_distribution': [],
            'correlation_analysis': [],
            'other': []
        }
        
        for plot_file in plot_files:
            if 'summary_dashboard' in plot_file:
                plot_types['summary_dashboard'].append(plot_file)
            elif 'performance_trends' in plot_file:
                plot_types['performance_trends'].append(plot_file)
            elif 'top_performers' in plot_file:
                plot_types['top_performers'].append(plot_file)
            elif 'score_distribution' in plot_file:
                plot_types['score_distribution'].append(plot_file)
            elif 'correlation' in plot_file:
                plot_types['correlation_analysis'].append(plot_file)
            else:
                plot_types['other'].append(plot_file)
        
        # Erstelle HTML Summary mit eingebetteten Bildern
        html_summary = self._create_html_summary(current_step, plot_types)
        
        # Erstelle Text Summary
        txt_summary = self._create_text_summary(current_step, plot_files)
        
        logger.info(f"Created plot summaries for step {current_step}")

    def _create_html_summary(self, current_step, plot_types):
        """Erstellt eine HTML-Zusammenfassung mit eingebetteten Plots"""
        
        html_file = os.path.join(self.plots_dir, f'plot_summary_step_{current_step}.html')
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Evaluation Plots Summary - Step {current_step}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; text-align: center; }}
                h2 {{ color: #666; border-bottom: 2px solid #ddd; }}
                .plot-container {{ margin: 20px 0; text-align: center; }}
                .plot-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(600px, 1fr)); gap: 20px; }}
                img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 8px; }}
                .summary-info {{ background-color: #f5f5f5; padding: 15px; border-radius: 8px; margin: 20px 0; }}
                .plot-section {{ margin: 30px 0; }}
            </style>
        </head>
        <body>
            <h1>Evaluation Plots Summary - Step {current_step}</h1>
            
            <div class="summary-info">
                <h3>Summary Information</h3>
                <p><strong>Current Step:</strong> {current_step}</p>
                <p><strong>Total Plots:</strong> {sum(len(plots) for plots in plot_types.values())}</p>
                <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Plots Directory:</strong> {self.plots_dir}</p>
            </div>
        """
        
        # Füge Plots nach Kategorien hinzu
        for category, plots in plot_types.items():
            if plots:
                html_content += f"""
                <div class="plot-section">
                    <h2>{category.replace('_', ' ').title()}</h2>
                    <div class="plot-grid">
                """
                
                for plot_file in sorted(plots):
                    plot_path = os.path.join(self.plots_dir, plot_file)
                    if os.path.exists(plot_path):
                        html_content += f"""
                        <div class="plot-container">
                            <h4>{plot_file}</h4>
                            <img src="{plot_file}" alt="{plot_file}">
                        </div>
                        """
                
                html_content += """
                    </div>
                </div>
                """
        
        html_content += """
        </body>
        </html>
        """
        
        with open(html_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Created HTML summary: {html_file}")
        return html_file

    def _create_text_summary(self, current_step, plot_files):
        """Erstellt eine Text-Zusammenfassung der Plots"""
        
        summary_file = os.path.join(self.plots_dir, f'plot_summary_step_{current_step}.txt')
        
        with open(summary_file, 'w') as f:
            f.write(f"Evaluation Plots Summary - Step {current_step}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated {len(plot_files)} plots:\n\n")
            
            # Gruppiere Plots nach Typ
            plot_categories = {}
            for plot_file in sorted(plot_files):
                # Extrahiere Kategorie aus Dateiname
                if 'summary_dashboard' in plot_file:
                    category = 'Summary Dashboards'
                elif 'performance_trends' in plot_file:
                    category = 'Performance Trends'
                elif 'top_performers' in plot_file:
                    category = 'Top Performers'
                elif 'score_distribution' in plot_file:
                    category = 'Score Distributions'
                elif 'correlation' in plot_file:
                    category = 'Correlation Analysis'
                else:
                    category = 'Other Plots'
                
                if category not in plot_categories:
                    plot_categories[category] = []
                plot_categories[category].append(plot_file)
            
            # Schreibe kategorisierte Liste
            for category, files in plot_categories.items():
                f.write(f"{category}:\n")
                f.write("-" * len(category) + "\n")
                for plot_file in files:
                    # Dateigröße hinzufügen
                    file_path = os.path.join(self.plots_dir, plot_file)
                    if os.path.exists(file_path):
                        file_size = os.path.getsize(file_path)
                        size_mb = file_size / (1024 * 1024)
                        f.write(f"  • {plot_file} ({size_mb:.2f} MB)\n")
                    else:
                        f.write(f"  • {plot_file} (file not found)\n")
                f.write("\n")
            
            f.write(f"Plots directory: {self.plots_dir}\n")
            f.write(f"Generated at: {datetime.now().isoformat()}\n")
            
            # Füge Statistiken hinzu
            f.write("\nFile Statistics:\n")
            f.write("-" * 16 + "\n")
            total_size = 0
            for plot_file in plot_files:
                file_path = os.path.join(self.plots_dir, plot_file)
                if os.path.exists(file_path):
                    total_size += os.path.getsize(file_path)
            
            f.write(f"Total files: {len(plot_files)}\n")
            f.write(f"Total size: {total_size / (1024 * 1024):.2f} MB\n")
            f.write(f"Average file size: {(total_size / len(plot_files)) / (1024 * 1024):.2f} MB\n")
        
        logger.info(f"Created text summary: {summary_file}")
        return summary_file

    def _create_combined_dashboard(self, current_step):
        """Erstellt ein kombiniertes Dashboard aus existierenden Plots (optional)"""
        
        # Finde die wichtigsten Plots
        key_plots = []
        plot_files = [f for f in os.listdir(self.plots_dir) 
                    if f.endswith('.png') and f'step_{current_step}' in f]
        
        # Priorisiere bestimmte Plot-Typen
        priority_types = ['summary_dashboard', 'performance_trends', 'top_performers', 'score_distribution']
        
        for plot_type in priority_types:
            matching_plots = [f for f in plot_files if plot_type in f]
            if matching_plots:
                key_plots.extend(matching_plots[:2])  # Max 2 pro Typ
        
        if len(key_plots) < 4:
            # Fülle mit anderen Plots auf
            other_plots = [f for f in plot_files if f not in key_plots]
            key_plots.extend(other_plots[:4-len(key_plots)])
        
        if not key_plots:
            logger.warning("No plots found for combined dashboard")
            return
        
        # Erstelle 2x2 Grid mit den wichtigsten Plots
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle(f'Combined Dashboard - Step {current_step}', fontsize=20, fontweight='bold')
        
        for i, plot_file in enumerate(key_plots[:4]):
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            plot_path = os.path.join(self.plots_dir, plot_file)
            if os.path.exists(plot_path):
                try:
                    img = plt.imread(plot_path)
                    ax.imshow(img)
                    ax.set_title(plot_file.replace(f'_step_{current_step}', '').replace('.png', '').replace('_', ' ').title())
                    ax.axis('off')
                except Exception as e:
                    ax.text(0.5, 0.5, f'Error loading\n{plot_file}', 
                        ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'Error: {plot_file}')
            else:
                ax.text(0.5, 0.5, f'File not found:\n{plot_file}', 
                    ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Missing: {plot_file}')
        
        # Leere übrige Subplots
        for i in range(len(key_plots), 4):
            row, col = i // 2, i % 2
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        combined_path = os.path.join(self.plots_dir, f'combined_dashboard_step_{current_step}.png')
        plt.savefig(combined_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Created combined dashboard: {combined_path}")


    def _calculate_health_indicators(self, combined_agg_results):
        """
        Berechnet verschiedene Gesundheitsindikatoren für das Training
        
        Args:
            combined_agg_results: DataFrame mit aggregierten Trainingsergebnissen
            
        Returns:
            dict: Dictionary mit Gesundheitsindikatoren (Werte zwischen 0 und 1)
        """
        if len(combined_agg_results) < 2:
            return {}
        
        health_indicators = {}
        
        try:
            # 1. Performance Stability (Stabilität der Hauptmetrik)
            if 'discrimination_score_l1' in combined_agg_results.columns:
                disc_scores = combined_agg_results['discrimination_score_l1'].dropna()
                if len(disc_scores) > 1:
                    # Berechne Variationskoeffizient (niedrig = stabil = gut)
                    cv = disc_scores.std() / disc_scores.mean() if disc_scores.mean() != 0 else 1
                    stability = max(0, 1 - min(cv, 1))  # Invertiere und normalisiere
                    health_indicators['Performance\nStability'] = stability
            
            # 2. Learning Progress (Fortschritt über Zeit)
            if 'discrimination_score_l1' in combined_agg_results.columns:
                disc_scores = combined_agg_results['discrimination_score_l1'].dropna()
                if len(disc_scores) >= 3:
                    # Berechne Trend der letzten Werte
                    recent_trend = np.polyfit(range(len(disc_scores)), disc_scores, 1)[0]
                    # Normalisiere Trend (positive Trends sind gut)
                    progress = max(0, min(1, (recent_trend + 0.1) / 0.2))  # Anpassbare Skalierung
                    health_indicators['Learning\nProgress'] = progress
            
            # 3. Error Convergence (MAE sollte sinken)
            if 'mae' in combined_agg_results.columns:
                mae_values = combined_agg_results['mae'].dropna()
                if len(mae_values) >= 3:
                    # Negativer Trend bei MAE ist gut
                    mae_trend = np.polyfit(range(len(mae_values)), mae_values, 1)[0]
                    convergence = max(0, min(1, (-mae_trend + 0.1) / 0.2))
                    health_indicators['Error\nConvergence'] = convergence
            
            # 4. Consistency (Konsistenz zwischen Evaluationen)
            if len(combined_agg_results) >= 3:
                # Berechne durchschnittliche Änderung zwischen aufeinanderfolgenden Evaluationen
                changes = []
                for col in ['discrimination_score_l1', 'mae']:
                    if col in combined_agg_results.columns:
                        values = combined_agg_results[col].dropna()
                        if len(values) > 1:
                            diffs = np.abs(np.diff(values))
                            changes.extend(diffs)
                
                if changes:
                    avg_change = np.mean(changes)
                    # Niedrige Änderungen = hohe Konsistenz
                    consistency = max(0, 1 - min(avg_change * 10, 1))  # Skalierungsfaktor anpassbar
                    health_indicators['Consistency'] = consistency
            
            # 5. Overlap Quality (falls Overlap-Metriken verfügbar)
            overlap_cols = [col for col in combined_agg_results.columns if 'overlap_at_' in col]
            if overlap_cols:
                latest_overlap = combined_agg_results[overlap_cols[0]].iloc[-1]
                # Höherer Overlap ist normalerweise besser
                overlap_quality = min(1, max(0, latest_overlap))
                health_indicators['Overlap\nQuality'] = overlap_quality
            
            # 6. Training Momentum (Beschleunigung des Lernens)
            if 'discrimination_score_l1' in combined_agg_results.columns and len(combined_agg_results) >= 4:
                disc_scores = combined_agg_results['discrimination_score_l1'].dropna()
                if len(disc_scores) >= 4:
                    # Berechne zweite Ableitung (Beschleunigung)
                    recent_scores = disc_scores.tail(4).values
                    first_diff = np.diff(recent_scores)
                    second_diff = np.diff(first_diff)
                    avg_acceleration = np.mean(second_diff)
                    
                    # Positive Beschleunigung ist gut
                    momentum = max(0, min(1, (avg_acceleration + 0.05) / 0.1))
                    health_indicators['Training\nMomentum'] = momentum
            
            # 7. Data Quality Indicator (basierend auf Varianz der Ergebnisse)
            if len(combined_agg_results) >= 3:
                numeric_cols = combined_agg_results.select_dtypes(include=[np.number]).columns
                variances = []
                for col in numeric_cols:
                    if col not in ['global_step', 'eval_step']:
                        values = combined_agg_results[col].dropna()
                        if len(values) > 1 and values.std() > 0:
                            # Normalisierte Varianz
                            cv = values.std() / abs(values.mean()) if values.mean() != 0 else 1
                            variances.append(cv)
                
                if variances:
                    avg_variance = np.mean(variances)
                    # Niedrige Varianz deutet auf gute Datenqualität hin
                    data_quality = max(0, 1 - min(avg_variance, 1))
                    health_indicators['Data\nQuality'] = data_quality
            
            # 8. Overall Performance Level
            if 'discrimination_score_l1' in combined_agg_results.columns:
                latest_performance = combined_agg_results['discrimination_score_l1'].iloc[-1]
                # Normalisiere basierend auf erwarteten Werten (anpassbar)
                performance_level = min(1, max(0, latest_performance / 0.8))  # Annahme: 0.8 ist sehr gut
                health_indicators['Performance\nLevel'] = performance_level
            
        except Exception as e:
            print(f"Warning: Error calculating health indicators: {e}")
            # Fallback: Minimale Indikatoren
            health_indicators = {
                'Performance\nStability': 0.5,
                'Learning\nProgress': 0.5,
                'Data\nQuality': 0.5
            }
        
        return health_indicators

    def _create_metrics_cards(self, fig, gs, combined_agg_results, current_step):
        """Erstellt Metric Cards für das Dashboard"""
        
        latest_data = combined_agg_results.iloc[-1] if len(combined_agg_results) > 0 else None
        
        if latest_data is None:
            return
        
        # Metric Cards
        metrics_info = [
            ('Current Step', current_step, 'step'),
            ('Discrimination Score', latest_data.get('discrimination_score_l1', 'N/A'), 'score'),
            ('MAE', latest_data.get('mae', 'N/A'), 'error'),
            ('Overlap at N', latest_data.get('overlap_at_N', 'N/A'), 'overlap')
        ]
        
        for i, (title, value, metric_type) in enumerate(metrics_info):
            ax = fig.add_subplot(gs[0, i])
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            
            # Hintergrundfarbe basierend auf Metric-Typ
            colors = {'step': '#E3F2FD', 'score': '#E8F5E8', 'error': '#FFF3E0', 'overlap': '#F3E5F5'}
            bg_color = colors.get(metric_type, '#F5F5F5')
            
            # Card-Hintergrund
            rect = plt.Rectangle((0.05, 0.2), 0.9, 0.6, 
                            facecolor=bg_color, edgecolor='gray', linewidth=2)
            ax.add_patch(rect)
            
            # Titel
            ax.text(0.5, 0.7, title, ha='center', va='center', 
                fontsize=12, fontweight='bold', transform=ax.transAxes)
            
            # Wert
            if isinstance(value, (int, float)) and value != 'N/A':
                if metric_type == 'step':
                    value_text = f"{int(value):,}"
                else:
                    value_text = f"{value:.4f}"
            else:
                value_text = str(value)
            
            ax.text(0.5, 0.4, value_text, ha='center', va='center', 
                fontsize=16, fontweight='bold', transform=ax.transAxes)
            
            # Trend-Indikator (falls mehr als ein Datenpunkt)
            if len(combined_agg_results) > 1 and metric_type != 'step':
                metric_col = {'score': 'discrimination_score_l1', 
                            'error': 'mae', 
                            'overlap': 'overlap_at_N'}.get(metric_type)
                
                if metric_col and metric_col in combined_agg_results.columns:
                    current_val = combined_agg_results[metric_col].iloc[-1]
                    prev_val = combined_agg_results[metric_col].iloc[-2]
                    
                    if pd.notna(current_val) and pd.notna(prev_val) and prev_val != 0:
                        change = (current_val - prev_val) / abs(prev_val) * 100
                        
                        # Trend-Symbol
                        if abs(change) > 0.1:  # Nur signifikante Änderungen
                            if change > 0:
                                symbol = '↗' if metric_type != 'error' else '↘'  # Für Error ist weniger besser
                                color = 'green' if metric_type != 'error' else 'red'
                            else:
                                symbol = '↘' if metric_type != 'error' else '↗'
                                color = 'red' if metric_type != 'error' else 'green'
                            
                            ax.text(0.85, 0.8, symbol, ha='center', va='center', 
                                fontsize=20, color=color, transform=ax.transAxes)

    def _plot_main_performance_chart(self, ax, combined_agg_results):
        """Haupt-Performance-Chart"""
        
        available_metrics = [col for col in ['discrimination_score_l1', 'mae', 'overlap_at_N'] 
                            if col in combined_agg_results.columns]
        
        if not available_metrics:
            ax.text(0.5, 0.5, 'No metrics available', ha='center', va='center', 
                transform=ax.transAxes, fontsize=14)
            return
        
        steps = combined_agg_results['global_step']
        
        # Doppelte Y-Achse für verschiedene Metriken
        if 'discrimination_score_l1' in available_metrics:
            line1 = ax.plot(steps, combined_agg_results['discrimination_score_l1'], 
                        'b-o', linewidth=3, markersize=8, label='Discrimination Score L1')
            ax.set_ylabel('Discrimination Score L1', color='b', fontweight='bold')
            ax.tick_params(axis='y', labelcolor='b')
        
        if 'mae' in available_metrics:
            ax2 = ax.twinx()
            line2 = ax2.plot(steps, combined_agg_results['mae'], 
                            'r-s', linewidth=3, markersize=8, label='MAE')
            ax2.set_ylabel('MAE', color='r', fontweight='bold')
            ax2.tick_params(axis='y', labelcolor='r')
        
        ax.set_xlabel('Global Step', fontweight='bold')
        ax.set_title('Main Performance Metrics', fontweight='bold', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Legende
        lines1 = ax.get_lines()
        labels1 = [l.get_label() for l in lines1]
        
        if 'mae' in available_metrics:
            lines2 = ax2.get_lines()
            labels2 = [l.get_label() for l in lines2]
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        else:
            ax.legend(loc='upper left')

    def _plot_training_progress(self, ax, combined_agg_results):
        """Training Progress Visualization"""
        
        steps = combined_agg_results['global_step']
        epochs = combined_agg_results.get('epoch', range(len(steps)))
        
        # Progress Bar Style
        if len(steps) > 1:
            progress = (steps - steps.min()) / (steps.max() - steps.min())
            
            # Erstelle Progress-Balken
            for i, (step, prog) in enumerate(zip(steps, progress)):
                color = plt.cm.viridis(prog)
                ax.barh(i, prog, color=color, alpha=0.7, height=0.8)
                
                # Step-Nummer anzeigen
                ax.text(prog + 0.02, i, f'{int(step):,}', 
                    va='center', fontsize=10, fontweight='bold')
        
        ax.set_xlim(0, 1.2)
        ax.set_ylim(-0.5, len(steps) - 0.5)
        ax.set_xlabel('Training Progress', fontweight='bold')
        ax.set_ylabel('Evaluation #', fontweight='bold')
        ax.set_title('Training Progress Overview', fontweight='bold', fontsize=14)
        
        # Y-Achse Labels
        ax.set_yticks(range(len(steps)))
        ax.set_yticklabels([f'Eval {i+1}' for i in range(len(steps))])

    def _plot_top_performers_summary(self, ax, combined_results):
        """Top Performers Summary"""
        
        latest_eval = combined_results[combined_results['eval_step'] == combined_results['eval_step'].max()]
        
        if 'discrimination_score_l1' in latest_eval.columns:
            top_5 = latest_eval.nlargest(5, 'discrimination_score_l1')
            
            y_pos = np.arange(len(top_5))
            bars = ax.barh(y_pos, top_5['discrimination_score_l1'])
            
            # Farbverlauf
            colors = plt.cm.RdYlGn(np.linspace(0.5, 1, len(bars)))
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels([name[:30] + '...' if len(name) > 30 else name 
                            for name in top_5['perturbation']], fontsize=10)
            ax.set_xlabel('Discrimination Score L1', fontweight='bold')
            ax.set_title('Top 5 Performing Perturbations', fontweight='bold', fontsize=14)
            ax.grid(True, alpha=0.3, axis='x')
            
            # Werte als Text
            for i, (bar, value) in enumerate(zip(bars, top_5['discrimination_score_l1'])):
                ax.text(bar.get_width() + bar.get_width()*0.01, 
                    bar.get_y() + bar.get_height()/2, 
                    f'{value:.3f}', ha='left', va='center', fontweight='bold')

    def _plot_recent_trends(self, ax, combined_agg_results):
        """Recent Trends Analysis"""
        
        if len(combined_agg_results) < 3:
            ax.text(0.5, 0.5, 'Need more data\nfor trend analysis', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
            return
        
        # Letzte 5 Evaluationen
        recent_data = combined_agg_results.tail(5)
        
        available_metrics = [col for col in ['discrimination_score_l1', 'mae', 'overlap_at_N'] 
                            if col in recent_data.columns]
        
        for metric in available_metrics:
            values = recent_data[metric]
            steps = recent_data['global_step']
            
            # Normalisierung für bessere Vergleichbarkeit
            #if values.max() != values.min():
            #    normalized_values = (values - values.min()) / (values.max() - values.min())
            #else:
            normalized_values = values
            
            ax.plot(range(len(steps)), normalized_values, 'o-', 
                linewidth=2, markersize=8, label=f'{metric}')
            
            # Trend-Linie
            if len(values) > 2:
                z = np.polyfit(range(len(steps)), normalized_values, 1)
                p = np.poly1d(z)
                ax.plot(range(len(steps)), p(range(len(steps))), 
                    '--', alpha=0.7, linewidth=2)
        
        ax.set_xticks(range(len(recent_data)))
        ax.set_xticklabels([f'Step {int(s):,}' for s in recent_data['global_step']], 
                        rotation=45)
        ax.set_ylabel('Normalized Value', fontweight='bold')
        ax.set_title('Recent Trends (Last 5 Evaluations)', fontweight='bold', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_performance_distribution_summary(self, ax, combined_results):
        """Performance Distribution Summary"""
        
        latest_eval = combined_results[combined_results['eval_step'] == combined_results['eval_step'].max()]
        
        if 'discrimination_score_l1' in latest_eval.columns:
            values = latest_eval['discrimination_score_l1'].dropna()
            
            # Histogram mit KDE
            ax.hist(values, bins=20, alpha=0.7, color='skyblue', density=True, edgecolor='black')
            
            # KDE-Kurve
            try:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(values)
                x_range = np.linspace(values.min(), values.max(), 100)
                ax.plot(x_range, kde(x_range), 'r-', linewidth=3, label='KDE')
            except ImportError:
                pass
            
            # Statistiken
            mean_val = values.mean()
            median_val = values.median()
            
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
            ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.3f}')
            
            ax.set_xlabel('Discrimination Score L1', fontweight='bold')
            ax.set_ylabel('Density', fontweight='bold')
            ax.set_title('Performance Distribution (Latest Eval)', fontweight='bold', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)

    def _plot_training_health(self, ax, combined_agg_results):
        """Training Health Indicators"""
        
        if len(combined_agg_results) < 3:
            ax.text(0.5, 0.5, 'Need more data\nfor health analysis', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
            return
        
        # Health Indicators berechnen
        health_indicators = {}
        
        # 1. Trend Health (ist Performance steigend?)
        if 'discrimination_score_l1' in combined_agg_results.columns:
            values = combined_agg_results['discrimination_score_l1']
            if len(values) >= 3:
                recent_trend = np.polyfit(range(len(values)), values, 1)[0]
                health_indicators['Trend'] = min(max(recent_trend * 100, -1), 1)  # Normalisiert
        
        # 2. Stability Health (niedrige Varianz ist gut)
        if 'discrimination_score_l1' in combined_agg_results.columns:
            values = combined_agg_results['discrimination_score_l1']
            if len(values) >= 3:
                stability = 1 / (1 + values.std())  # Invertiert: niedrige Std = hohe Stabilität
                health_indicators['Stability'] = min(stability, 1)
        
        # 3. Progress Health (Verbesserung über Zeit)
        if 'discrimination_score_l1' in combined_agg_results.columns:
            values = combined_agg_results['discrimination_score_l1']
            if len(values) >= 2:
                first_half = values[:len(values)//2].mean()
                second_half = values[len(values)//2:].mean()
                if first_half != 0:
                    progress = (second_half - first_half) / abs(first_half)
                    health_indicators['Progress'] = min(max(progress, -1), 1)
        
        # 4. Consistency Health (gleichmäßige Verbesserung)
        if len(combined_agg_results) >= 4:
            steps = combined_agg_results['global_step']
            step_diffs = np.diff(steps)
            consistency = 1 / (1 + np.std(step_diffs) / np.mean(step_diffs))
            health_indicators['Consistency'] = min(consistency, 1)
        
        if not health_indicators:
            ax.text(0.5, 0.5, 'Insufficient data\nfor health indicators', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
            return
        
        # Health Indicators visualisieren
        indicators = list(health_indicators.keys())
        values = list(health_indicators.values())
        
        # Radial/Polar Plot für Health Dashboard
        angles = np.linspace(0, 2 * np.pi, len(indicators), endpoint=False).tolist()
        values += values[:1]  # Schließe den Kreis
        angles += angles[:1]
        
        ax.clear()
        ax = plt.subplot(111, projection='polar')
        
        # Plot
        ax.plot(angles, values, 'o-', linewidth=3, color='blue')
        ax.fill(angles, values, alpha=0.25, color='blue')
        
        # Labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(indicators)
        ax.set_ylim(0, 1)
        
        # Farb-Kodierung für Gesundheitszonen
        ax.fill_between(angles, 0, 0.3, alpha=0.1, color='red', label='Poor')
        ax.fill_between(angles, 0.3, 0.7, alpha=0.1, color='yellow', label='Fair')
        ax.fill_between(angles, 0.7, 1, alpha=0.1, color='green', label='Good')
        
        ax.set_title('Training Health Dashboard', fontweight='bold', fontsize=14, pad=20)
        
        # Gesamt-Health-Score
        overall_health = np.mean(values[:-1])  # Ohne den duplizierten Wert
        health_status = 'Good' if overall_health > 0.7 else 'Fair' if overall_health > 0.3 else 'Poor'
        health_color = 'green' if overall_health > 0.7 else 'orange' if overall_health > 0.3 else 'red'
        
        ax.text(0, -0.3, f'Overall Health: {overall_health:.2f}\nStatus: {health_status}', 
            ha='center', va='center', transform=ax.transAxes, 
            fontsize=12, fontweight='bold', color=health_color,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    def _plot_statistical_analysis(self, combined_results, current_step):
        """Erweiterte statistische Analyse"""
        
        if combined_results is None or len(combined_results) == 0:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Statistical Analysis (Step: {current_step})', 
                    fontsize=16, fontweight='bold')
        
        latest_eval = combined_results[combined_results['eval_step'] == combined_results['eval_step'].max()]
        
        # 1. Q-Q Plot für Normalverteilung
        ax1 = axes[0, 0]
        if 'discrimination_score_l1' in latest_eval.columns:
            try:
                from scipy import stats
                values = latest_eval['discrimination_score_l1'].dropna()
                stats.probplot(values, dist="norm", plot=ax1)
                ax1.set_title('Q-Q Plot (Normal Distribution)')
                ax1.grid(True, alpha=0.3)
            except ImportError:
                ax1.text(0.5, 0.5, 'scipy not available\nfor Q-Q plot', 
                        ha='center', va='center', transform=ax1.transAxes)
        
        # 2. Bootstrap Confidence Intervals
        ax2 = axes[0, 1]
        if 'discrimination_score_l1' in latest_eval.columns:
            values = latest_eval['discrimination_score_l1'].dropna()
            
            # Bootstrap sampling
            n_bootstrap = 1000
            bootstrap_means = []
            
            for _ in range(n_bootstrap):
                sample = np.random.choice(values, size=len(values), replace=True)
                bootstrap_means.append(np.mean(sample))
            
            bootstrap_means = np.array(bootstrap_means)
            
            # Confidence intervals
            ci_lower = np.percentile(bootstrap_means, 2.5)
            ci_upper = np.percentile(bootstrap_means, 97.5)
            
            ax2.hist(bootstrap_means, bins=30, alpha=0.7, color='lightblue', edgecolor='black')
            ax2.axvline(np.mean(values), color='red', linestyle='-', linewidth=2, label='Original Mean')
            ax2.axvline(ci_lower, color='green', linestyle='--', linewidth=2, label='95% CI')
            ax2.axvline(ci_upper, color='green', linestyle='--', linewidth=2)
            
            ax2.set_xlabel('Bootstrap Sample Means')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Bootstrap Confidence Intervals')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Performance Percentiles über Zeit
        ax3 = axes[0, 2]
        if len(combined_results['eval_step'].unique()) > 1:
            eval_steps = sorted(combined_results['eval_step'].unique())
            percentiles = [10, 25, 50, 75, 90]
            
            percentile_data = {p: [] for p in percentiles}
            
            for eval_step in eval_steps:
                eval_data = combined_results[combined_results['eval_step'] == eval_step]
                if 'discrimination_score_l1' in eval_data.columns:
                    values = eval_data['discrimination_score_l1'].dropna()
                    for p in percentiles:
                        percentile_data[p].append(np.percentile(values, p))
            
            colors = plt.cm.viridis(np.linspace(0, 1, len(percentiles)))
            for i, p in enumerate(percentiles):
                ax3.plot(eval_steps, percentile_data[p], 'o-', 
                        color=colors[i], label=f'{p}th percentile')
            
            ax3.set_xlabel('Evaluation Step')
            ax3.set_ylabel('Discrimination Score L1')
            ax3.set_title('Performance Percentiles Over Time')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Effect Size Analysis
        ax4 = axes[1, 0]
        if len(combined_results['eval_step'].unique()) >= 2:
            eval_steps = sorted(combined_results['eval_step'].unique())
            
            if len(eval_steps) >= 2:
                first_eval = combined_results[combined_results['eval_step'] == eval_steps[0]]
                last_eval = combined_results[combined_results['eval_step'] == eval_steps[-1]]
                
                if 'discrimination_score_l1' in first_eval.columns and 'discrimination_score_l1' in last_eval.columns:
                    first_values = first_eval['discrimination_score_l1'].dropna()
                    last_values = last_eval['discrimination_score_l1'].dropna()
                    
                    # Cohen's d (Effect Size)
                    pooled_std = np.sqrt(((len(first_values) - 1) * first_values.var() + 
                                        (len(last_values) - 1) * last_values.var()) / 
                                    (len(first_values) + len(last_values) - 2))
                    
                    if pooled_std > 0:
                        cohens_d = (last_values.mean() - first_values.mean()) / pooled_std
                        
                        # Visualisierung
                        ax4.bar(['First Eval', 'Last Eval'], 
                            [first_values.mean(), last_values.mean()],
                            color=['lightcoral', 'lightgreen'], alpha=0.7)
                        
                        # Error bars
                        ax4.errorbar(['First Eval', 'Last Eval'], 
                                [first_values.mean(), last_values.mean()],
                                yerr=[first_values.std(), last_values.std()],
                                fmt='none', color='black', capsize=5)
                        
                        ax4.set_ylabel('Discrimination Score L1')
                        ax4.set_title(f'Effect Size Analysis\nCohen\'s d = {cohens_d:.3f}')
                        ax4.grid(True, alpha=0.3)
                        
                        # Effect Size Interpretation
                        if abs(cohens_d) < 0.2:
                            effect_size = "Small"
                        elif abs(cohens_d) < 0.8:
                            effect_size = "Medium"
                        else:
                            effect_size = "Large"
                        
                        ax4.text(0.5, 0.95, f'Effect Size: {effect_size}', 
                                transform=ax4.transAxes, ha='center', 
                                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        # 5. Outlier Analysis mit Z-Score
        ax5 = axes[1, 1]
        if 'discrimination_score_l1' in latest_eval.columns:
            values = latest_eval['discrimination_score_l1'].dropna()
            
            # Z-Scores berechnen
            z_scores = np.abs((values - values.mean()) / values.std())
            outlier_threshold = 2.5
            
            # Scatter plot mit Z-Score Färbung
            colors = ['red' if z > outlier_threshold else 'blue' for z in z_scores]
            ax5.scatter(range(len(values)), values, c=colors, alpha=0.7, s=50)
            
            # Outlier-Linie
            mean_val = values.mean()
            std_val = values.std()
            ax5.axhline(mean_val + outlier_threshold * std_val, 
                    color='red', linestyle='--', alpha=0.7, label='Outlier Threshold')
            ax5.axhline(mean_val - outlier_threshold * std_val, 
                    color='red', linestyle='--', alpha=0.7)
            ax5.axhline(mean_val, color='green', linestyle='-', alpha=0.7, label='Mean')
            
            ax5.set_xlabel('Perturbation Index')
            ax5.set_ylabel('Discrimination Score L1')
            ax5.set_title(f'Outlier Detection (Z-Score > {outlier_threshold})')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
            
            # Anzahl Outliers
            n_outliers = sum(z > outlier_threshold for z in z_scores)
            ax5.text(0.02, 0.98, f'Outliers: {n_outliers}/{len(values)}', 
                    transform=ax5.transAxes, va='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # 6. Metric Stability Analysis
        ax6 = axes[1, 2]
        if len(combined_results['eval_step'].unique()) > 2:
            eval_steps = sorted(combined_results['eval_step'].unique())
            
            stability_metrics = []
            step_labels = []
            
            for i in range(1, len(eval_steps)):
                prev_eval = combined_results[combined_results['eval_step'] == eval_steps[i-1]]
                curr_eval = combined_results[combined_results['eval_step'] == eval_steps[i]]
                
                if 'discrimination_score_l1' in prev_eval.columns and 'discrimination_score_l1' in curr_eval.columns:
                    prev_values = prev_eval['discrimination_score_l1'].dropna()
                    curr_values = curr_eval['discrimination_score_l1'].dropna()
                    
                    # Correlation als Stabilitätsmaß
                    if len(prev_values) > 0 and len(curr_values) > 0:
                        # Gemeinsame Perturbations finden
                        common_perts = set(prev_eval['perturbation']).intersection(set(curr_eval['perturbation']))
                        
                        if len(common_perts) > 5:  # Mindestens 5 gemeinsame Perturbations
                            prev_common = prev_eval[prev_eval['perturbation'].isin(common_perts)].set_index('perturbation')['discrimination_score_l1']
                            curr_common = curr_eval[curr_eval['perturbation'].isin(common_perts)].set_index('perturbation')['discrimination_score_l1']
                            
                            # Sortiere nach Index für korrekte Zuordnung
                            prev_common = prev_common.sort_index()
                            curr_common = curr_common.sort_index()
                            
                            correlation = prev_common.corr(curr_common)
                            stability_metrics.append(correlation)
                            step_labels.append(f'{eval_steps[i-1]}→{eval_steps[i]}')
            
            if stability_metrics:
                bars = ax6.bar(range(len(stability_metrics)), stability_metrics)
                
                # Farb-Kodierung basierend auf Korrelation
                for bar, corr in zip(bars, stability_metrics):
                    if corr > 0.8:
                        bar.set_color('green')
                    elif corr > 0.6:
                        bar.set_color('yellow')
                    else:
                        bar.set_color('red')
                
                ax6.set_xticks(range(len(stability_metrics)))
                ax6.set_xticklabels(step_labels, rotation=45, ha='right')
                ax6.set_ylabel('Correlation')
                ax6.set_title('Metric Stability\n(Inter-evaluation Correlation)')
                ax6.set_ylim(0, 1)
                ax6.grid(True, alpha=0.3)
                
                # Durchschnittliche Stabilität
                mean_stability = np.mean(stability_metrics)
                ax6.axhline(mean_stability, color='blue', linestyle='--', 
                        label=f'Mean: {mean_stability:.3f}')
                ax6.legend()
        
        plt.tight_layout()
        save_path = os.path.join(self.plots_dir, f'statistical_analysis_step_{current_step}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved statistical analysis: {save_path}")



    """def on_train_batch_end(
        self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule, 
        outputs, 
        batch, 
        batch_idx: int
    ) -> None:
        
        Called after each training batch
        old
        
        if trainer.global_step % self.eval_every_n_steps == 0 and trainer.global_step > 0:            
            logger.info(f"Running dynamic evaluation at step {trainer.global_step}")
        
            # Create temporary checkpoint
            temp_checkpoint_path = self._save_temp_checkpoint(trainer)
            
            try:
                # Call run_tx_predict directly
                #metrics = self._call_run_tx_predict(temp_checkpoint_path)
                (results, agg_results) = self._call_run_tx_predict(temp_checkpoint_path)
                
                #print("Buchi after run_tx_predict")
                # Metrics loggen
                #print(f"Buchi results: {results}")
                #print(f"Buchi agg_results: {agg_results}")
        
                #if metrics and self.log_to_wandb:
                if self.log_to_wandb:
                    print("Buchi Yeah metrics log!")
                    #self._log_metrics(trainer, results, agg_results)
                    #self._process_eval_results(trainer, results, trainer.global_step)
                            
            except Exception as e:
                logger.error(f"Error during dynamic evaluation: {e}")
            finally:
                # Cleanup
                #self._cleanup_temp_files(temp_checkpoint_path)
                print("no cleanup necessary!")

    

    def _plot_performance_trends(self, combined_agg_results, current_step):
        """Detailed trend analysis with predictions"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Performance Trend Analysis & Predictions (Step: {current_step})', 
                    fontsize=16, fontweight='bold')
        
        available_metrics = [col for col in ['overlap_at_N', 'mae', 'discrimination_score_l1'] 
                            if col in combined_agg_results.columns]
        
        # 1. Trend with confidence interval
        ax1 = axes[0, 0]
        for metric in available_metrics:
            values = combined_agg_results[metric]
            steps = combined_agg_results['global_step']
            
            # Moving average
            window_size = min(3, len(values))
            if len(values) >= window_size:
                rolling_mean = values.rolling(window=window_size, center=True).mean()
                rolling_std = values.rolling(window=window_size, center=True).std()
                
                ax1.plot(steps, values, 'o-', alpha=0.6, label=f'{metric} (raw)')
                ax1.plot(steps, rolling_mean, '-', linewidth=3, 
                        label=f'{metric} (smoothed)')
                ax1.fill_between(steps, rolling_mean - rolling_std, 
                            rolling_mean + rolling_std, alpha=0.2)
        
        ax1.set_title('Metrics with Confidence Intervals')
        ax1.set_xlabel('Global Step')
        ax1.set_ylabel('Metric Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Improvement rate (derivative)
        ax2 = axes[0, 1]
        for metric in available_metrics:
            values = combined_agg_results[metric]
            steps = combined_agg_results['global_step']
            
            if len(values) > 1:
                # Calculate improvement rate
                improvement_rate = np.gradient(values, steps)
                ax2.plot(steps, improvement_rate, 'o-', label=f'{metric} improvement rate')
        
        ax2.set_title('Improvement Rate (Gradient)')
        ax2.set_xlabel('Global Step')
        ax2.set_ylabel('Improvement Rate')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Relative performance (% improvement from start)
        ax3 = axes[1, 0]
        for metric in available_metrics:
            values = combined_agg_results[metric]
            steps = combined_agg_results['global_step']
            
            if len(values) > 1:
                baseline = values.iloc[0]
                if baseline != 0:
                    relative_improvement = ((values - baseline) / abs(baseline)) * 100
                    ax3.plot(steps, relative_improvement, 'o-', label=f'{metric}')
        
        ax3.set_title('Relative Improvement from Baseline (%)')
        ax3.set_xlabel('Global Step')
        ax3.set_ylabel('Improvement (%)')
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Performance Stability (Rolling Variance)
        ax4 = axes[1, 1]
        for metric in available_metrics:
            values = combined_agg_results[metric]
            steps = combined_agg_results['global_step']
            
            if len(values) >= 3:
                rolling_var = values.rolling(window=3, center=True).var()
                ax4.plot(steps, rolling_var, 'o-', label=f'{metric} variance')
        
        ax4.set_title('Performance Stability (Rolling Variance)')
        ax4.set_xlabel('Global Step')
        ax4.set_ylabel('Variance')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.plots_dir, f'performance_trends_step_{current_step}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved performance trends plot: {save_path}")

    def _plot_perturbation_ranking_analysis(self, combined_results, current_step):
        """Analysis of perturbation rankings over time"""
        
        if combined_results is None or len(combined_results) == 0:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle(f'Perturbation Ranking & Consistency Analysis (Step: {current_step})', 
                    fontsize=16, fontweight='bold')
        
        # 1. Top Perturbations Heatmap over time
        ax1 = axes[0, 0]
        
        # Find the top 10 perturbations for each evaluation
        eval_steps = sorted(combined_results['eval_step'].unique())
        top_perturbations_over_time = {}
        
        for eval_step in eval_steps:
            eval_data = combined_results[combined_results['eval_step'] == eval_step]
            if 'discrimination_score_l1' in eval_data.columns:
                top_10 = eval_data.nlargest(10, 'discrimination_score_l1')['perturbation'].tolist()
                top_perturbations_over_time[eval_step] = top_10
        
        # Collect all unique top perturbations
        all_top_perts = set()
        for perts in top_perturbations_over_time.values():
            all_top_perts.update(perts)
        
        # Create heat map matrix
        if all_top_perts and len(eval_steps) > 1:
            heatmap_data = []
            pert_labels = sorted(list(all_top_perts))
            
            for eval_step in eval_steps:
                row = []
                top_perts = top_perturbations_over_time.get(eval_step, [])
                for pert in pert_labels:
                    if pert in top_perts:
                        rank = top_perts.index(pert) + 1
                        row.append(11 - rank)  # Inverted for better visualization
                    else:
                        row.append(0)
                heatmap_data.append(row)
            
            heatmap_data = np.array(heatmap_data).T
            
            im = ax1.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
            ax1.set_xticks(range(len(eval_steps)))
            ax1.set_xticklabels([f'Eval {i}' for i in eval_steps], rotation=45)
            ax1.set_yticks(range(len(pert_labels)))
            ax1.set_yticklabels(pert_labels, fontsize=8)
            ax1.set_title('Top Perturbations Ranking Over Time\n(Darker = Higher Rank)')
            
            # Colorbar
            cbar = plt.colorbar(im, ax=ax1, shrink=0.8)
            cbar.set_label('Inverted Rank (10=Top, 0=Not in Top 10)')
        
        # 2. Ranking Stability Score
        ax2 = axes[0, 1]
        
        if len(eval_steps) > 1:
            stability_scores = []
            
            for i in range(1, len(eval_steps)):
                prev_eval = eval_steps[i-1]
                curr_eval = eval_steps[i]
                
                prev_data = combined_results[combined_results['eval_step'] == prev_eval]
                curr_data = combined_results[combined_results['eval_step'] == curr_eval]
                
                if 'discrimination_score_l1' in prev_data.columns:
                    prev_top10 = set(prev_data.nlargest(10, 'discrimination_score_l1')['perturbation'])
                    curr_top10 = set(curr_data.nlargest(10, 'discrimination_score_l1')['perturbation'])
                    
                    # Jaccard similarity
                    intersection = len(prev_top10.intersection(curr_top10))
                    union = len(prev_top10.union(curr_top10))
                    stability = intersection / union if union > 0 else 0
                    stability_scores.append(stability)
            
            if stability_scores:
                ax2.plot(eval_steps[1:], stability_scores, 'o-', linewidth=2, markersize=8)
                ax2.set_title('Top-10 Ranking Stability\n(Jaccard Similarity)')
                ax2.set_xlabel('Evaluation Step')
                ax2.set_ylabel('Stability Score')
                ax2.set_ylim(0, 1)
                ax2.grid(True, alpha=0.3)
                
                # Display average stability
                mean_stability = np.mean(stability_scores)
                ax2.axhline(y=mean_stability, color='red', linestyle='--', 
                        label=f'Mean: {mean_stability:.3f}')
                ax2.legend()
        
        # 3. Performance distribution across all perturbations
        ax3 = axes[1, 0]
        
        latest_eval = combined_results[combined_results['eval_step'] == combined_results['eval_step'].max()]
        
        for metric in ['overlap_at_N', 'mae', 'discrimination_score_l1']:
            if metric in latest_eval.columns:
                values = latest_eval[metric].dropna()
                if len(values) > 0:
                    ax3.hist(values, bins=20, alpha=0.6, label=f'{metric}', density=True)
        
        ax3.set_title('Performance Distribution (Latest Evaluation)')
        ax3.set_xlabel('Metric Value')
        ax3.set_ylabel('Density')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Outlier Detection
        ax4 = axes[1, 1]
        
        if 'discrimination_score_l1' in latest_eval.columns:
            values = latest_eval['discrimination_score_l1'].dropna()
            
            # Box plot
            bp = ax4.boxplot(values, vert=True, patch_artist=True)
            bp['boxes'][0].set_facecolor('lightblue')
            
            # Outliers identifycation
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = latest_eval[(latest_eval['discrimination_score_l1'] < lower_bound) | 
                                (latest_eval['discrimination_score_l1'] > upper_bound)]
            
            ax4.set_title(f'Outlier Detection\n({len(outliers)} outliers found)')
            ax4.set_ylabel('Discrimination Score L1')
            
            # show Outlier-Names 
            if len(outliers) > 0 and len(outliers) <= 10:
                outlier_text = '\n'.join(outliers['perturbation'].head(10).tolist())
                ax4.text(1.1, 0.5, f'Outliers:\n{outlier_text}', 
                        transform=ax4.transAxes, fontsize=8, 
                        verticalalignment='center')
        
        plt.tight_layout()
        save_path = os.path.join(self.plots_dir, f'perturbation_ranking_analysis_step_{current_step}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved perturbation ranking analysis: {save_path}")

    def _plot_correlation_and_clustering(self, combined_results, current_step):
        """Korrelations- und Clustering-Analyse der Metriken"""
        
        if combined_results is None or len(combined_results) == 0:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Correlation & Clustering Analysis (Step: {current_step})', 
                    fontsize=16, fontweight='bold')
        
        latest_eval = combined_results[combined_results['eval_step'] == combined_results['eval_step'].max()]
        
        # Numerische Spalten für Analyse
        numeric_cols = ['overlap_at_N', 'mae', 'discrimination_score_l1']
        available_numeric_cols = [col for col in numeric_cols if col in latest_eval.columns]
        
        if len(available_numeric_cols) < 2:
            logger.warning("Need at least 2 numeric columns for correlation analysis")
            return
        
        # 1. Korrelations-Heatmap
        ax1 = axes[0, 0]
        
        corr_data = latest_eval[available_numeric_cols].corr()
        
        # Heatmap erstellen
        im = ax1.imshow(corr_data, cmap='RdBu_r', vmin=-1, vmax=1)
        
        # Ticks und Labels
        ax1.set_xticks(range(len(available_numeric_cols)))
        ax1.set_yticks(range(len(available_numeric_cols)))
        ax1.set_xticklabels([col.replace('_', '\n') for col in available_numeric_cols], rotation=45)
        ax1.set_yticklabels([col.replace('_', '\n') for col in available_numeric_cols])
        
        # Korrelationswerte als Text
        for i in range(len(available_numeric_cols)):
            for j in range(len(available_numeric_cols)):
                text = ax1.text(j, i, f'{corr_data.iloc[i, j]:.3f}',
                            ha="center", va="center", color="black", fontweight='bold')
        
        ax1.set_title('Metric Correlations')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax1, shrink=0.8)
        cbar.set_label('Correlation Coefficient')
        
        # 2. Scatter Plot Matrix
        ax2 = axes[0, 1]
        
        if len(available_numeric_cols) >= 2:
            # Nimm die ersten zwei verfügbaren Metriken
            x_metric = available_numeric_cols[0]
            y_metric = available_numeric_cols[1]
            
            x_vals = latest_eval[x_metric]
            y_vals = latest_eval[y_metric]
            
            scatter = ax2.scatter(x_vals, y_vals, alpha=0.6, s=50)
            
            # Trend-Linie
            if len(x_vals) > 2:
                #print(f"DAN: x_vals: {x_vals}")
                #print(f"DAN: y_vals: {x_vals}")
                #print(f"DAN: type(x_vals): {type(x_vals)}")
                #print(f"DAN: type(y_vals): {type(x_vals)}")
                z = np.polyfit(x_vals, y_vals, 1)
                p = np.poly1d(z)
                ax2.plot(x_vals, p(x_vals), "r--", alpha=0.8, linewidth=2)
                
                # Korrelation anzeigen
                correlation = x_vals.corr(y_vals)
                ax2.text(0.05, 0.95, f'r = {correlation:.3f}', 
                        transform=ax2.transAxes, fontsize=12, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
            
            ax2.set_xlabel(x_metric.replace('_', ' ').title())
            ax2.set_ylabel(y_metric.replace('_', ' ').title())
            ax2.set_title(f'{x_metric} vs {y_metric}')
            ax2.grid(True, alpha=0.3)
        
        # 3. K-Means Clustering (falls genug Daten)
        ax3 = axes[1, 0]
        
        if len(available_numeric_cols) >= 2 and len(latest_eval) >= 6:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            
            # Daten vorbereiten
            cluster_data = latest_eval[available_numeric_cols].dropna()
            
            if len(cluster_data) >= 6:
                # Standardisieren
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(cluster_data)
                
                # K-Means mit 3 Clustern
                kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(scaled_data)
                
                # Plot (erste zwei Dimensionen)
                colors = ['red', 'blue', 'green']
                for i in range(3):
                    mask = clusters == i
                    ax3.scatter(scaled_data[mask, 0], scaled_data[mask, 1], 
                            c=colors[i], label=f'Cluster {i}', alpha=0.7, s=50)
                
                # Cluster-Zentren
                centers = kmeans.cluster_centers_
                ax3.scatter(centers[:, 0], centers[:, 1], 
                        c='black', marker='x', s=200, linewidths=3, label='Centroids')
                
                ax3.set_xlabel(f'{available_numeric_cols[0]} (standardized)')
                ax3.set_ylabel(f'{available_numeric_cols[1]} (standardized)')
                ax3.set_title('K-Means Clustering (k=3)')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
        
        # 4. Performance Quartile Analysis
        ax4 = axes[1, 1]
        
        if 'discrimination_score_l1' in latest_eval.columns:
            # Quartile basierend auf discrimination_score_l1
            quartiles = pd.qcut(latest_eval['discrimination_score_l1'], 
                            q=4, labels=['Q1 (Bottom)', 'Q2', 'Q3', 'Q4 (Top)'])
            
            # Für jede andere Metrik die Quartil-Verteilung zeigen
            other_metrics = [col for col in available_numeric_cols if col != 'discrimination_score_l1']
            
            if other_metrics:
                metric_to_analyze = other_metrics[0]  # Nimm die erste verfügbare
                
                quartile_data = []
                quartile_labels = []
                
                for quartile in ['Q1 (Bottom)', 'Q2', 'Q3', 'Q4 (Top)']:
                    mask = quartiles == quartile
                    if mask.sum() > 0:
                        values = latest_eval.loc[mask, metric_to_analyze].dropna()
                        if len(values) > 0:
                            quartile_data.append(values)
                            quartile_labels.append(quartile)
                
                if quartile_data:
                    bp = ax4.boxplot(quartile_data, labels=quartile_labels, patch_artist=True)
                    
                    # Farben für Boxplots
                    colors = ['lightcoral', 'lightblue', 'lightgreen', 'gold']
                    for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                        patch.set_facecolor(color)
                    
                    ax4.set_title(f'{metric_to_analyze} by Discrimination Score Quartiles')
                    ax4.set_ylabel(metric_to_analyze.replace('_', ' ').title())
                    ax4.tick_params(axis='x', rotation=45)
                    ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.plots_dir, f'correlation_clustering_step_{current_step}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved correlation and clustering analysis: {save_path}")

    def _plot_training_dynamics(self, combined_agg_results, current_step):
        """Analyse der Training-Dynamik und Learning-Curves"""
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f'Training Dynamics & Learning Curves (Step: {current_step})', 
                    fontsize=16, fontweight='bold')
        
        # 1. Learning Rate vs Performance (falls LR-Info verfügbar)
        ax1 = axes[0, 0]
        
        if len(self.eval_metadata) > 0:
            lr_data = []
            steps_data = []
            
            for metadata in self.eval_metadata:
                if metadata.get('learning_rate') is not None:
                    lr_data.append(metadata['learning_rate'])
                    steps_data.append(metadata['global_step'])
            
            if lr_data and len(lr_data) > 1:
                ax1_twin = ax1.twinx()
                
                # Learning Rate
                line1 = ax1.plot(steps_data, lr_data, 'r-o', label='Learning Rate')
                ax1.set_ylabel('Learning Rate', color='r')
                ax1.tick_params(axis='y', labelcolor='r')
                
                # Performance (z.B. discrimination_score_l1)
                if 'discrimination_score_l1' in combined_agg_results.columns:
                    performance = combined_agg_results['discrimination_score_l1']
                    perf_steps = combined_agg_results['global_step']
                    line2 = ax1_twin.plot(perf_steps, performance, 'b-s', label='Performance')
                    ax1_twin.set_ylabel('Discrimination Score L1', color='b')
                    ax1_twin.tick_params(axis='y', labelcolor='b')
                
                ax1.set_xlabel('Global Step')
                ax1.set_title('Learning Rate vs Performance')
                ax1.grid(True, alpha=0.3)
        
        # 2. Training Velocity (Schritte pro Zeit)
        ax2 = axes[0, 1]
        
        if len(self.eval_metadata) > 1:
            timestamps = []
            steps = []
            
            for metadata in self.eval_metadata:
                try:
                    timestamp = datetime.fromisoformat(metadata['timestamp'])
                    timestamps.append(timestamp)
                    steps.append(metadata['global_step'])
                except:
                    continue
            
            if len(timestamps) > 1:
                # Berechne Schritte pro Minute
                velocities = []
                time_points = []
                
                for i in range(1, len(timestamps)):
                    time_diff = (timestamps[i] - timestamps[i-1]).total_seconds() / 60  # Minuten
                    step_diff = steps[i] - steps[i-1]
                    
                    if time_diff > 0:
                        velocity = step_diff / time_diff
                        velocities.append(velocity)
                        time_points.append(steps[i])
                
                if velocities:
                    ax2.plot(time_points, velocities, 'g-o', linewidth=2)
                    ax2.set_xlabel('Global Step')
                    ax2.set_ylabel('Steps per Minute')
                    ax2.set_title('Training Velocity')
                    ax2.grid(True, alpha=0.3)
                    
                    # Durchschnittliche Geschwindigkeit
                    mean_velocity = np.mean(velocities)
                    ax2.axhline(y=mean_velocity, color='red', linestyle='--', 
                            label=f'Mean: {mean_velocity:.1f} steps/min')
                    ax2.legend()
        
        # 3. Epoch vs Global Step Relationship
        ax3 = axes[0, 2]
        
        epochs = combined_agg_results['epoch'] if 'epoch' in combined_agg_results.columns else None
        steps = combined_agg_results['global_step']
        
        if epochs is not None and len(epochs) > 1:
            ax3.plot(epochs, steps, 'purple', marker='o', linewidth=2)
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Global Step')
            ax3.set_title('Training Progress')
            ax3.grid(True, alpha=0.3)
            
            # Steps per Epoch berechnen
            if len(epochs) > 1:
                steps_per_epoch = np.gradient(steps, epochs)
                ax3_twin = ax3.twinx()
                ax3_twin.plot(epochs, steps_per_epoch, 'orange', linestyle='--', 
                            label='Steps/Epoch')
                ax3_twin.set_ylabel('Steps per Epoch', color='orange')
                ax3_twin.tick_params(axis='y', labelcolor='orange')
        
        # 4. Performance Plateau Detection
        ax4 = axes[1, 0]
        
        available_metrics = [col for col in ['discrimination_score_l1', 'mae', 'overlap_at_N'] 
                            if col in combined_agg_results.columns]
        
        for metric in available_metrics:
            values = combined_agg_results[metric]
            steps = combined_agg_results['global_step']
            
            if len(values) >= 5:
                # Gleitender Durchschnitt für Plateau-Erkennung
                window_size = min(5, len(values))
                rolling_mean = values.rolling(window=window_size).mean()
                rolling_std = values.rolling(window=window_size).std()
                
                # Plateau erkennen (geringe Standardabweichung)
                plateau_threshold = rolling_std.quantile(0.3)  # Bottom 30%
                plateau_mask = rolling_std <= plateau_threshold
                
                ax4.plot(steps, values, 'o-', alpha=0.7, label=f'{metric}')
                
                # Plateau-Bereiche markieren
                if plateau_mask.any():
                    plateau_steps = steps[plateau_mask]
                    plateau_values = values[plateau_mask]
                    ax4.scatter(plateau_steps, plateau_values, 
                            s=100, alpha=0.8, marker='s', 
                            label=f'{metric} plateau')
        
        ax4.set_xlabel('Global Step')
        ax4.set_ylabel('Metric Value')
        ax4.set_title('Plateau Detection')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Performance Acceleration/Deceleration
        ax5 = axes[1, 1]
        
        for metric in available_metrics:
            values = combined_agg_results[metric]
            steps = combined_agg_results['global_step']
            
            if len(values) >= 3:
                # Erste und zweite Ableitung
                first_derivative = np.gradient(values, steps)
                second_derivative = np.gradient(first_derivative, steps)
                
                ax5.plot(steps, second_derivative, 'o-', label=f'{metric} acceleration')
        
        ax5.set_xlabel('Global Step')
        ax5.set_ylabel('Acceleration (2nd Derivative)')
        ax5.set_title('Performance Acceleration')
        ax5.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Training Efficiency Score
        ax6 = axes[1, 2]
        
        if 'discrimination_score_l1' in combined_agg_results.columns:
            performance = combined_agg_results['discrimination_score_l1']
            steps = combined_agg_results['global_step']
            
            # Effizienz = Performance / Steps (normalisiert)
            if len(performance) > 1:
                max_performance = performance.max()
                max_steps = steps.max()
                
                efficiency = (performance / max_performance) / (steps / max_steps)
                
                ax6.plot(steps, efficiency, 'o-', color='darkgreen', linewidth=2)
                ax6.set_xlabel('Global Step')
                ax6.set_ylabel('Training Efficiency')
                ax6.set_title('Training Efficiency Score\n(Performance/Steps normalized)')
                ax6.grid(True, alpha=0.3)
                
                # Beste Effizienz markieren
                best_efficiency_idx = efficiency.idxmax()
                best_step = steps.iloc[best_efficiency_idx]
                best_efficiency = efficiency.iloc[best_efficiency_idx]
                
                ax6.scatter(best_step, best_efficiency, color='red', s=100, 
                        label=f'Best: {best_efficiency:.3f} at step {best_step}')
                ax6.legend()
        
        plt.tight_layout()
        save_path = os.path.join(self.plots_dir, f'training_dynamics_step_{current_step}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved training dynamics analysis: {save_path}")

'''
    def _call_run_tx_predict(self, checkpoint_path):
        """Call run_tx_predict with the correct arguments"""
        
        # Create arguments for run_tx_predict
        class MockArgs:
            def __init__(self, output_dir, checkpoint, profile="minimal", predict_only=False):
                self.output_dir = output_dir
                self.checkpoint = os.path.basename(checkpoint)  # only filename
                self.test_time_finetune = 0  # No fine-tuning during training
                self.profile = profile
                self.predict_only = predict_only
        
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
                    predict_only=False
                )
                res_baseline, self.agg_baseline = run_tx_predict(args)
                logger.info(f"Cell-eval baseline results: {self.agg_baseline}")
            # create mock arguments
            args = MockArgs(
                output_dir=self.output_dir,
                checkpoint=expected_checkpoint_path,                
                profile="vcc",
                predict_only=False
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
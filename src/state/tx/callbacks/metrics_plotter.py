import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import json
from datetime import datetime
import logging
from typing import Dict, List, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)

class MetricsPlotter:
    """Dedicated class for creating all evaluation plots"""
    
    def __init__(self, 
                 plots_dir: str, 
                 results_history: List[Dict],
                 agg_results_history: List[Dict],
                 all_metrics_history: List[Dict],
                 #metric_weights: Optional[Dict[str, float]]
                 primary_metric: str = "discrimination_score_l1",
                 style: str = "default",
                 figsize_base: tuple = (12, 8),
                 dpi: int = 300):
        """
        Initialize MetricsPlotter
        
        Args:
            plots_dir: Directory to save plots
            primary_metric: Primary metric for highlighting
            style: Matplotlib style
            figsize_base: Base figure size
            dpi: Resolution for saved plots
        """
        self.plots_dir = Path(plots_dir)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        self.primary_metric = primary_metric
        self.figsize_base = figsize_base
        self.dpi = dpi
        
         # Historien-Speicher für verschiedene Datentypen
        self.agg_results_history = agg_results_history
        self.results_history = results_history
        self.all_metrics_history = all_metrics_history
        #self.metric_weights = metric_weights if metric_weights else {}

        # Plot configuration
        plt.style.use(style)
        sns.set_palette("husl")
        
        # Color schemes
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'success': '#F18F01',
            'warning': '#C73E1D',
            'neutral': '#6C757D',
            'background': '#F8F9FA'
        }
        
        logger.info(f"MetricsPlotter initialized - plots will be saved to: {self.plots_dir}")
    
    def create_all_plots(self, 
                        all_metrics_history: List[Dict],
                        best_scores_per_metric: Dict[str, float],
                        best_steps_per_metric: Dict[str, int],
                        metric_statistics: Dict[str, Dict],
                        current_step: int,
                        composite_score_method: str = "weighted_average",
                        metric_weights: Optional[Dict[str, float]] = None,
                        best_composite_score: Optional[float] = None,
                        best_composite_step: Optional[int] = None):
        """Creates all evaluation plots"""
        self.all_metric_history = all_metrics_history
        if len(all_metrics_history) < 2:
            logger.info("Not enough data for plotting (need at least 2 data points)")
            return
        
        try:
            # Convert to DataFrame
            df_history = pd.DataFrame(all_metrics_history)
            
            logger.info(f"Creating plots for {len(df_history)} data points, {len(df_history.columns)-3} metrics")
            
            # Create all plots
            self._plot_all_metrics_raw(df_history, current_step, best_scores_per_metric, best_steps_per_metric)
            self._plot_metric_correlations(df_history, current_step)
            self._plot_best_scores_dashboard(current_step, best_scores_per_metric, best_steps_per_metric, metric_statistics)
            self._plot_composite_score_evolution(df_history, current_step, composite_score_method, 
                                               metric_weights, best_composite_score, best_composite_step)
            self._plot_metric_distributions(df_history, current_step, metric_statistics, best_scores_per_metric)
            self._plot_training_progress_overview(df_history, current_step, best_scores_per_metric)
            self._plot_metrics_heatmap_timeline(df_history, current_step)
            self._create_evaluation_plots(df_history, current_step)
            #self._create_all_metrics_plot(df_history, current_step)
            #self._create_metric_distributions_plot(df_history, current_step)
            #self._create_training_progress_overview(df_history, current_step)
            #self._create_metrics_heatmap_timeline(df_history, current_step)
            #self._create_metric_improvement_timeline(df_history, current_step)
            #self._create_performance_stability_analysis(df_history, current_step)
            #self._create_metric_ranking_evolution(df_history, current_step)
            #self._create_convergence_analysis(df_history, current_step)
            self._create_metric_volatility_analysis(df_history, current_step)
            self._create_pareto_frontier_analysis(df_history, current_step)
            #self._create_from_baseline_analysis_plots(df_history, current_step)
            
            # HTML-Zusammenfassung
            self.generate_final_report(current_step)
                
            logger.info(f"Successfully created all plots for step {current_step}")
            
        except Exception as e:
            logger.error(f"Error creating plots: {e}")
            import traceback
            traceback.print_exc()
    
    def _plot_all_metrics_raw(self, df_history, current_step, best_scores_per_metric, best_steps_per_metric):
        """Plot aller Metriken mit ihren ursprünglichen from_baseline Werten"""
        
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
                    
                    # Verwende direkt die from_baseline Werte (bereits normalisiert)
                    
                    # Linienstärke für primäre Metrik
                    linewidth = 3 if metric == self.primary_metric else 2
                    alpha = 1.0 if metric == self.primary_metric else 0.8
                    
                    ax.plot(steps, values, marker='o', color=color, 
                           label=metric, linewidth=linewidth, markersize=4, alpha=alpha)
                    
                    # Markiere besten Wert
                    if metric in best_scores_per_metric:
                        best_step = best_steps_per_metric[metric]
                        if best_step in steps.values:
                            best_idx = steps[steps == best_step].index[0]
                            if best_idx < len(values):
                                ax.scatter(best_step, values.iloc[best_idx], 
                                         color=color, s=150, marker='*', 
                                         edgecolors='black', linewidth=2, zorder=5)
            
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7, linewidth=2, label='Baseline (0)')
            ax.set_title(f'All Metrics from Baseline Over Time (Step: {current_step})', 
                        fontsize=16, fontweight='bold')
            ax.set_xlabel('Training Step', fontsize=12)
            ax.set_ylabel('From Baseline Value', fontsize=12)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            
            # Highlight primary metric in legend
            legend = ax.get_legend()
            if legend:
                for text in legend.get_texts():
                    if text.get_text() == self.primary_metric:
                        text.set_weight('bold')
            
            plt.tight_layout()
            
            save_path = self.plots_dir / f'all_metrics_from_baseline_step_{current_step}.png'
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved from_baseline metrics plot: {save_path}")
            
        except Exception as e:
            logger.error(f"Error creating from_baseline metrics plot: {e}")
            import traceback
            traceback.print_exc()
    
    def _plot_metric_correlations(self, df_history, current_step):
        """Plot der Korrelationen zwischen Metriken"""
        
        metrics = [col for col in df_history.columns if not col.startswith('_')]
        
        if len(metrics) < 2:
            logger.warning("Need at least 2 metrics for correlation analysis")
            return
        
        try:
            # Erstelle Korrelationsmatrix
            metric_data = df_history[metrics].dropna()
            
            if len(metric_data) < 2:
                logger.warning("Not enough data points for correlation analysis")
                return
            
            correlation_matrix = metric_data.corr()
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
            
            # Heatmap der Korrelationen
            mask = np.triu(np.ones_like(correlation_matrix.values, dtype=bool), k=1)
            im1 = ax1.imshow(correlation_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1)
            ax1.set_title('Metric Correlations Heatmap', fontsize=14, fontweight='bold')
            ax1.set_xticks(range(len(metrics)))
            ax1.set_yticks(range(len(metrics)))
            ax1.set_xticklabels(metrics, rotation=45, ha='right')
            ax1.set_yticklabels(metrics)
            
            # Füge Korrelationswerte hinzu
            for i in range(len(metrics)):
                for j in range(len(metrics)):
                    corr_val = correlation_matrix.iloc[i, j]
                    text_color = "white" if abs(corr_val) > 0.5 else "black"
                    ax1.text(j, i, f'{corr_val:.2f}',
                           ha="center", va="center", color=text_color, fontweight='bold')
            
            cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
            cbar1.set_label('Correlation Coefficient', fontsize=12)
            
            # Scatter plot der stärksten Korrelationen
            masked_corr = correlation_matrix.values.copy()
            masked_corr[mask] = 0
            np.fill_diagonal(masked_corr, 0)  # Entferne Diagonale
            
            max_corr_idx = np.unravel_index(np.argmax(np.abs(masked_corr)), masked_corr.shape)
            
            if masked_corr[max_corr_idx] != 0:
                metric1 = metrics[max_corr_idx[0]]
                metric2 = metrics[max_corr_idx[1]]
                corr_value = correlation_matrix.iloc[max_corr_idx[0], max_corr_idx[1]]
                
                x_data = metric_data[metric1]
                y_data = metric_data[metric2]
                
                ax2.scatter(x_data, y_data, alpha=0.7, s=60, c=range(len(x_data)), cmap='viridis')
                ax2.set_xlabel(f'{metric1} (from baseline)', fontsize=12)
                ax2.set_ylabel(f'{metric2} (from baseline)', fontsize=12)
                ax2.set_title(f'Strongest Correlation: {metric1} vs {metric2}\n(r = {corr_value:.3f})', 
                             fontsize=14, fontweight='bold')
                ax2.grid(True, alpha=0.3)
                
                # Trendlinie
                try:
                    z = np.polyfit(x_data, y_data, 1)
                    p = np.poly1d(z)
                    x_trend = np.linspace(x_data.min(), x_data.max(), 100)
                    ax2.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label=f'Trend (r²={corr_value**2:.3f})')
                    ax2.legend()
                except:
                    pass
                
                # Colorbar für Zeitverlauf
                cbar2 = plt.colorbar(ax2.collections[0], ax=ax2, shrink=0.8)
                cbar2.set_label('Training Progress', fontsize=12)
            
            plt.tight_layout()
            
            save_path = self.plots_dir / f'metric_correlations_step_{current_step}.png'
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved correlation plot: {save_path}")
            
        except Exception as e:
            logger.error(f"Error creating correlation plot: {e}")
            import traceback
            traceback.print_exc()
    
    def _plot_best_scores_dashboard(self, current_step, best_scores_per_metric, best_steps_per_metric, metric_statistics):
        """Dashboard mit allen besten Scores"""
        
        if not best_scores_per_metric:
            logger.warning("No best scores available for dashboard")
            return
        
        try:
            metrics = list(best_scores_per_metric.keys())
            best_scores = list(best_scores_per_metric.values())
            best_steps = [best_steps_per_metric[m] for m in metrics]
            
            fig = plt.figure(figsize=(20, 16))
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
            
            fig.suptitle(f'Best Scores Dashboard (Current Step: {current_step})', 
                        fontsize=20, fontweight='bold', y=0.98)
            
            # 1. Best Scores Bar Chart
            ax1 = fig.add_subplot(gs[0, 0])
            colors = ['green' if score > 0 else 'red' if score < 0 else 'gray' for score in best_scores]
            bars = ax1.bar(range(len(metrics)), best_scores, color=colors, alpha=0.7, edgecolor='black')
            
            # Highlight primary metric
            if self.primary_metric in metrics:
                primary_idx = metrics.index(self.primary_metric)
                bars[primary_idx].set_edgecolor('gold')
                bars[primary_idx].set_linewidth(3)
            
            ax1.axhline(y=0, color='black', linestyle='-', alpha=0.8)
            ax1.set_title('Best From-Baseline Scores', fontweight='bold')
            ax1.set_xlabel('Metrics')
            ax1.set_ylabel('Best From-Baseline Value')
            ax1.set_xticks(range(len(metrics)))
            ax1.set_xticklabels(metrics, rotation=45, ha='right')
            ax1.grid(True, alpha=0.3)
            
            # Werte auf Balken
            for bar, value in zip(bars, best_scores):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + (0.001 if height >= 0 else -0.001),
                        f'{value:.4f}', ha='center', va='bottom' if height >= 0 else 'top', 
                        fontsize=8, fontweight='bold')
            
            # 2. Steps wo beste Scores erreicht wurden
            ax2 = fig.add_subplot(gs[0, 1])
            bars2 = ax2.bar(range(len(metrics)), best_steps, alpha=0.7, color='skyblue', edgecolor='black')
            
            # Highlight primary metric
            if self.primary_metric in metrics:
                primary_idx = metrics.index(self.primary_metric)
                bars2[primary_idx].set_color('gold')
                bars2[primary_idx].set_edgecolor('black')
                bars2[primary_idx].set_linewidth(2)
            
            ax2.set_title('Steps of Best Scores', fontweight='bold')
            ax2.set_xlabel('Metrics')
            ax2.set_ylabel('Training Step')
            ax2.set_xticks(range(len(metrics)))
            ax2.set_xticklabels(metrics, rotation=45, ha='right')
            ax2.grid(True, alpha=0.3)
            
            # Aktuelle Step-Linie
            ax2.axhline(y=current_step, color='red', linestyle='--', alpha=0.8, 
                       linewidth=2, label=f'Current Step: {current_step}')
            ax2.legend()
            
            # 3. Performance Radar Chart (für from_baseline Werte angepasst)
            ax3 = fig.add_subplot(gs[0, 2], projection='polar')
            if len(best_scores) >= 3:
                # Normalisiere from_baseline Werte für Radar Chart (0 = schlechtester Wert, 1 = bester Wert)
                radar_scores = []
                for score in best_scores:
                    # Einfache Normalisierung: positive Werte sind gut, negative schlecht
                    # Skaliere auf [0, 1] basierend auf dem Wertebereich
                    min_score = min(best_scores)
                    max_score = max(best_scores)
                    if max_score != min_score:
                        normalized = (score - min_score) / (max_score - min_score)
                    else:
                        normalized = 0.5
                    radar_scores.append(normalized)
                
                angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
                angles = np.concatenate((angles, [angles[0]]))  # Complete the circle
                
                radar_scores_circle = radar_scores + [radar_scores[0]]
                
                ax3.plot(angles, radar_scores_circle, 'o-', linewidth=2, color=self.colors['primary'])
                ax3.fill(angles, radar_scores_circle, alpha=0.25, color=self.colors['primary'])
                ax3.set_xticks(angles[:-1])
                ax3.set_xticklabels(metrics)
                ax3.set_ylim(0, 1)
                ax3.set_title('Performance Radar Chart\n(Normalized from Baseline)', 
                             fontweight='bold', pad=20)
                ax3.grid(True)
            
            # 4. Improvement Timeline
            ax4 = fig.add_subplot(gs[1, 0])
            steps_since_improvement = []
            for metric in metrics:
                best_step = best_steps_per_metric[metric]
                steps_since = current_step - best_step
                steps_since_improvement.append(steps_since)
            
            colors = ['green' if steps == 0 else 'yellow' if steps < 1000 else 'red' 
                     for steps in steps_since_improvement]
            
            bars4 = ax4.bar(range(len(metrics)), steps_since_improvement, color=colors, 
                           alpha=0.7, edgecolor='black')
            ax4.set_title('Steps Since Last Improvement', fontweight='bold')
            ax4.set_xlabel('Metrics')
            ax4.set_ylabel('Steps Since Best')
            ax4.set_xticks(range(len(metrics)))
            ax4.set_xticklabels(metrics, rotation=45, ha='right')
            ax4.grid(True, alpha=0.3)
            
            # Werte anzeigen
            for i, steps in enumerate(steps_since_improvement):
                ax4.text(i, steps + max(steps_since_improvement) * 0.01, 
                        f'{steps}', ha='center', va='bottom', fontsize=8, fontweight='bold')
            
            # 5. Metric Statistics Overview
            ax5 = fig.add_subplot(gs[1, 1:])
            if metric_statistics:
                stats_data = []
                for metric in metrics:
                    if metric in metric_statistics:
                        stats = metric_statistics[metric]
                        stats_data.append([
                            metric,
                            f"{stats.get('mean', 0):.4f}",
                            f"{stats.get('std', 0):.4f}",
                            f"{stats.get('min', 0):.4f}",
                            f"{stats.get('max', 0):.4f}",
                            f"{stats.get('count', 0)}"
                        ])
                
                if stats_data:
                    table = ax5.table(cellText=stats_data,
                                    colLabels=['Metric', 'Mean', 'Std', 'Min', 'Max', 'Count'],
                                    cellLoc='center',
                                    loc='center')
                    table.auto_set_font_size(False)
                    table.set_fontsize(9)
                    table.scale(1.2, 1.5)
                    
                    # Highlight primary metric row
                    if self.primary_metric in metrics:
                        primary_idx = metrics.index(self.primary_metric)
                        for j in range(6):
                            table[(primary_idx + 1, j)].set_facecolor('gold')
                            table[(primary_idx + 1, j)].set_text_props(weight='bold')
                    
                    ax5.set_title('Metric Statistics Overview (From Baseline)', fontweight='bold')
                    ax5.axis('off')
            
            # 6. Score Distribution Overview
            ax6 = fig.add_subplot(gs[2, :])
            positive_scores = [s for s in best_scores if s > 0]
            negative_scores = [s for s in best_scores if s < 0]
            zero_scores = [s for s in best_scores if s == 0]
            
            categories = ['Positive\n(Better than Baseline)', 'Zero\n(Same as Baseline)', 'Negative\n(Worse than Baseline)']
            counts = [len(positive_scores), len(zero_scores), len(negative_scores)]
            colors_dist = ['green', 'gray', 'red']
            
            bars6 = ax6.bar(categories, counts, color=colors_dist, alpha=0.7, edgecolor='black')
            ax6.set_title('Score Distribution (From Baseline)', fontweight='bold')
            ax6.set_ylabel('Number of Metrics')
            
            # Werte auf Balken
            for bar, count in zip(bars6, counts):
                height = bar.get_height()
                ax6.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{count}', ha='center', va='bottom', fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            
            save_path = self.plots_dir / f'best_scores_dashboard_step_{current_step}.png'
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved best scores dashboard: {save_path}")
            
        except Exception as e:
            logger.error(f"Error creating best scores dashboard: {e}")
            import traceback
            traceback.print_exc()
    
    def _plot_composite_score_evolution(self, df_history, current_step, composite_score_method, 
                                       metric_weights, best_composite_score, best_composite_step):
        """Plot der Composite Score Evolution"""
        
        if len(df_history) < 2:
            return
        
        try:
            # Berechne Composite Scores für alle historischen Punkte
            composite_scores = []
            steps = []
            
            for _, record in df_history.iterrows():
                score = self._calculate_composite_score(record, metric_weights or {}, composite_score_method)
                composite_scores.append(score)
                steps.append(record['_step'])
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
            
            # 1. Composite Score über Zeit
            ax1.plot(steps, composite_scores, 'b-o', linewidth=3, markersize=8, alpha=0.8, 
                    color=self.colors['primary'])
            ax1.set_title(f'Composite Score Evolution ({composite_score_method})', 
                         fontsize=14, fontweight='bold')
            ax1.set_xlabel('Training Step', fontsize=12)
            ax1.set_ylabel('Composite Score (From Baseline)', fontsize=12)
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.7, label='Baseline')
            
            # Markiere besten Score
            if best_composite_score is not None and best_composite_step is not None:
                ax1.scatter(best_composite_step, best_composite_score, 
                           color='red', s=200, marker='*', 
                           label=f'Best: {best_composite_score:.4f} (Step {best_composite_step})', 
                           zorder=5, edgecolors='black', linewidth=2)
                ax1.legend(fontsize=12)
            
            # Trend-Analyse
            if len(composite_scores) > 5:
                # Gleitender Durchschnitt
                window_size = min(5, len(composite_scores) // 2)
                moving_avg = pd.Series(composite_scores).rolling(window=window_size, center=True).mean()
                ax1.plot(steps, moving_avg, 'r--', alpha=0.7, linewidth=2, 
                        label=f'Moving Average (window={window_size})')
                ax1.legend(fontsize=12)
            
            # 2. Composite Score Komponenten (letzter Wert)
            if not df_history.empty:
                last_record = df_history.iloc[-1]
                component_scores = {}
                
                for col in df_history.columns:
                    if not col.startswith('_'):
                        value = last_record[col]
                        if pd.notna(value) and isinstance(value, (int, float)):
                            # Verwende direkt from_baseline Werte (keine weitere Normalisierung nötig)
                            weight = metric_weights.get(col, 1.0) if metric_weights else 1.0
                            if col == self.primary_metric:
                                weight *= 2.0
                            component_scores[col] = value * weight
                
                if component_scores:
                    metrics = list(component_scores.keys())
                    scores = list(component_scores.values())
                    
                    # Sortiere nach Beitrag
                    sorted_items = sorted(zip(metrics, scores), key=lambda x: x[1], reverse=True)
                    metrics, scores = zip(*sorted_items)
                    
                    colors = plt.cm.viridis(np.linspace(0, 1, len(metrics)))
                    bars = ax2.bar(range(len(metrics)), scores, color=colors, alpha=0.8, edgecolor='black')
                    
                    ax2.set_title(f'Composite Score Components (Step {current_step})', 
                                 fontsize=14, fontweight='bold')
                    ax2.set_xlabel('Metrics', fontsize=12)
                    ax2.set_ylabel('Weighted From-Baseline Score', fontsize=12)
                    ax2.set_xticks(range(len(metrics)))
                    ax2.set_xticklabels(metrics, rotation=45, ha='right')
                    ax2.grid(True, alpha=0.3)
                    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
                    
                    # Werte auf Balken
                    for bar, value in zip(bars, scores):
                        height = bar.get_height()
                        y_pos = height + (max(scores) * 0.01 if height >= 0 else min(scores) * 0.01)
                        ax2.text(bar.get_x() + bar.get_width()/2., y_pos,
                                f'{value:.3f}', ha='center', 
                                va='bottom' if height >= 0 else 'top', 
                                fontsize=10, fontweight='bold')
                    
                    # Markiere primäre Metrik
                    if self.primary_metric in metrics:
                        primary_idx = metrics.index(self.primary_metric)
                        bars[primary_idx].set_edgecolor('red')
                        bars[primary_idx].set_linewidth(3)
            
            plt.tight_layout()
            
            save_path = self.plots_dir / f'composite_score_evolution_step_{current_step}.png'
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved composite score evolution plot: {save_path}")
            
        except Exception as e:
            logger.error(f"Error creating composite score evolution plot: {e}")
            import traceback
            traceback.print_exc()
    
    def _plot_metric_distributions(self, df_history, current_step, metric_statistics, best_scores_per_metric):
        """Plot der Metrik-Verteilungen"""
        
        metrics = [col for col in df_history.columns if not col.startswith('_')]
        
        if not metrics:
            return
        
        try:
            # Bestimme Layout
            n_metrics = len(metrics)
            n_cols = 3
            n_rows = (n_metrics + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
            fig.suptitle(f'Metric Distributions (From Baseline, Step: {current_step})', fontsize=18, fontweight='bold')
            
            if n_rows == 1:
                axes = [axes] if n_cols == 1 else axes
            else:
                axes = axes.flatten()
            
            for i, metric in enumerate(metrics):
                ax = axes[i]
                
                values = df_history[metric].dropna()
                
                if len(values) < 2:
                    ax.text(0.5, 0.5, f'Not enough data\nfor {metric}', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=12)
                    ax.set_title(metric, fontweight='bold')
                    continue
                
                # Histogram mit KDE
                ax.hist(values, bins=min(20, len(values)), alpha=0.7, color='skyblue', 
                       edgecolor='black', density=True, label='Distribution')
                
                # KDE overlay
                try:
                    from scipy import stats
                    kde = stats.gaussian_kde(values)
                    x_range = np.linspace(values.min(), values.max(), 100)
                    ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
                except:
                    pass
                
                # Statistiken
                                # Statistiken
                mean_val = values.mean()
                std_val = values.std()
                current_val = values.iloc[-1] if len(values) > 0 else None
                best_val = best_scores_per_metric.get(metric, None)
                
                # Vertikale Linien für wichtige Werte
                ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, linewidth=2,
                          label=f'Mean: {mean_val:.4f}')
                
                if current_val is not None:
                    ax.axvline(current_val, color='blue', linestyle='-', alpha=0.8, linewidth=2,
                              label=f'Current: {current_val:.4f}')
                
                if best_val is not None:
                    ax.axvline(best_val, color='green', linestyle='-', alpha=0.8, linewidth=3,
                              label=f'Best: {best_val:.4f}')
                
                # Baseline-Linie
                ax.axvline(0, color='gray', linestyle=':', alpha=0.6, linewidth=2, label='Baseline (0)')
                
                ax.set_title(f'{metric}', fontweight='bold')
                ax.set_xlabel('From Baseline Value')
                ax.set_ylabel('Density')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
                
                # Highlight primary metric
                if metric == self.primary_metric:
                    ax.set_facecolor('#FFF8DC')  # Light yellow background
                    for spine in ax.spines.values():
                        spine.set_edgecolor('gold')
                        spine.set_linewidth(2)
            
            # Entferne leere Subplots
            for i in range(len(metrics), len(axes)):
                fig.delaxes(axes[i])
            
            plt.tight_layout()
            
            save_path = self.plots_dir / f'metric_distributions_step_{current_step}.png'
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved metric distributions plot: {save_path}")
            
        except Exception as e:
            logger.error(f"Error creating metric distributions plot: {e}")
            import traceback
            traceback.print_exc()
    
    def _plot_training_progress_overview(self, df_history, current_step, best_scores_per_metric):
        """Übersichts-Plot des gesamten Trainingsfortschritts"""
        
        try:
            fig = plt.figure(figsize=(20, 12))
            gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
            
            fig.suptitle(f'Training Progress Overview (Step: {current_step})', 
                        fontsize=18, fontweight='bold')
            
            metrics = [col for col in df_history.columns if not col.startswith('_')]
            
            # 1. Zeitverlauf aller Metriken (kompakt)
            ax1 = fig.add_subplot(gs[0, :2])
            
            for i, metric in enumerate(metrics):
                values = df_history[metric].dropna()
                steps = df_history.loc[values.index, '_step']
                
                color = plt.cm.tab10(i % 10)
                alpha = 1.0 if metric == self.primary_metric else 0.7
                linewidth = 3 if metric == self.primary_metric else 1.5
                
                ax1.plot(steps, values, color=color, alpha=alpha, linewidth=linewidth, 
                        label=metric, marker='o' if metric == self.primary_metric else None, 
                        markersize=4)
            
            ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Baseline')
            ax1.set_title('All Metrics Over Time (From Baseline)', fontweight='bold')
            ax1.set_xlabel('Training Step')
            ax1.set_ylabel('From Baseline Value')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            ax1.grid(True, alpha=0.3)
            
            # 2. Performance Summary
            ax2 = fig.add_subplot(gs[0, 2])
            
            if best_scores_per_metric:
                positive_metrics = sum(1 for score in best_scores_per_metric.values() if score > 0)
                negative_metrics = sum(1 for score in best_scores_per_metric.values() if score < 0)
                neutral_metrics = sum(1 for score in best_scores_per_metric.values() if score == 0)
                
                labels = ['Better than\nBaseline', 'Worse than\nBaseline', 'Same as\nBaseline']
                sizes = [positive_metrics, negative_metrics, neutral_metrics]
                colors = ['green', 'red', 'gray']
                
                # Entferne leere Kategorien
                non_zero_data = [(label, size, color) for label, size, color in zip(labels, sizes, colors) if size > 0]
                if non_zero_data:
                    labels, sizes, colors = zip(*non_zero_data)
                    
                    wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.0f',
                                                      startangle=90, explode=[0.1 if 'Better' in label else 0 for label in labels])
                    ax2.set_title('Performance Summary', fontweight='bold')
            
            # 3. Trend Analysis
            ax3 = fig.add_subplot(gs[1, 0])
            
            if len(df_history) >= 5:
                # Berechne Trends für jede Metrik
                trend_data = []
                
                for metric in metrics[:10]:  # Limitiere auf 10 Metriken für Übersichtlichkeit
                    values = df_history[metric].dropna()
                    if len(values) >= 3:
                        # Einfacher linearer Trend
                        x = np.arange(len(values))
                        try:
                            slope, _ = np.polyfit(x, values, 1)
                            trend_data.append((metric, slope))
                        except:
                            trend_data.append((metric, 0))
                
                if trend_data:
                    metrics_trend, slopes = zip(*trend_data)
                    colors = ['green' if slope > 0 else 'red' if slope < 0 else 'gray' for slope in slopes]
                    
                    bars = ax3.barh(range(len(metrics_trend)), slopes, color=colors, alpha=0.7)
                    ax3.set_yticks(range(len(metrics_trend)))
                    ax3.set_yticklabels(metrics_trend)
                    ax3.set_xlabel('Trend Slope (From Baseline/Step)')
                    ax3.set_title('Metric Trends', fontweight='bold')
                    ax3.axvline(x=0, color='black', linestyle='--', alpha=0.5)
                    ax3.grid(True, alpha=0.3)
                    
                    # Highlight primary metric
                    if self.primary_metric in metrics_trend:
                        primary_idx = metrics_trend.index(self.primary_metric)
                        bars[primary_idx].set_edgecolor('gold')
                        bars[primary_idx].set_linewidth(2)
            
            # 4. Recent Performance (letzte 5 Evaluationen)
            ax4 = fig.add_subplot(gs[1, 1])
            
            if len(df_history) >= 2:
                recent_data = df_history.tail(min(5, len(df_history)))
                
                # Zeige nur die wichtigsten Metriken
                important_metrics = [self.primary_metric] + [m for m in metrics if m != self.primary_metric][:4]
                important_metrics = [m for m in important_metrics if m in recent_data.columns]
                
                x_pos = np.arange(len(recent_data))
                width = 0.8 / len(important_metrics)
                
                for i, metric in enumerate(important_metrics):
                    values = recent_data[metric]
                    offset = (i - len(important_metrics)/2 + 0.5) * width
                    
                    color = plt.cm.Set3(i)
                    alpha = 1.0 if metric == self.primary_metric else 0.8
                    
                    bars = ax4.bar(x_pos + offset, values, width, label=metric, 
                                  color=color, alpha=alpha, edgecolor='black', linewidth=0.5)
                
                ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                ax4.set_xlabel('Recent Evaluations')
                ax4.set_ylabel('From Baseline Value')
                ax4.set_title('Recent Performance', fontweight='bold')
                ax4.set_xticks(x_pos)
                ax4.set_xticklabels([f"Step {int(step)}" for step in recent_data['_step']], rotation=45)
                ax4.legend(fontsize=8)
                ax4.grid(True, alpha=0.3)
            
            # 5. Improvement Rate Analysis
            ax5 = fig.add_subplot(gs[1, 2])
            
            if len(df_history) >= 3:
                improvement_rates = []
                
                for metric in metrics[:8]:  # Top 8 Metriken
                    values = df_history[metric].dropna()
                    if len(values) >= 3:
                        # Berechne Verbesserungsrate (letzte 3 vs erste 3 Werte)
                        early_mean = values.head(3).mean()
                        recent_mean = values.tail(3).mean()
                        
                        if early_mean != 0:
                            improvement_rate = (recent_mean - early_mean) / abs(early_mean) * 100
                        else:
                            improvement_rate = recent_mean * 100
                        
                        improvement_rates.append((metric, improvement_rate))
                
                if improvement_rates:
                    metrics_imp, rates = zip(*improvement_rates)
                    colors = ['green' if rate > 5 else 'yellow' if rate > -5 else 'red' for rate in rates]
                    
                    bars = ax5.bar(range(len(metrics_imp)), rates, color=colors, alpha=0.7, edgecolor='black')
                    ax5.set_xticks(range(len(metrics_imp)))
                    ax5.set_xticklabels(metrics_imp, rotation=45, ha='right')
                    ax5.set_ylabel('Improvement Rate (%)')
                    ax5.set_title('Early vs Recent Performance', fontweight='bold')
                    ax5.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                    ax5.grid(True, alpha=0.3)
                    
                    # Werte auf Balken
                    for bar, rate in zip(bars, rates):
                        height = bar.get_height()
                        ax5.text(bar.get_x() + bar.get_width()/2., 
                                height + (max(rates) * 0.02 if height >= 0 else min(rates) * 0.02),
                                f'{rate:.1f}%', ha='center', 
                                va='bottom' if height >= 0 else 'top', 
                                fontsize=8, fontweight='bold')
            
            plt.tight_layout()
            
            save_path = self.plots_dir / f'training_progress_overview_step_{current_step}.png'
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved training progress overview: {save_path}")
            
        except Exception as e:
            logger.error(f"Error creating training progress overview: {e}")
            import traceback
            traceback.print_exc()
    
    def _plot_metrics_heatmap_timeline(self, df_history, current_step):
        """Heatmap Timeline aller Metriken"""
        
        metrics = [col for col in df_history.columns if not col.startswith('_')]
        
        if len(metrics) < 2 or len(df_history) < 3:
            logger.warning("Not enough data for heatmap timeline")
            return
        
        try:
            # Erstelle Matrix für Heatmap
            metric_data = df_history[metrics + ['_step']].copy()
            
            # Pivot für bessere Darstellung
            heatmap_data = metric_data[metrics].T  # Metriken als Zeilen, Steps als Spalten
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
            
            # 1. Heatmap der absoluten Werte
            im1 = ax1.imshow(heatmap_data.values, cmap='RdBu_r', aspect='auto')
            ax1.set_title(f'Metrics Heatmap Timeline (From Baseline Values, Step: {current_step})', 
                         fontsize=14, fontweight='bold')
            ax1.set_yticks(range(len(metrics)))
            ax1.set_yticklabels(metrics)
            ax1.set_xlabel('Evaluation Number')
            
            # Colorbar
            cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
            cbar1.set_label('From Baseline Value', fontsize=12)
            
            # Markiere primäre Metrik
            if self.primary_metric in metrics:
                primary_idx = metrics.index(self.primary_metric)
                ax1.axhline(y=primary_idx, color='gold', linewidth=3, alpha=0.8)
            
            # 2. Normalisierte Heatmap (pro Metrik)
            normalized_data = heatmap_data.copy()
            for i, metric in enumerate(metrics):
                values = normalized_data.iloc[i]
                if values.std() != 0:
                    normalized_data.iloc[i] = (values - values.mean()) / values.std()
                else:
                    normalized_data.iloc[i] = 0
            
            im2 = ax2.imshow(normalized_data.values, cmap='RdBu_r', aspect='auto', vmin=-2, vmax=2)
            ax2.set_title('Metrics Heatmap Timeline (Z-Score Normalized)', fontsize=14, fontweight='bold')
            ax2.set_yticks(range(len(metrics)))
            ax2.set_yticklabels(metrics)
            ax2.set_xlabel('Evaluation Number')
            
            # Colorbar
            cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
            cbar2.set_label('Z-Score', fontsize=12)
            
            # Markiere primäre Metrik
            if self.primary_metric in metrics:
                primary_idx = metrics.index(self.primary_metric)
                ax2.axhline(y=primary_idx, color='gold', linewidth=3, alpha=0.8)
            
            # X-Achsen Labels mit Steps
            step_labels = [f"{int(step)}" for step in df_history['_step']]
            n_labels = min(10, len(step_labels))  # Maximal 10 Labels
            step_indices = np.linspace(0, len(step_labels)-1, n_labels, dtype=int)
            
            for ax in [ax1, ax2]:
                ax.set_xticks(step_indices)
                ax.set_xticklabels([step_labels[i] for i in step_indices], rotation=45)
            
            plt.tight_layout()
            
            save_path = self.plots_dir / f'metrics_heatmap_timeline_step_{current_step}.png'
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved metrics heatmap timeline: {save_path}")
            
        except Exception as e:
            logger.error(f"Error creating metrics heatmap timeline: {e}")
            import traceback
            traceback.print_exc()
    
    def _calculate_composite_score(self, record, metric_weights = {}, method = "simple_average"):
        """Berechnet Composite Score aus from_baseline Werten"""
        
        metric_values = {}
        
        for col, value in record.items():
            if not col.startswith('_') and pd.notna(value) and isinstance(value, (int, float)):
                metric_values[col] = value
        
        if not metric_values:
            return 0.0
        
        if method == "weighted_average":
            total_weighted_score = 0.0
            total_weight = 0.0
            
            for metric, value in metric_values.items():
                weight = metric_weights.get(metric, 1.0)
                if metric == self.primary_metric:
                    weight *= 2.0  # Double weight for primary metric
                
                total_weighted_score += value * weight
                total_weight += weight
            
            return total_weighted_score / total_weight if total_weight > 0 else 0.0
        
        elif method == "rank_based":
            # Sortiere Metriken nach Wert und gewichte nach Rang
            sorted_metrics = sorted(metric_values.items(), key=lambda x: x[1], reverse=True)
            
            total_score = 0.0
            for rank, (metric, value) in enumerate(sorted_metrics):
                rank_weight = len(sorted_metrics) - rank
                total_score += value * rank_weight
            
            return total_score / sum(range(1, len(sorted_metrics) + 1))
        
        else:  # simple average
            return sum(metric_values.values()) / len(metric_values)

    def _create_from_baseline_analysis_plots(self, df_history, current_step):
        """
        Erstellt spezielle Plots für from_baseline Metriken mit detaillierter Analyse.
        
        Args:
            df_history (pd.DataFrame): DataFrame mit der Trainingshistorie
            current_step (int): Aktueller Trainingsschritt
        """
        
        if len(df_history) < 2:
            logger.info("Need at least 2 evaluations for from_baseline plots")
            return
        
        try:
            # Identifiziere alle Metriken (außer internen Spalten)
            metrics = [col for col in df_history.columns if not col.startswith('_')]
            
            if not metrics:
                logger.warning("No metrics found for from_baseline analysis")
                return
            
            # Erstelle 2x2 Subplot-Layout für umfassende Analyse [[0]](#__0)
            fig, axes = plt.subplots(2, 2, figsize=(20, 16))
            fig.suptitle(f'📊 From-Baseline Metrics Analysis (Step: {current_step:,})', 
                        fontsize=18, fontweight='bold', y=0.98)
            
            # Plot 1: Zeitverlauf aller Metriken mit Trend-Analyse [[1]](#__1)
            ax1 = axes[0, 0]
            
            # Farbpalette für bessere Unterscheidung
            colors = plt.cm.Set3(np.linspace(0, 1, len(metrics)))
            
            for i, metric in enumerate(metrics):
                if metric in df_history.columns:
                    values = df_history[metric].dropna()
                    if len(values) > 0:
                        steps = df_history.loc[values.index, '_step']
                        
                        # Hauptlinie
                        line = ax1.plot(steps, values, marker='o', label=metric, 
                                    linewidth=2.5, markersize=4, color=colors[i], alpha=0.8)
                        
                        # Trend-Linie hinzufügen
                        if len(values) > 3:
                            z = np.polyfit(range(len(values)), values, 1)
                            trend_line = np.poly1d(z)
                            ax1.plot(steps, trend_line(range(len(values))), 
                                    '--', color=colors[i], alpha=0.5, linewidth=1.5)
                        
                        # Markiere primäre Metrik
                        if metric == self.primary_metric:
                            ax1.plot(steps, values, linewidth=4, alpha=0.3, color=colors[i])
            
            # Baseline-Linie und Styling
            ax1.axhline(y=0, color='black', linestyle='-', alpha=0.7, linewidth=2, label='Baseline')
            ax1.fill_between(ax1.get_xlim(), -0.01, 0.01, alpha=0.1, color='gray', label='±0.01 Zone')
            
            ax1.set_title('🎯 All Metrics vs Baseline Over Time', fontsize=14, fontweight='bold', pad=20)
            ax1.set_xlabel('Training Step', fontsize=12)
            ax1.set_ylabel('From Baseline Value', fontsize=12)
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
            ax1.grid(True, alpha=0.3, linestyle=':')
            ax1.tick_params(axis='x', rotation=45)
            
            # Plot 2: Aktuelle Werte als verbessertes Bar Chart [[2]](#__2)
            ax2 = axes[0, 1]
            
            # Hole aktuelle Werte
            current_row = df_history.iloc[-1]
            metric_names = [k for k in metrics if k in current_row and pd.notna(current_row[k])]
            metric_values = [current_row[k] for k in metric_names]
            
            # Erweiterte Farbkodierung
            colors_bar = []
            for v in metric_values:
                if v > 0.05:
                    colors_bar.append('#2E8B57')  # Dunkelgrün für große Verbesserungen
                elif v > 0.01:
                    colors_bar.append('#90EE90')  # Hellgrün für kleine Verbesserungen
                elif v > -0.01:
                    colors_bar.append('#FFD700')  # Gold für neutrale Werte
                elif v > -0.05:
                    colors_bar.append('#FFA500')  # Orange für kleine Verschlechterungen
                else:
                    colors_bar.append('#DC143C')  # Rot für große Verschlechterungen
            
            bars = ax2.bar(range(len(metric_names)), metric_values, color=colors_bar, 
                        alpha=0.8, edgecolor='black', linewidth=0.5)
            
            # Baseline und Zonen
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=2)
            ax2.axhspan(-0.01, 0.01, alpha=0.1, color='gray', label='Neutral Zone')
            
            ax2.set_title(f'📈 Current From-Baseline Values (Step {current_step:,})', 
                        fontsize=14, fontweight='bold', pad=20)
            ax2.set_xlabel('Metrics', fontsize=12)
            ax2.set_ylabel('From Baseline Value', fontsize=12)
            ax2.set_xticks(range(len(metric_names)))
            ax2.set_xticklabels(metric_names, rotation=45, ha='right', fontsize=10)
            ax2.grid(True, alpha=0.3, axis='y', linestyle=':')
            
            # Werte auf Balken mit verbessertem Styling
            for i, (bar, value, metric) in enumerate(zip(bars, metric_values, metric_names)):
                height = bar.get_height()
                offset = 0.005 if height >= 0 else -0.005
                
                # Hervorhebung für primäre Metrik
                if metric == self.primary_metric:
                    bar.set_edgecolor('red')
                    bar.set_linewidth(3)
                    ax2.text(bar.get_x() + bar.get_width()/2., height + offset,
                            f'★ {value:.4f}', ha='center', va='bottom' if height >= 0 else 'top',
                            fontweight='bold', fontsize=10, color='red')
                else:
                    ax2.text(bar.get_x() + bar.get_width()/2., height + offset,
                            f'{value:.4f}', ha='center', va='bottom' if height >= 0 else 'top',
                            fontsize=9)
            
            # Plot 3: Erweiterte Improvement Heatmap mit Zeitfenster-Analyse [[3]](#__3)
            ax3 = axes[1, 0]
            
            # Dynamisches Zeitfenster basierend auf verfügbaren Daten
            window_size = min(15, len(df_history))
            recent_history = df_history.tail(window_size)
            
            # Erstelle Heatmap-Matrix
            heatmap_data = []
            step_labels = []
            
            for _, row in recent_history.iterrows():
                step_labels.append(f"{int(row['_step']):,}")
                row_data = [row.get(metric, np.nan) for metric in metrics]
                heatmap_data.append(row_data)
            
            heatmap_data = np.array(heatmap_data).T  # Transponieren für bessere Darstellung
            
            # Erweiterte Colormap mit besserer Skalierung
            vmin, vmax = np.nanpercentile(heatmap_data, [5, 95])
            vmax = max(abs(vmin), abs(vmax), 0.01)  # Symmetrische Skalierung
            
            im = ax3.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', 
                        vmin=-vmax, vmax=vmax, interpolation='nearest')
            
            ax3.set_title(f'🔥 From-Baseline Heatmap (Last {window_size} Steps)', 
                        fontsize=14, fontweight='bold', pad=20)
            ax3.set_xlabel('Training Steps', fontsize=12)
            ax3.set_ylabel('Metrics', fontsize=12)
            
            # Achsen-Labels
            ax3.set_xticks(range(len(step_labels)))
            ax3.set_xticklabels(step_labels, rotation=45, ha='right', fontsize=9)
            ax3.set_yticks(range(len(metrics)))
            ax3.set_yticklabels(metrics, fontsize=10)
            
            # Primäre Metrik hervorheben
            if self.primary_metric in metrics:
                primary_idx = metrics.index(self.primary_metric)
                ax3.axhline(y=primary_idx, color='red', linewidth=3, alpha=0.7)
            
            # Verbesserte Colorbar
            cbar = plt.colorbar(im, ax=ax3, shrink=0.8, aspect=20)
            cbar.set_label('From Baseline Value', fontsize=11)
            cbar.ax.tick_params(labelsize=9)
            
            # Werte in Heatmap anzeigen (nur für kleine Matrizen)
            if len(metrics) <= 8 and len(step_labels) <= 10:
                for i in range(len(metrics)):
                    for j in range(len(step_labels)):
                        if not np.isnan(heatmap_data[i, j]):
                            text_color = 'white' if abs(heatmap_data[i, j]) > vmax*0.5 else 'black'
                            ax3.text(j, i, f'{heatmap_data[i, j]:.3f}', 
                                ha='center', va='center', color=text_color, fontsize=8)
            
            # Plot 4: Erweiterte Composite Score Evolution mit Statistiken
            ax4 = axes[1, 1]
            
            if len(df_history) > 1:
                # Berechne Composite Scores für alle Schritte
                composite_scores = []
                steps = []
                
                for _, row in df_history.iterrows():
                    # Konvertiere Row zu Dict für _calculate_composite_score
                    record = row.to_dict()
                    score = self._calculate_composite_score(record)
                    composite_scores.append(score)
                    steps.append(row['_step'])
                
                # Hauptlinie mit Markern
                line = ax4.plot(steps, composite_scores, 'b-o', linewidth=3, markersize=6, 
                            alpha=0.8, label='Composite Score')
                
                # Gleitender Durchschnitt
                if len(composite_scores) > 5:
                    window = min(5, len(composite_scores)//3)
                    rolling_mean = pd.Series(composite_scores).rolling(window=window, center=True).mean()
                    ax4.plot(steps, rolling_mean, 'r--', linewidth=2, alpha=0.7, 
                            label=f'Rolling Mean (w={window})')
                
                # Statistiken und Markierungen
                best_score = max(composite_scores)
                worst_score = min(composite_scores)
                current_score = composite_scores[-1]
                
                # Markiere besten Score
                best_idx = composite_scores.index(best_score)
                ax4.scatter(steps[best_idx], best_score, color='green', s=150, 
                        marker='*', label=f'Best: {best_score:.4f}', zorder=5, edgecolor='black')
                
                # Markiere schlechtesten Score
                worst_idx = composite_scores.index(worst_score)
                ax4.scatter(steps[worst_idx], worst_score, color='red', s=100, 
                        marker='v', label=f'Worst: {worst_score:.4f}', zorder=5, edgecolor='black')
                
                # Aktueller Score
                ax4.scatter(steps[-1], current_score, color='blue', s=120, 
                        marker='D', label=f'Current: {current_score:.4f}', zorder=5, edgecolor='black')
                
                # Trend-Analyse
                if len(composite_scores) > 3:
                    recent_trend = np.polyfit(range(len(composite_scores[-10:])), composite_scores[-10:], 1)[0]
                    trend_text = "📈 Improving" if recent_trend > 0.001 else "📉 Declining" if recent_trend < -0.001 else "➡️ Stable"
                    ax4.text(0.02, 0.98, f'Recent Trend: {trend_text}', transform=ax4.transAxes, 
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                
                ax4.set_title('🎯 Composite Score Evolution & Analysis', fontsize=14, fontweight='bold', pad=20)
                ax4.set_xlabel('Training Step', fontsize=12)
                ax4.set_ylabel('Composite Score', fontsize=12)
                ax4.grid(True, alpha=0.3, linestyle=':')
                ax4.legend(loc='lower right', fontsize=10)
                ax4.tick_params(axis='x', rotation=45)
                
                # Y-Achse Formatierung
                ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.3f}'))
            
            else:
                ax4.text(0.5, 0.5, 'Insufficient data\nfor composite score analysis', 
                        ha='center', va='center', transform=ax4.transAxes, fontsize=12)
                ax4.set_title('🎯 Composite Score Evolution', fontsize=14, fontweight='bold')
            
            # Finales Layout und Speichern
            plt.tight_layout(rect=[0, 0.03, 1, 0.96])
            
            save_path = self.plots_dir / f'from_baseline_analysis_step_{current_step}.png'
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
            plt.close()
            
            logger.info(f"Saved from-baseline analysis plot: {save_path}")
            
            # Zusätzlicher detaillierter Einzelmetrik-Plot
            self._create_detailed_metric_analysis(df_history, current_step, metrics)
            
        except Exception as e:
            logger.error(f"Error creating from-baseline analysis plots: {e}")
            import traceback
            traceback.print_exc()

    def _create_detailed_metric_analysis(self, df_history, current_step, metrics):
        """
        Erstellt detaillierte Einzelmetrik-Analyse mit erweiterten Statistiken.
        
        Args:
            df_history (pd.DataFrame): DataFrame mit der Trainingshistorie
            current_step (int): Aktueller Trainingsschritt
            metrics (list): Liste der zu analysierenden Metriken
        """
        try:
            # Berechne Anzahl der benötigten Subplots
            n_metrics = len(metrics)
            if n_metrics == 0:
                return
            
            # Dynamisches Layout basierend auf Anzahl der Metriken
            if n_metrics <= 4:
                rows, cols = 2, 2
            elif n_metrics <= 6:
                rows, cols = 2, 3
            elif n_metrics <= 9:
                rows, cols = 3, 3
            else:
                rows, cols = 4, 3
            
            fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows))
            fig.suptitle(f'📊 Detailed Individual Metric Analysis (Step: {current_step:,})', 
                        fontsize=16, fontweight='bold')
            
            # Flache Axes für einfachere Iteration
            if n_metrics == 1:
                axes = [axes]
            elif rows == 1 or cols == 1:
                axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
            else:
                axes = axes.flatten()
            
            for i, metric in enumerate(metrics):
                if i >= len(axes):
                    break
                    
                ax = axes[i]
                
                if metric in df_history.columns:
                    values = df_history[metric].dropna()
                    steps = df_history.loc[values.index, '_step']
                    
                    if len(values) > 0:
                        # Hauptplot
                        color = 'red' if metric == self.primary_metric else 'blue'
                        linewidth = 3 if metric == self.primary_metric else 2
                        
                        ax.plot(steps, values, 'o-', color=color, linewidth=linewidth, 
                            markersize=4, alpha=0.8, label=metric)
                        
                        # Statistiken berechnen
                        mean_val = values.mean()
                        std_val = values.std()
                        current_val = values.iloc[-1]
                        best_val = values.max()
                        worst_val = values.min()
                        
                        # Baseline und Statistik-Linien
                        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
                        ax.axhline(y=mean_val, color='green', linestyle='--', alpha=0.7, 
                                label=f'Mean: {mean_val:.4f}')
                        
                        # Konfidenzintervall
                        ax.fill_between(steps, mean_val - std_val, mean_val + std_val, 
                                    alpha=0.2, color='green', label=f'±1σ: {std_val:.4f}')
                        
                        # Markiere beste und aktuelle Werte
                        best_idx = values.idxmax()
                        best_step = df_history.loc[best_idx, '_step']
                        ax.scatter(best_step, best_val, color='gold', s=100, marker='*', 
                                label=f'Best: {best_val:.4f}', zorder=5, edgecolor='black')
                        
                        ax.scatter(steps.iloc[-1], current_val, color='red', s=80, marker='D', 
                                label=f'Current: {current_val:.4f}', zorder=5, edgecolor='black')
                        
                        # Titel mit primärer Metrik Kennzeichnung
                        title = f"{'🎯 ' if metric == self.primary_metric else ''}{metric}"
                        ax.set_title(title, fontsize=12, fontweight='bold')
                        ax.set_xlabel('Training Step')
                        ax.set_ylabel('From Baseline Value')
                        ax.legend(fontsize=8, loc='best')
                        ax.grid(True, alpha=0.3)
                        
                        # Trend-Pfeil
                        if len(values) > 1:
                            trend = values.iloc[-1] - values.iloc[-2]
                            arrow_color = 'green' if trend > 0 else 'red' if trend < 0 else 'gray'
                            arrow_symbol = '↗' if trend > 0 else '↘' if trend < 0 else '→'
                            ax.text(0.02, 0.98, f'{arrow_symbol} {trend:+.4f}', 
                                transform=ax.transAxes, fontsize=10, color=arrow_color,
                                verticalalignment='top', fontweight='bold')
                    
                    else:
                        ax.text(0.5, 0.5, f'No data for {metric}', ha='center', va='center', 
                            transform=ax.transAxes)
                        ax.set_title(metric)
                
                else:
                    ax.text(0.5, 0.5, f'Metric {metric} not found', ha='center', va='center', 
                        transform=ax.transAxes)
                    ax.set_title(metric)
            
            # Verstecke überschüssige Subplots
            for i in range(n_metrics, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.96])
            
            save_path = self.plots_dir / f'detailed_metric_analysis_step_{current_step}.png'
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
            plt.close()
            
            logger.info(f"Saved detailed metric analysis plot: {save_path}")
            
        except Exception as e:
            logger.error(f"Error creating detailed metric analysis: {e}")
            import traceback
            traceback.print_exc()

    def _create_evaluation_plots(self, df_history, current_step):
        """Erstellt robuste Plots basierend auf verfügbaren Daten mit umfassender Analyse"""
        if len(self.agg_results_history) < 2:
            logger.info("Need at least 2 evaluations for meaningful plots")
            return
        
        try:
            # Kombinierte DataFrames erstellen für statistische Analyse [[1]](#__1)
            combined_agg_results = pd.concat(self.agg_results_history, ignore_index=True)
            combined_results = None
            
            if self.results_history:
                combined_results = pd.concat(self.results_history, ignore_index=True)
            
            logger.info(f"Creating comprehensive evaluation plots for step {current_step}")
            #logger.info(f"Available agg_results columns: {list(combined_agg_results.columns)}")
            #if combined_results is not None:
            #    logger.info(f"Available results columns: {list(combined_results.columns)}")
            
            # === KERN-ANALYSEN === [[2]](#__2)
            # 1. Basis-Metriken über Zeit
            self._plot_aggregated_metrics_over_time(combined_results, current_step)
            
            # 2. Detaillierte Metrik-Analyse
            self._plot_current_metrics_analysis(combined_results, current_step)
            
            # 3. Performance-Trends mit Vorhersagen
            self._plot_performance_trends(combined_results, current_step)
            
            # 4. Training-Dynamik und Learning-Curves
            self._plot_training_dynamics(current_step)
            
            # === PERTURBATION-ANALYSEN === [[3]](#__3)
            if combined_results is not None:
                # 5. Perturbation-Ranking und Konsistenz
                self._plot_perturbation_ranking_analysis(combined_results, current_step)
                
                # 6. Korrelations- und Clustering-Analyse
                self._plot_correlation_and_clustering(combined_results, current_step)
                
                # 7. Perturbation Deep-Dive
                self._plot_perturbation_deep_dive(combined_results, current_step)
                
                # 8. Statistische Analyse
                self._plot_statistical_analysis(combined_results, current_step)
            
            # === ZUSÄTZLICHE SPEZIAL-PLOTS ===
            # 10. Performance Heatmap über Zeit (falls genug Daten)
            if combined_results is not None and len(combined_results['eval_step'].unique()) > 3:
                self._plot_performance_heatmap_over_time(combined_results, current_step)
            
            # 11. Model Comparison (falls mehrere Modelle)
            if hasattr(self, 'model_comparison_data') and self.model_comparison_data:
                self._plot_model_comparison(current_step)
            
            # From-baseline spezifische Plots
            self._create_from_baseline_analysis_plots(df_history, current_step)
            
            # Multi-Metrik Analyse Plots
            if len(self.all_metrics_history) >= 2:
                self._create_multi_metric_analysis_plots(current_step)

            logger.info(f"Successfully created comprehensive evaluation plots for step {current_step}")
            logger.info(f"Plots saved in: {self.plots_dir}")
            
            # Plot-Zusammenfassung erstellen
            self._create_plot_summary(current_step)
            
        except Exception as e:
            logger.error(f"Error creating evaluation plots: {e}")
            import traceback
            traceback.print_exc()

    def _plot_performance_heatmap_over_time(self, combined_results, current_step):
        """Performance Heatmap über Zeit für Top Perturbations mit verbesserter Visualisierung"""
        
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Top 20 Perturbations aus der letzten Evaluation
        latest_eval = combined_results[combined_results['eval_step'] == combined_results['eval_step'].max()]
        
        if 'discrimination_score_l1' in latest_eval.columns:
            top_20_perts = latest_eval.nlargest(20, 'discrimination_score_l1')['perturbation'].tolist()
            
            eval_steps = sorted(combined_results['eval_step'].unique())
            
            # Heatmap-Daten erstellen mit verbesserter Datenbehandlung [[0]](#__0)
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
            
            # Erweiterte Heatmap mit besserer Farbskalierung
            vmin, vmax = np.nanpercentile(heatmap_data, [5, 95])
            im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', 
                          interpolation='nearest', vmin=vmin, vmax=vmax)
            
            # Achsen-Labels mit verbesserter Formatierung
            ax.set_xticks(range(len(eval_steps)))
            ax.set_xticklabels([f'Step {step:,}' for step in eval_steps], rotation=45, ha='right')
            ax.set_yticks(range(len(top_20_perts)))
            ax.set_yticklabels([pert[:40] + '...' if len(pert) > 40 else pert 
                              for pert in top_20_perts], fontsize=8)
            
            ax.set_xlabel('Evaluation Steps', fontweight='bold')
            ax.set_ylabel('Top Perturbations', fontweight='bold')
            ax.set_title(f'🔥 Performance Heatmap Over Time (Step: {current_step:,})', 
                        fontsize=14, fontweight='bold', pad=20)
            
            # Verbesserte Colorbar mit Beschriftung
            cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=20)
            cbar.set_label('Discrimination Score L1', fontsize=11, fontweight='bold')
            cbar.ax.tick_params(labelsize=9)
            
            # Statistiken hinzufügen
            ax.text(0.02, 0.98, f'Data Range: {vmin:.3f} - {vmax:.3f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            save_path = os.path.join(self.plots_dir, f'performance_heatmap_step_{current_step}.png')
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved performance heatmap: {save_path}")

    def _plot_aggregated_metrics_over_time(self, combined_agg_results, current_step):
        """Plot der aggregierten Metriken über die Zeit mit erweiterten Trend-Analysen"""
        
        # Identifiziere verfügbare Metriken für statistische Visualisierung [[1]](#__1)
        available_metrics = []
        possible_metrics = ['overlap_at_N', 'mae', 'discrimination_score_l1', 'pearson_delta', 'mse']
        
        for metric in possible_metrics:
            if metric in combined_agg_results.columns:
                available_metrics.append(metric)
        
        if not available_metrics:
            logger.warning("No metrics available for plotting")
            return
        
        # Dynamische Plot-Größe basierend auf verfügbaren Metriken
        n_metrics = len(available_metrics)
        n_cols = min(2, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 5*n_rows))
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(f'📊 Available Metrics Over Training Steps (Current: {current_step:,})', 
                    fontsize=16, fontweight='bold')
        
        for idx, metric in enumerate(available_metrics):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            # Erweiterte Visualisierung mit Seaborn-Integration [[2]](#__2)
            steps = combined_agg_results['global_step']
            values = combined_agg_results[metric]
            
            # Hauptlinie mit verbessertem Styling
            line = ax.plot(steps, values, 'o-', linewidth=3, markersize=8, 
                          alpha=0.8, label=metric)
            
            # Trend-Linie mit Konfidenzintervall
            if len(combined_agg_results) > 2:
                try:
                    # Polynomiale Regression für bessere Trend-Erkennung
                    z = np.polyfit(steps, values, min(2, len(steps)-1))
                    p = np.poly1d(z)
                    trend_line = p(steps)
                    
                    ax.plot(steps, trend_line, "--", alpha=0.7, color='red', 
                           linewidth=2, label='Trend')
                    
                    # Konfidenzbereich (vereinfacht)
                    residuals = values - trend_line
                    std_residuals = np.std(residuals)
                    ax.fill_between(steps, trend_line - std_residuals, 
                                   trend_line + std_residuals, 
                                   alpha=0.2, color='red', label='±1σ')
                except:
                    pass
            
            # Statistiken und Annotationen
            current_value = values.iloc[-1]
            best_value = values.max() if metric != 'mae' else values.min()
            
            # Markiere aktuellen und besten Wert
            ax.scatter(steps.iloc[-1], current_value, color='blue', s=150, 
                      marker='D', zorder=5, edgecolor='black', linewidth=2,
                      label=f'Current: {current_value:.4f}')
            
            best_idx = values.idxmax() if metric != 'mae' else values.idxmin()
            ax.scatter(steps.iloc[best_idx], best_value, color='gold', s=150, 
                      marker='*', zorder=5, edgecolor='black', linewidth=2,
                      label=f'Best: {best_value:.4f}')
            
            ax.set_title(f'{metric.replace("_", " ").title()}', fontweight='bold', fontsize=12)
            ax.set_xlabel('Global Training Step', fontweight='bold')
            ax.set_ylabel(metric.replace("_", " ").title(), fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle=':')
            ax.legend(loc='best', fontsize=9)
            
            # Verbesserungen-Indikator
            if len(values) > 1:
                recent_change = values.iloc[-1] - values.iloc[-2]
                change_pct = (recent_change / abs(values.iloc[-2])) * 100 if values.iloc[-2] != 0 else 0
                
                # Farbkodierung basierend auf Metrik-Typ
                is_improvement = (recent_change > 0) if metric != 'mae' else (recent_change < 0)
                color = 'green' if is_improvement else 'red'
                symbol = '↗' if is_improvement else '↘'
                
                ax.text(0.02, 0.98, f'{symbol} {change_pct:+.1f}%', 
                       transform=ax.transAxes, fontsize=12, color=color,
                       verticalalignment='top', fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Verstecke leere Subplots
        if n_metrics < n_rows * n_cols:
            for idx in range(n_metrics, n_rows * n_cols):
                row = idx // n_cols
                col = idx % n_cols
                ax = axes[row, col] if n_rows > 1 else axes[col]
                ax.set_visible(False)
        
        plt.tight_layout()
        
        save_path = os.path.join(self.plots_dir, f'available_metrics_step_{current_step}.png')
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved metrics plot with {n_metrics} available metrics: {save_path}")

    def _plot_current_metrics_analysis(self, combined_agg_results, current_step):
        """Spezialisierte Analyse für overlap_at_N, mae, discrimination_score_l1 mit erweiterten Statistiken"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'📈 Detailed Metrics Analysis (Step: {current_step:,})', 
                    fontsize=16, fontweight='bold')
        
        # 1. Alle drei Metriken normalisiert in einem Plot mit verbesserter Skalierung [[3]](#__3)
        ax1 = axes[0, 0]
        
        # Normalisiere Daten für bessere Vergleichbarkeit
        metrics_to_normalize = ['overlap_at_N', 'mae', 'discrimination_score_l1']
        available_metrics = [m for m in metrics_to_normalize if m in combined_agg_results.columns]
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, metric in enumerate(available_metrics):
            values = combined_agg_results[metric]
            steps = combined_agg_results['global_step']
            
            # Robuste Normalisierung
            if values.max() != values.min():
                normalized_values = (values - values.min()) / (values.max() - values.min())
            else:
                normalized_values = values
            
            line = ax1.plot(steps, normalized_values, marker='o', 
                           label=f'{metric} (normalized)', linewidth=2.5, 
                           color=colors[i % len(colors)], markersize=6)
            
            # Gleitender Durchschnitt für Trend-Glättung
            if len(normalized_values) > 3:
                window = min(3, len(normalized_values)//2)
                rolling_mean = pd.Series(normalized_values).rolling(window=window, center=True).mean()
                ax1.plot(steps, rolling_mean, '--', alpha=0.7, color=colors[i % len(colors)], 
                        linewidth=1.5, label=f'{metric} trend')
        
        ax1.set_title('🎯 Normalized Metrics Comparison', fontweight='bold')
        ax1.set_xlabel('Global Training Step', fontweight='bold')
        ax1.set_ylabel('Normalized Value [0-1]', fontweight='bold')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax1.grid(True, alpha=0.3, linestyle=':')
        ax1.set_ylim(-0.1, 1.1)
        
        # 2. MAE detaillierte Analyse mit Statistiken
        ax2 = axes[0, 1]
        if 'mae' in combined_agg_results.columns:
            mae_values = combined_agg_results['mae']
            steps = combined_agg_results['global_step']
            
            # Hauptlinie mit verbessertem Styling
            ax2.plot(steps, mae_values, 'b-o', linewidth=3, markersize=8, alpha=0.8)
            
            # Statistik-Linien
            mean_mae = mae_values.mean()
            median_mae = mae_values.median()
            
            ax2.axhline(mean_mae, color='red', linestyle='--', alpha=0.7, 
                       linewidth=2, label=f'Mean: {mean_mae:.4f}')
            ax2.axhline(median_mae, color='green', linestyle='--', alpha=0.7, 
                       linewidth=2, label=f'Median: {median_mae:.4f}')
            
            # Konfidenzbereich
            std_mae = mae_values.std()
            ax2.fill_between(steps, mean_mae - std_mae, mean_mae + std_mae, 
                           alpha=0.2, color='gray', label=f'±1σ: {std_mae:.4f}')
            
            ax2.set_title('📉 MAE Progression & Statistics', fontweight='bold')
            ax2.set_xlabel('Global Training Step', fontweight='bold')
            ax2.set_ylabel('Mean Absolute Error', fontweight='bold')
            ax2.grid(True, alpha=0.3, linestyle=':')
            
            # Markiere beste und schlechteste MAE
            best_idx = mae_values.idxmin()
            worst_idx = mae_values.idxmax()
            
            ax2.scatter(steps.iloc[best_idx], mae_values.iloc[best_idx], 
                       color='green', s=150, marker='*', zorder=5, edgecolor='black',
                       label=f'Best: {mae_values.iloc[best_idx]:.4f}')
            ax2.scatter(steps.iloc[worst_idx], mae_values.iloc[worst_idx], 
                       color='red', s=150, marker='v', zorder=5, edgecolor='black',
                       label=f'Worst: {mae_values.iloc[worst_idx]:.4f}')
            
            ax2.legend(loc='best', fontsize=9)
        
        # 3. Discrimination Score Analyse mit Trend-Prognose
        ax3 = axes[1, 0]
        if 'discrimination_score_l1' in combined_agg_results.columns:
            disc_values = combined_agg_results['discrimination_score_l1']
            steps = combined_agg_results['global_step']
            
            # Hauptlinie
            ax3.plot(steps, disc_values, 'g-o', linewidth=3, markersize=8, alpha=0.8)
            
            # Trend-Prognose für nächste Schritte
            if len(disc_values) >= 3:
                # Lineare Extrapolation
                z = np.polyfit(range(len(disc_values)), disc_values, 1)
                p = np.poly1d(z)
                
                # Prognostiziere 3 weitere Schritte
                future_indices = range(len(disc_values), len(disc_values) + 3)
                future_steps = [steps.iloc[-1] + (i+1) * (steps.iloc[-1] - steps.iloc[-2]) 
                               for i in range(3)]
                future_values = [p(i) for i in future_indices]
                
                ax3.plot(future_steps, future_values, 'g--', alpha=0.6, 
                        linewidth=2, marker='s', markersize=6, label='Trend Projection')
            
            # Statistiken
            mean_score = disc_values.mean()
            ax3.axhline(mean_score, color='orange', linestyle='--', alpha=0.7,
                       linewidth=2, label=f'Mean: {mean_score:.3f}')
            
            # Performance-Zonen
            if disc_values.std() > 0:
                high_perf = mean_score + disc_values.std()
                low_perf = mean_score - disc_values.std()
                
                ax3.axhspan(high_perf, disc_values.max(), alpha=0.1, color='green', 
                           label='High Performance Zone')
                ax3.axhspan(disc_values.min(), low_perf, alpha=0.1, color='red', 
                           label='Low Performance Zone')
            
            ax3.set_title('🎯 Discrimination Score L1 & Projection', fontweight='bold')
            ax3.set_xlabel('Global Training Step', fontweight='bold')
            ax3.set_ylabel('Discrimination Score L1', fontweight='bold')
            ax3.grid(True, alpha=0.3, linestyle=':')
            ax3.legend(loc='best', fontsize=9)
        
        # 4. Overlap at N Analyse mit Distribution
        ax4 = axes[1, 1]
        if 'overlap_at_N' in combined_agg_results.columns:
            overlap_values = combined_agg_results['overlap_at_N']
            steps = combined_agg_results['global_step']
            
            # Hauptlinie
            ax4.plot(steps, overlap_values, 'r-o', linewidth=3, markersize=8, alpha=0.8)
            
            # Inset für Distribution
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes
            inset_ax = inset_axes(ax4, width="35%", height="35%", loc='upper right')
            
            # Histogram der Overlap-Werte
            inset_ax.hist(overlap_values, bins=min(10, len(overlap_values)), 
                         alpha=0.7, color='red', edgecolor='black')
            inset_ax.set_title('Distribution', fontsize=8)
            inset_ax.tick_params(labelsize=6)
            
            # Quartile markieren
            q1, q2, q3 = overlap_values.quantile([0.25, 0.5, 0.75])
            
            for q, label, color in [(q1, 'Q1', 'orange'), (q2, 'Q2', 'green'), (q3, 'Q3', 'blue')]:
                ax4.axhline(q, color=color, linestyle=':', alpha=0.7, 
                           linewidth=1.5, label=f'{label}: {q:.3f}')
            
            ax4.set_title('📊 Overlap at N Analysis', fontweight='bold')
            ax4.set_xlabel('Global Training Step', fontweight='bold')
            ax4.set_ylabel('Overlap at N', fontweight='bold')
            ax4.grid(True, alpha=0.3, linestyle=':')
            ax4.legend(loc='best', fontsize=9)
        
        plt.tight_layout()
        
        save_path = os.path.join(self.plots_dir, f'detailed_analysis_step_{current_step}.png')
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved detailed analysis plot: {save_path}")
    
    def _plot_perturbation_deep_dive(self, combined_results, current_step):
        """Tiefgehende Analyse spezifischer Perturbations mit erweiterten ML-Techniken"""
        
        if combined_results is None or len(combined_results) == 0:
            return
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        fig.suptitle(f'🔍 Perturbation Deep Dive Analysis (Step: {current_step:,})', 
                    fontsize=16, fontweight='bold')
        
        latest_eval = combined_results[combined_results['eval_step'] == combined_results['eval_step'].max()]
        
        # 1. Performance Distribution by Perturbation Type mit erweiterten Statistiken [[0]](#__0)
        ax1 = axes[0, 0]
        
        def extract_perturbation_type(pert_name):
            """Extrahiert Perturbation-Typ aus dem Namen mit verbesserter Kategorisierung"""
            pert_lower = pert_name.lower()
            if any(word in pert_lower for word in ['noise', 'gaussian', 'random']):
                return 'Noise'
            elif any(word in pert_lower for word in ['blur', 'smooth', 'filter']):
                return 'Blur'
            elif any(word in pert_lower for word in ['bright', 'light', 'illumination']):
                return 'Brightness'
            elif any(word in pert_lower for word in ['contrast', 'gamma']):
                return 'Contrast'
            elif any(word in pert_lower for word in ['rotation', 'rotate', 'angle']):
                return 'Rotation'
            elif any(word in pert_lower for word in ['scale', 'zoom', 'resize']):
                return 'Scale'
            elif any(word in pert_lower for word in ['crop', 'cut']):
                return 'Cropping'
            elif any(word in pert_lower for word in ['color', 'hue', 'saturation']):
                return 'Color'
            else:
                return 'Other'
        
        if 'perturbation' in latest_eval.columns:
            latest_eval['pert_type'] = latest_eval['perturbation'].apply(extract_perturbation_type)
            
            if 'discrimination_score_l1' in latest_eval.columns:
                pert_types = latest_eval['pert_type'].unique()
                
                # Erweiterte Box-Plot-Analyse
                box_data = []
                box_labels = []
                colors = plt.cm.Set3(np.linspace(0, 1, len(pert_types)))
                
                for i, pert_type in enumerate(pert_types):
                    type_data = latest_eval[latest_eval['pert_type'] == pert_type]['discrimination_score_l1']
                    if len(type_data) > 0:
                        box_data.append(type_data)
                        box_labels.append(f'{pert_type}\n(n={len(type_data)})')
                
                if box_data:
                    bp = ax1.boxplot(box_data, labels=box_labels, patch_artist=True, 
                                    showmeans=True, meanline=True)
                    
                    # Farbkodierung und Statistiken
                    for i, (patch, data) in enumerate(zip(bp['boxes'], box_data)):
                        patch.set_facecolor(colors[i])
                        patch.set_alpha(0.7)
                        
                        # Füge Median-Werte als Text hinzu
                        median_val = np.median(data)
                        ax1.text(i+1, median_val, f'{median_val:.3f}', 
                                ha='center', va='bottom', fontweight='bold', fontsize=8)
                    
                    ax1.set_title('📊 Performance by Perturbation Type', fontweight='bold')
                    ax1.set_ylabel('Discrimination Score L1', fontweight='bold')
                    ax1.tick_params(axis='x', rotation=45)
                    ax1.grid(True, alpha=0.3, axis='y')
        
        # 2. Top vs Bottom Performers Comparison mit statistischen Tests [[1]](#__1)
                # 2. Top vs Bottom Performers Comparison mit statistischen Tests 
        ax2 = axes[0, 1]
        
        if 'discrimination_score_l1' in latest_eval.columns and len(latest_eval) >= 10:
            # Top 10 und Bottom 10
            top_10 = latest_eval.nlargest(10, 'discrimination_score_l1')
            bottom_10 = latest_eval.nsmallest(10, 'discrimination_score_l1')
            
            # Vergleiche andere Metriken
            other_metrics = [col for col in ['overlap_at_N', 'mae'] if col in latest_eval.columns]
            
            if other_metrics:
                metric = other_metrics[0]
                
                top_values = top_10[metric].dropna()
                bottom_values = bottom_10[metric].dropna()
                
                # Erweiterte Violin Plot mit statistischen Annotationen
                data_to_plot = [top_values, bottom_values]
                parts = ax2.violinplot(data_to_plot, positions=[1, 2], 
                                     showmeans=True, showmedians=True, showextrema=True)
                
                # Verbesserte Farbkodierung
                colors = ['lightgreen', 'lightcoral']
                for pc, color in zip(parts['bodies'], colors):
                    pc.set_facecolor(color)
                    pc.set_alpha(0.7)
                    pc.set_edgecolor('black')
                
                # Statistische Signifikanz testen
                try:
                    from scipy.stats import mannwhitneyu
                    statistic, p_value = mannwhitneyu(top_values, bottom_values, 
                                                    alternative='two-sided')
                    significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                    
                    ax2.text(0.5, 0.95, f'Mann-Whitney U test\np = {p_value:.4f} {significance}', 
                            transform=ax2.transAxes, ha='center', va='top',
                            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
                except ImportError:
                    pass
                
                ax2.set_xticks([1, 2])
                ax2.set_xticklabels(['Top 10\nPerformers', 'Bottom 10\nPerformers'])
                ax2.set_ylabel(metric.replace('_', ' ').title(), fontweight='bold')
                ax2.set_title(f'📈 {metric} Distribution:\nTop vs Bottom Performers', fontweight='bold')
                ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Performance Evolution Over Time mit Trend-Analyse 
        ax3 = axes[1, 0]
        
        if len(combined_results['eval_step'].unique()) > 1:
            # Wähle Top 5 Perturbations aus der letzten Evaluation
            if 'discrimination_score_l1' in latest_eval.columns:
                top_5_perts = latest_eval.nlargest(5, 'discrimination_score_l1')['perturbation'].tolist()
                
                eval_steps = sorted(combined_results['eval_step'].unique())
                colors = plt.cm.tab10(np.linspace(0, 1, len(top_5_perts)))
                
                for i, pert in enumerate(top_5_perts):
                    pert_data = combined_results[combined_results['perturbation'] == pert]
                    if len(pert_data) > 1:
                        steps = pert_data['eval_step']
                        values = pert_data['discrimination_score_l1']
                        
                        # Hauptlinie
                        line = ax3.plot(steps, values, 'o-', 
                                       label=pert[:20] + '...' if len(pert) > 20 else pert,
                                       linewidth=2.5, markersize=6, color=colors[i], alpha=0.8)
                        
                        # Trend-Linie
                        if len(values) > 2:
                            z = np.polyfit(range(len(values)), values, 1)
                            trend_slope = z[0]
                            trend_line = np.poly1d(z)
                            ax3.plot(steps, trend_line(range(len(values))), 
                                    '--', color=colors[i], alpha=0.5, linewidth=1.5)
                            
                            # Trend-Indikator
                            trend_symbol = '↗' if trend_slope > 0 else '↘' if trend_slope < 0 else '→'
                            ax3.text(steps.iloc[-1], values.iloc[-1], trend_symbol, 
                                    fontsize=12, color=colors[i], fontweight='bold')
                
                ax3.set_xlabel('Evaluation Step', fontweight='bold')
                ax3.set_ylabel('Discrimination Score L1', fontweight='bold')
                ax3.set_title('🚀 Top 5 Perturbations Evolution', fontweight='bold')
                ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
                ax3.grid(True, alpha=0.3)
        
        # 4. Metric Relationships Scatter Plot mit Clustering 
        ax4 = axes[1, 1]
        
        available_metrics = [col for col in ['overlap_at_N', 'mae', 'discrimination_score_l1'] 
                            if col in latest_eval.columns]
        
        if len(available_metrics) >= 2:
            x_metric = available_metrics[0]
            y_metric = available_metrics[1]
            
            # Scatter plot mit erweiterten Features
            if 'pert_type' in latest_eval.columns:
                pert_types = latest_eval['pert_type'].unique()
                colors = plt.cm.tab10(np.linspace(0, 1, len(pert_types)))
                
                for i, pert_type in enumerate(pert_types):
                    type_data = latest_eval[latest_eval['pert_type'] == pert_type]
                    scatter = ax4.scatter(type_data[x_metric], type_data[y_metric], 
                                        c=[colors[i]], label=pert_type, alpha=0.7, s=60,
                                        edgecolors='black', linewidth=0.5)
            else:
                ax4.scatter(latest_eval[x_metric], latest_eval[y_metric], 
                           alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
            
            # Korrelationslinie
            try:
                correlation = latest_eval[x_metric].corr(latest_eval[y_metric])
                z = np.polyfit(latest_eval[x_metric], latest_eval[y_metric], 1)
                p = np.poly1d(z)
                x_line = np.linspace(latest_eval[x_metric].min(), latest_eval[x_metric].max(), 100)
                ax4.plot(x_line, p(x_line), 'r--', alpha=0.8, linewidth=2,
                        label=f'Correlation: {correlation:.3f}')
            except:
                pass
            
            ax4.set_xlabel(x_metric.replace('_', ' ').title(), fontweight='bold')
            ax4.set_ylabel(y_metric.replace('_', ' ').title(), fontweight='bold')
            ax4.set_title(f'🎯 {x_metric} vs {y_metric}\nby Perturbation Type', fontweight='bold')
            ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
            ax4.grid(True, alpha=0.3)
        
        # 5. Performance Consistency Analysis mit erweiterten Metriken
        ax5 = axes[2, 0]
        
        if len(combined_results['eval_step'].unique()) > 2:
            # Berechne erweiterte Konsistenz-Metriken
            consistency_data = []
            stability_data = []
            pert_names = []
            
            for pert in combined_results['perturbation'].unique():
                pert_data = combined_results[combined_results['perturbation'] == pert]
                
                if len(pert_data) >= 3 and 'discrimination_score_l1' in pert_data.columns:
                    values = pert_data['discrimination_score_l1']
                    
                    # Konsistenz = 1 / (Variationskoeffizient + kleine Konstante)
                    cv = values.std() / values.mean() if values.mean() != 0 else 1
                    consistency = 1 / (cv + 0.001)
                    
                    # Stabilität = 1 - (Max Änderung / Durchschnitt)
                    max_change = np.max(np.abs(np.diff(values)))
                    avg_value = values.mean()
                    stability = 1 - (max_change / avg_value) if avg_value != 0 else 0
                    stability = max(0, min(1, stability))
                    
                    consistency_data.append(consistency)
                    stability_data.append(stability)
                    pert_names.append(pert)
            
            if consistency_data and stability_data:
                # Kombiniere Konsistenz und Stabilität
                combined_score = [(c + s) / 2 for c, s in zip(consistency_data, stability_data)]
                
                # Sortiere nach kombiniertem Score
                sorted_indices = np.argsort(combined_score)[::-1][:15]  # Top 15
                
                top_scores = [combined_score[i] for i in sorted_indices]
                top_names = [pert_names[i] for i in sorted_indices]
                
                y_pos = np.arange(len(top_scores))
                bars = ax5.barh(y_pos, top_scores)
                
                # Farbkodierung basierend auf Score
                colors = plt.cm.RdYlGn(np.array(top_scores) / max(top_scores))
                for bar, color in zip(bars, colors):
                    bar.set_color(color)
                    bar.set_alpha(0.8)
                    bar.set_edgecolor('black')
                    bar.set_linewidth(0.5)
                
                ax5.set_yticks(y_pos)
                ax5.set_yticklabels([name[:25] + '...' if len(name) > 25 else name 
                                    for name in top_names], fontsize=8)
                ax5.set_xlabel('Combined Consistency Score', fontweight='bold')
                ax5.set_title('🎯 Most Consistent Perturbations\n(Consistency + Stability)', fontweight='bold')
                ax5.grid(True, alpha=0.3, axis='x')
                
                # Score-Werte auf Balken
                for i, (bar, score) in enumerate(zip(bars, top_scores)):
                    ax5.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                            f'{score:.2f}', ha='left', va='center', fontsize=8, fontweight='bold')
        
        # 6. Advanced Anomaly Detection mit Multiple Algorithmen
        ax6 = axes[2, 1]
        
        if 'discrimination_score_l1' in latest_eval.columns and len(latest_eval) >= 10:
            try:
                from sklearn.ensemble import IsolationForest
                from sklearn.preprocessing import StandardScaler
                
                # Verwende alle verfügbaren numerischen Metriken
                numeric_data = latest_eval[available_metrics].dropna()
                
                if len(numeric_data) >= 10:
                    # Standardisierung für bessere Anomalie-Erkennung
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(numeric_data)
                    
                    # Isolation Forest mit optimierten Parametern
                    iso_forest = IsolationForest(contamination=0.1, random_state=42, 
                                               n_estimators=100)
                    anomaly_labels = iso_forest.fit_predict(scaled_data)
                    
                    # Anomalie-Scores für Farbkodierung
                    anomaly_scores = iso_forest.decision_function(scaled_data)
                    
                    # Visualisierung mit Anomalie-Scores
                    normal_mask = anomaly_labels == 1
                    anomaly_mask = anomaly_labels == -1
                    
                    if len(available_metrics) >= 2:
                        x_vals = numeric_data[available_metrics[0]]
                        y_vals = numeric_data[available_metrics[1]]
                        
                        # Normale Punkte
                        scatter_normal = ax6.scatter(x_vals[normal_mask], y_vals[normal_mask], 
                                                   c=anomaly_scores[normal_mask], cmap='viridis',
                                                   alpha=0.7, s=60, edgecolors='black', 
                                                   linewidth=0.5, label='Normal')
                        
                        # Anomalien
                        ax6.scatter(x_vals[anomaly_mask], y_vals[anomaly_mask], 
                                   c='red', alpha=0.9, s=120, marker='^', 
                                   edgecolors='darkred', linewidth=2, label='Anomaly')
                        
                        # Colorbar für Anomalie-Scores
                        cbar = plt.colorbar(scatter_normal, ax=ax6, shrink=0.8)
                        cbar.set_label('Anomaly Score', fontsize=9)
                        
                        ax6.set_xlabel(available_metrics[0].replace('_', ' ').title(), fontweight='bold')
                        ax6.set_ylabel(available_metrics[1].replace('_', ' ').title(), fontweight='bold')
                        ax6.set_title(f'🔍 Advanced Anomaly Detection\n({anomaly_mask.sum()} anomalies found)', 
                                     fontweight='bold')
                        ax6.legend()
                        ax6.grid(True, alpha=0.3)
                        
                        # Top-Anomalien anzeigen
                        if anomaly_mask.sum() > 0 and anomaly_mask.sum() <= 3:
                            anomaly_indices = numeric_data.index[anomaly_mask]
                            anomaly_perts = latest_eval.loc[anomaly_indices, 'perturbation']
                            anomaly_text = '\n'.join([p[:15] + '...' if len(p) > 15 else p 
                                                    for p in anomaly_perts])
                            ax6.text(0.02, 0.98, f'Anomalies:\n{anomaly_text}', 
                                    transform=ax6.transAxes, fontsize=8, 
                                    verticalalignment='top',
                                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
            except ImportError:
                ax6.text(0.5, 0.5, 'sklearn not available\nfor anomaly detection', 
                        ha='center', va='center', transform=ax6.transAxes, fontsize=12)
        
        plt.tight_layout()
        save_path = os.path.join(self.plots_dir, f'perturbation_deep_dive_step_{current_step}.png')
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved perturbation deep dive analysis: {save_path}")

    def _plot_statistical_analysis(self, combined_results, current_step):
        """Erweiterte statistische Analyse mit robusten statistischen Tests"""
        
        if combined_results is None or len(combined_results) == 0:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'📊 Advanced Statistical Analysis (Step: {current_step:,})', 
                    fontsize=16, fontweight='bold')
        
        latest_eval = combined_results[combined_results['eval_step'] == combined_results['eval_step'].max()]
        
        # 1. Enhanced Q-Q Plot mit Shapiro-Wilk Test 
        ax1 = axes[0, 0]
        if 'discrimination_score_l1' in latest_eval.columns:
            try:
                from scipy import stats
                values = latest_eval['discrimination_score_l1'].dropna()
                
                # Q-Q Plot
                stats.probplot(values, dist="norm", plot=ax1)
                
                # Shapiro-Wilk Test für Normalität
                shapiro_stat, shapiro_p = stats.shapiro(values[:5000] if len(values) > 5000 else values)
                normality = "Normal" if shapiro_p > 0.05 else "Non-Normal"
                
                ax1.set_title(f'📈 Q-Q Plot (Normal Distribution)\nShapiro-Wilk: p={shapiro_p:.4f} ({normality})', 
                             fontweight='bold')
                ax1.grid(True, alpha=0.3)
                
                # R² für Normalitäts-Fit
                theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(values)))
                sample_quantiles = np.sort(values)
                r_squared = stats.pearsonr(theoretical_quantiles, sample_quantiles)[0]**2
                
                ax1.text(0.05, 0.95, f'R² = {r_squared:.3f}', transform=ax1.transAxes,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
            except ImportError:
                ax1.text(0.5, 0.5, 'scipy not available\nfor Q-Q plot', 
                        ha='center', va='center', transform=ax1.transAxes)
        
        # 2. Enhanced Bootstrap Analysis mit Bias-Correction 
        ax2 = axes[0, 1]
        if 'discrimination_score_l1' in latest_eval.columns:
            values = latest_eval['discrimination_score_l1'].dropna()
            
            # Bootstrap sampling mit Bias-Korrektur
            n_bootstrap = 2000
            bootstrap_means = []
            bootstrap_medians = []
            
            original_mean = np.mean(values)
            
            for _ in range(n_bootstrap):
                sample = np.random.choice(values, size=len(values), replace=True)
                bootstrap_means.append(np.mean(sample))
                bootstrap_medians.append(np.median(sample))
            
            bootstrap_means = np.array(bootstrap_means)
            bootstrap_medians = np.array(bootstrap_medians)
            
            # Bias-corrected Bootstrap Confidence Intervals
            alpha = 0.05  # 95% CI
            ci_lower_mean = np.percentile(bootstrap_means, 100 * alpha/2)
            ci_upper_mean = np.percentile(bootstrap_means, 100 * (1 - alpha/2))
            
            # Bias correction
            bias = np.mean(bootstrap_means) - original_mean
            bias_corrected_mean = original_mean - bias
            
            # Plotting
            ax2.hist(bootstrap_means, bins=50, alpha=0.7, color='lightblue', 
                    edgecolor='black', density=True, label='Bootstrap Means')
            ax2.axvline(original_mean, color='red', linestyle='-', linewidth=3, 
                       label=f'Original Mean: {original_mean:.4f}')
            ax2.axvline(bias_corrected_mean, color='orange', linestyle='-', linewidth=3,
                       label=f'Bias-Corrected: {bias_corrected_mean:.4f}')
            ax2.axvline(ci_lower_mean, color='green', linestyle='--', linewidth=2, 
                       label=f'95% CI: [{ci_lower_mean:.4f}, {ci_upper_mean:.4f}]')
            ax2.axvline(ci_upper_mean, color='green', linestyle='--', linewidth=2)
            
            ax2.set_xlabel('Bootstrap Sample Means', fontweight='bold')
            ax2.set_ylabel('Density', fontweight='bold')
            ax2.set_title(f'🔄 Bootstrap Analysis (n={n_bootstrap})\nBias: {bias:.6f}', fontweight='bold')
            ax2.legend(fontsize=9)
            ax2.grid(True, alpha=0.3)
        
        # 3. Performance Percentiles Over Time mit Trend-Tests 
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
                if len(percentile_data[p]) > 0:
                    line = ax3.plot(eval_steps, percentile_data[p], 'o-', 
                                   color=colors[i], label=f'{p}th percentile',
                                   linewidth=2, markersize=6)
                    
                    # Trend-Test (Mann-Kendall)
                    if len(percentile_data[p]) > 2:
                        try:
                            from scipy.stats import kendalltau
                            tau, p_value = kendalltau(range(len(percentile_data[p])), percentile_data[p])
                            trend_symbol = '↗' if tau > 0 and p_value < 0.05 else '↘' if tau < 0 and p_value < 0.05 else '→'
                            
                            # Füge Trend-Symbol hinzu
                            ax3.text(eval_steps[-1], percentile_data[p][-1], trend_symbol, 
                                    fontsize=12, color=colors[i], fontweight='bold')
                        except ImportError:
                            pass
            
            ax3.set_xlabel('Evaluation Step', fontweight='bold')
            ax3.set_ylabel('Discrimination Score L1', fontweight='bold')
            ax3.set_title('📊 Performance Percentiles Over Time\n(with Trend Analysis)', fontweight='bold')
            ax3.legend(fontsize=9)
            ax3.grid(True, alpha=0.3)
        
        # 4. Enhanced Effect Size Analysis mit Cohen's d und Hedges' g 
        ax4 = axes[1, 0]
        if len(combined_results['eval_step'].unique()) >= 2:
            eval_steps = sorted(combined_results['eval_step'].unique())
            
            if len(eval_steps) >= 2:
                first_eval = combined_results[combined_results['eval_step'] == eval_steps[0]]
                last_eval = combined_results[combined_results['eval_step'] == eval_steps[-1]]
                
                if 'discrimination_score_l1' in first_eval.columns and 'discrimination_score_l1' in last_eval.columns:
                    first_values = first_eval['discrimination_score_l1'].dropna()
                    last_values = last_eval['discrimination_score_l1'].dropna()
                    
                    # Cohen's d
                    pooled_std = np.sqrt(((len(first_values) - 1) * first_values.var() + 
                                        (len(last_values) - 1) * last_values.var()) / 
                                       (len(first_values) + len(last_values) - 2))
                    
                    cohens_d = (last_values.mean() - first_values.mean()) / pooled_std if pooled_std > 0 else 0
                    
                    # Hedges' g (bias-corrected)
                    j = 1 - (3 / (4 * (len(first_values) + len(last_values)) - 9))
                    hedges_g = cohens_d * j
                    
                    # Visualisierung mit erweiterten Statistiken
                    positions = [1, 2]
                    means = [first_values.mean(), last_values.mean()]
                    stds = [first_values.std(), last_values.std()]
                    
                    bars = ax4.bar(positions, means, 
                                  color=['lightcoral', 'lightgreen'], alpha=0.7,
                                  edgecolor='black', linewidth=1.5)
                    
                    # Error bars mit 95% CI
                    ci_first = 1.96 * first_values.std() / np.sqrt(len(first_values))
                    ci_last = 1.96 * last_values.std() / np.sqrt(len(last_values))
                    
                    ax4.errorbar(positions, means, yerr=[ci_first, ci_last],
                                fmt='none', color='black', capsize=8, capthick=2)
                    
                    ax4.set_xticks(positions)
                    ax4.set_xticklabels(['First Eval', 'Last Eval'])
                    ax4.set_ylabel('Discrimination Score L1', fontweight='bold')
                    ax4.set_title(f'📈 Effect Size Analysis\nCohen\'s d = {cohens_d:.3f}, Hedges\' g = {hedges_g:.3f}', 
                                 fontweight='bold')
                    ax4.grid(True, alpha=0.3, axis='y')
                    
                    # Effect Size Interpretation mit Farben
                    if abs(cohens_d) < 0.2:
                        effect_size, color = "Small", "yellow"
                    elif abs(cohens_d) < 0.5:
                        effect_size, color = "Medium", "orange"
                    elif abs(cohens_d) < 0.8:
                        effect_size, color = "Large", "lightgreen"
                    else:
                        effect_size, color = "Very Large", "green"
                    
                    ax4.text(0.5, 0.95, f'Effect Size: {effect_size}', 
                            transform=ax4.transAxes, ha='center', va='top',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8),
                            fontweight='bold')
        
        # 5. Multi-Algorithm Outlier Detection
        ax5 = axes[1, 1]
        if 'discrimination_score_l1' in latest_eval.columns and len(latest_eval) >= 10:
            try:
                from sklearn.ensemble import IsolationForest
                from sklearn.neighbors import LocalOutlierFactor
                from sklearn.preprocessing import StandardScaler
                
                # Verwende alle verfügbaren numerischen Metriken
                available_metrics = [col for col in ['overlap_at_N', 'mae', 'discrimination_score_l1'] 
                                   if col in latest_eval.columns]
                numeric_data = latest_eval[available_metrics].dropna()
                
                if len(numeric_data) >= 10:
                    # Standardisierung
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(numeric_data)
                    
                    # Multiple Outlier Detection Algorithmen
                    iso_forest = IsolationForest(contamination=0.1, random_state=42)
                    lof = LocalOutlierFactor(n_neighbors=min(20, len(numeric_data)//2), contamination=0.1)
                    
                    iso_outliers = iso_forest.fit_predict(scaled_data)
                    lof_outliers = lof.fit_predict(scaled_data)
                    
                    # Consensus Outliers (beide Algorithmen stimmen überein)
                    consensus_outliers = (iso_outliers == -1) & (lof_outliers == -1)
                    iso_only = (iso_outliers == -1) & (lof_outliers == 1)
                    lof_only = (iso_outliers == 1) & (lof_outliers == -1)
                    normal = (iso_outliers == 1) & (lof_outliers == 1)
                    
                    if len(available_metrics) >= 2:
                        x_vals = numeric_data[available_metrics[0]]
                        y_vals = numeric_data[available_metrics[1]]
                        
                        # Verschiedene Outlier-Typen plotten
                        ax5.scatter(x_vals[normal], y_vals[normal], 
                                   c='blue', alpha=0.6, s=50, label='Normal', marker='o')
                        ax5.scatter(x_vals[iso_only], y_vals[iso_only], 
                                   c='orange', alpha=0.8, s=80, label='Isolation Forest Only', marker='s')
                        ax5.scatter(x_vals[lof_only], y_vals[lof_only], 
                                   c='purple', alpha=0.8, s=80, label='LOF Only', marker='^')
                        ax5.scatter(x_vals[consensus_outliers], y_vals[consensus_outliers], 
                                   c='red', alpha=0.9, s=120, label='Consensus Outliers', marker='*')
                        ax5.set_xlabel(available_metrics[0].replace('_', ' ').title(), fontweight='bold')
                        ax5.set_ylabel(available_metrics[1].replace('_', ' ').title(), fontweight='bold')
                        ax5.set_title(f'🔍 Multi-Algorithm Outlier Detection\n'
                                     f'Consensus: {consensus_outliers.sum()}, '
                                     f'ISO: {iso_only.sum()}, LOF: {lof_only.sum()}', 
                                     fontweight='bold')
                        ax5.legend(fontsize=9)
                        ax5.grid(True, alpha=0.3)
                        
                        # Zeige extreme Outliers
                        if consensus_outliers.sum() > 0:
                            outlier_indices = numeric_data.index[consensus_outliers]
                            outlier_perts = latest_eval.loc[outlier_indices, 'perturbation']
                            outlier_text = '\n'.join([p[:20] + '...' if len(p) > 20 else p 
                                                    for p in outlier_perts[:3]])
                            ax5.text(0.02, 0.98, f'Extreme Outliers:\n{outlier_text}', 
                                    transform=ax5.transAxes, fontsize=8, 
                                    verticalalignment='top',
                                    bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.3))
            except ImportError:
                ax5.text(0.5, 0.5, 'sklearn not available\nfor outlier detection', 
                        ha='center', va='center', transform=ax5.transAxes, fontsize=12)
        
        # 6. Advanced Correlation Matrix mit Hierarchical Clustering
        ax6 = axes[1, 2]
        
        # Erweiterte Korrelationsanalyse mit verfügbaren Metriken
        available_metrics = [col for col in ['overlap_at_N', 'mae', 'discrimination_score_l1'] 
                           if col in latest_eval.columns]
        
        if len(available_metrics) >= 2:
            correlation_data = latest_eval[available_metrics].corr()
            
            try:
                from scipy.cluster.hierarchy import dendrogram, linkage
                from scipy.spatial.distance import squareform
                
                # Hierarchical Clustering der Korrelationsmatrix
                distance_matrix = 1 - np.abs(correlation_data)
                condensed_distances = squareform(distance_matrix, checks=False)
                linkage_matrix = linkage(condensed_distances, method='average')
                
                # Dendrogram für Clustering-Visualisierung
                dendro = dendrogram(linkage_matrix, labels=available_metrics, 
                                  ax=ax6, orientation='top', leaf_rotation=45)
                
                ax6.set_title('🌳 Metric Clustering Dendrogram\n(Based on Correlation Distance)', 
                             fontweight='bold')
                ax6.set_ylabel('Distance (1 - |correlation|)', fontweight='bold')
                
                # Füge Korrelationswerte als Text hinzu
                cluster_info = f"Metrics: {len(available_metrics)}\n"
                avg_correlation = np.mean(np.abs(correlation_data.values[np.triu_indices_from(correlation_data.values, k=1)]))
                cluster_info += f"Avg |correlation|: {avg_correlation:.3f}"
                
                ax6.text(0.02, 0.98, cluster_info, transform=ax6.transAxes, 
                        verticalalignment='top', fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
                
            except ImportError:
                # Fallback: Einfache Korrelationsmatrix
                im = ax6.imshow(correlation_data, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
                ax6.set_xticks(range(len(available_metrics)))
                ax6.set_yticks(range(len(available_metrics)))
                ax6.set_xticklabels([m.replace('_', '\n') for m in available_metrics], rotation=45)
                ax6.set_yticklabels([m.replace('_', '\n') for m in available_metrics])
                
                # Korrelationswerte in Zellen
                for i in range(len(available_metrics)):
                    for j in range(len(available_metrics)):
                        text = ax6.text(j, i, f'{correlation_data.iloc[i, j]:.2f}',
                                       ha="center", va="center", color="black", fontweight='bold')
                
                plt.colorbar(im, ax=ax6, shrink=0.8)
                ax6.set_title('📊 Correlation Matrix\n(Fallback View)', fontweight='bold')
        
        plt.tight_layout()
        save_path = os.path.join(self.plots_dir, f'statistical_analysis_step_{current_step}.png')
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved advanced statistical analysis: {save_path}")
    
    def _create_multi_metric_analysis_plots(self, current_step):
        """Multi-Metrik Analyse mit erweiterten Korrelationen und Clustering"""
        
        if len(self.all_metrics_history) < 2:
            return
        
        # Sammle alle verfügbaren Metriken
        all_metric_names = set()
        for metrics_dict in self.all_metrics_history:
            all_metric_names.update(metrics_dict.keys())
        
        # Filtere numerische Metriken
        numeric_metrics = {}
        for metric_name in all_metric_names:
            values = []
            for metrics_dict in self.all_metrics_history:
                if metric_name in metrics_dict:
                    try:
                        val = float(metrics_dict[metric_name])
                        values.append(val)
                    except (ValueError, TypeError):
                        values.append(np.nan)
                else:
                    values.append(np.nan)
            
            # Nur Metriken mit genügend validen Werten
            valid_count = sum(1 for v in values if not np.isnan(v))
            if valid_count >= max(2, len(values) * 0.5):
                numeric_metrics[metric_name] = values
        
        if len(numeric_metrics) < 2:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'🔍 Multi-Metric Analysis (Step: {current_step:,})', 
                    fontsize=16, fontweight='bold')
        
        # 1. Korrelations-Heatmap mit erweiterten Statistiken 
        ax1 = axes[0, 0]
        
        # DataFrame für Korrelationsanalyse
        df_metrics = pd.DataFrame(numeric_metrics)
        correlation_matrix = df_metrics.corr()
        
        # Erweiterte Heatmap
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))  # Obere Dreiecksmatrix maskieren
        
        im = ax1.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        
        # Achsen-Labels
        ax1.set_xticks(range(len(correlation_matrix.columns)))
        ax1.set_yticks(range(len(correlation_matrix.index)))
        ax1.set_xticklabels([col.replace('_', '\n')[:15] for col in correlation_matrix.columns], 
                           rotation=45, ha='right', fontsize=9)
        ax1.set_yticklabels([idx.replace('_', '\n')[:15] for idx in correlation_matrix.index], 
                           fontsize=9)
        
        # Korrelationswerte in Zellen
        for i in range(len(correlation_matrix.index)):
            for j in range(len(correlation_matrix.columns)):
                if not mask[i, j]:  # Nur untere Dreiecksmatrix
                    corr_val = correlation_matrix.iloc[i, j]
                    color = 'white' if abs(corr_val) > 0.5 else 'black'
                    ax1.text(j, i, f'{corr_val:.2f}', ha="center", va="center", 
                            color=color, fontweight='bold', fontsize=8)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax1, shrink=0.8)
        cbar.set_label('Correlation Coefficient', fontsize=10, fontweight='bold')
        
        ax1.set_title('🔥 Metric Correlation Matrix', fontweight='bold')
        
        # Statistiken
        avg_abs_corr = np.mean(np.abs(correlation_matrix.values[~mask]))
        max_corr = np.max(correlation_matrix.values[~mask])
        min_corr = np.min(correlation_matrix.values[~mask])
        
        stats_text = f'Avg |r|: {avg_abs_corr:.3f}\nMax r: {max_corr:.3f}\nMin r: {min_corr:.3f}'
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # 2. Principal Component Analysis (falls sklearn verfügbar) 
        ax2 = axes[0, 1]
        
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            
            # Daten vorbereiten
            df_clean = df_metrics.dropna()
            
            if len(df_clean) >= 3 and len(df_clean.columns) >= 2:
                # Standardisierung
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(df_clean)
                
                # PCA
                pca = PCA()
                pca_result = pca.fit_transform(scaled_data)
                
                # Scree Plot
                explained_variance_ratio = pca.explained_variance_ratio_
                cumulative_variance = np.cumsum(explained_variance_ratio)
                
                ax2.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, 
                       alpha=0.7, color='lightblue', edgecolor='black', label='Individual')
                ax2.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 
                        'ro-', linewidth=2, markersize=6, label='Cumulative')
                
                # 80% Varianz-Linie
                ax2.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, 
                           linewidth=2, label='80% Variance')
                
                ax2.set_xlabel('Principal Component', fontweight='bold')
                ax2.set_ylabel('Explained Variance Ratio', fontweight='bold')
                ax2.set_title('📊 PCA Scree Plot', fontweight='bold')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                # Anzahl Komponenten für 80% Varianz
                n_components_80 = np.argmax(cumulative_variance >= 0.8) + 1
                ax2.text(0.02, 0.98, f'Components for 80%: {n_components_80}', 
                        transform=ax2.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        except ImportError:
            ax2.text(0.5, 0.5, 'sklearn not available\nfor PCA analysis', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        
        # 3. Metric Evolution Clustering 
        ax3 = axes[1, 0]
        
        # Identifiziere ähnliche Entwicklungsmuster
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            
            # Transpose für Clustering von Metriken (nicht Zeitpunkten)
            df_for_clustering = df_metrics.dropna().T
            
            if len(df_for_clustering) >= 3:
                scaler = StandardScaler()
                scaled_metrics = scaler.fit_transform(df_for_clustering)
                
                # Optimale Anzahl Cluster (Elbow Method)
                max_clusters = min(5, len(df_for_clustering) - 1)
                inertias = []
                
                for k in range(1, max_clusters + 1):
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    kmeans.fit(scaled_metrics)
                    inertias.append(kmeans.inertia_)
                
                # Elbow Plot
                ax3.plot(range(1, len(inertias) + 1), inertias, 'bo-', linewidth=2, markersize=8)
                ax3.set_xlabel('Number of Clusters', fontweight='bold')
                ax3.set_ylabel('Inertia', fontweight='bold')
                ax3.set_title('📈 Metric Clustering (Elbow Method)', fontweight='bold')
                ax3.grid(True, alpha=0.3)
                
                # Optimale Cluster-Anzahl schätzen
                if len(inertias) > 2:
                    # Einfache Elbow-Erkennung
                    diffs = np.diff(inertias)
                    second_diffs = np.diff(diffs)
                    optimal_k = np.argmax(second_diffs) + 2  # +2 wegen doppeltem diff
                    
                    ax3.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7,
                               linewidth=2, label=f'Suggested k={optimal_k}')
                    ax3.legend()
        
        except ImportError:
            ax3.text(0.5, 0.5, 'sklearn not available\nfor clustering analysis', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        
        # 4. Metric Importance Ranking mit Variabilität 
        ax4 = axes[1, 1]
        
        # Berechne verschiedene Wichtigkeits-Metriken
        importance_scores = {}
        
        for metric_name, values in numeric_metrics.items():
            clean_values = [v for v in values if not np.isnan(v)]
            
            if len(clean_values) > 1:
                # Kombinierte Wichtigkeit: Variabilität + Trend + Stabilität
                variability = np.std(clean_values) / (np.mean(np.abs(clean_values)) + 1e-8)
                
                # Trend-Stärke
                if len(clean_values) > 2:
                    trend_strength = abs(np.polyfit(range(len(clean_values)), clean_values, 1)[0])
                else:
                    trend_strength = 0
                
                # Stabilität (inverse der relativen Standardabweichung)
                stability = 1 / (1 + variability)
                
                # Kombinierter Score
                combined_score = 0.4 * variability + 0.4 * trend_strength + 0.2 * stability
                importance_scores[metric_name] = combined_score
        
        if importance_scores:
            # Sortiere nach Wichtigkeit
            sorted_metrics = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Top 10 oder alle (falls weniger)
            top_metrics = sorted_metrics[:min(10, len(sorted_metrics))]
            
            names = [name.replace('_', '\n')[:20] for name, _ in top_metrics]
            scores = [score for _, score in top_metrics]
            
            # Horizontales Balkendiagramm
            y_pos = np.arange(len(names))
            bars = ax4.barh(y_pos, scores, color=plt.cm.viridis(np.linspace(0, 1, len(scores))),
                           alpha=0.8, edgecolor='black', linewidth=0.5)
            
            ax4.set_yticks(y_pos)
            ax4.set_yticklabels(names, fontsize=9)
            ax4.set_xlabel('Importance Score', fontweight='bold')
            ax4.set_title('🏆 Metric Importance Ranking\n(Variability + Trend + Stability)', fontweight='bold')
            ax4.grid(True, alpha=0.3, axis='x')
            
            # Score-Werte auf Balken
            for i, (bar, score) in enumerate(zip(bars, scores)):
                ax4.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{score:.3f}', ha='left', va='center', fontsize=8, fontweight='bold')
        
        plt.tight_layout()
        save_path = os.path.join(self.plots_dir, f'multi_metric_analysis_step_{current_step}.png')
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved multi-metric analysis: {save_path}")

    def _create_plot_summary(self, current_step):
        """Erstellt eine Zusammenfassung aller generierten Plots"""
        
        summary_info = {
            'step': current_step,
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_evaluations': len(self.agg_results_history),
            'plots_generated': [],
            'key_insights': []
        }
        
        # Liste aller generierten Plot-Dateien
        plot_files = list(self.plots_dir.glob(f'*step_{current_step}.png'))
        summary_info['plots_generated'] = [f.name for f in plot_files]
        
        # Key Insights basierend auf verfügbaren Daten
        if self.agg_results_history:
            latest_agg = self.agg_results_history[-1]
            
            if 'discrimination_score_l1' in latest_agg.columns:
                current_disc_score = latest_agg['discrimination_score_l1'].iloc[-1]
                summary_info['key_insights'].append(f"Current discrimination score: {current_disc_score:.4f}")
            
            if len(self.agg_results_history) > 1:
                prev_agg = self.agg_results_history[-2]
                if 'discrimination_score_l1' in prev_agg.columns and 'discrimination_score_l1' in latest_agg.columns:
                    prev_score = prev_agg['discrimination_score_l1'].iloc[-1]
                    current_score = latest_agg['discrimination_score_l1'].iloc[-1]
                    change = current_score - prev_score
                    summary_info['key_insights'].append(f"Score change: {change:+.4f}")
        
        # Speichere Zusammenfassung als JSON
        summary_path = os.path.join(self.plots_dir, f'plot_summary_step_{current_step}.json')
        with open(summary_path, 'w') as f:
            json.dump(summary_info, f, indent=2)
        
        logger.info(f"Created plot summary: {summary_path}")
        logger.info(f"Generated {len(summary_info['plots_generated'])} plots for step {current_step}")

    def _plot_performance_trends(self, combined_agg_results, current_step):
        """Performance-Trends mit Vorhersagen und erweiterten Analysen"""
        
        if len(combined_agg_results) < 3:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'📈 Performance Trends & Predictions (Step: {current_step:,})', 
                    fontsize=16, fontweight='bold')
        
        # Verfügbare Metriken identifizieren
        trend_metrics = [col for col in ['discrimination_score_l1', 'mae', 'overlap_at_N'] 
                        if col in combined_agg_results.columns]
        
        for idx, metric in enumerate(trend_metrics[:4]):  # Max 4 Plots
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]
            
            steps = combined_agg_results['global_step']
            values = combined_agg_results[metric]
            
            # Hauptdaten plotten
            ax.plot(steps, values, 'o-', linewidth=3, markersize=8, alpha=0.8, label='Actual')
            
            # Trend-Analyse und Vorhersage
            if len(values) >= 3:
                # Polynomiale Regression für bessere Anpassung
                degree = min(2, len(values) - 1)
                z = np.polyfit(range(len(values)), values, degree)
                p = np.poly1d(z)
                
                # Aktuelle Trend-Linie
                trend_values = p(range(len(values)))
                ax.plot(steps, trend_values, '--', alpha=0.7, color='orange', 
                       linewidth=2, label='Trend')
                
                # Vorhersage für nächste 3 Schritte
                future_indices = range(len(values), len(values) + 3)
                step_diff = steps.iloc[-1] - steps.iloc[-2] if len(steps) > 1 else 1000
                future_steps = [steps.iloc[-1] + (i+1) * step_diff for i in range(3)]
                future_values = [p(i) for i in future_indices]
                
                ax.plot(future_steps, future_values, 's--', alpha=0.6, color='red',
                       linewidth=2, markersize=6, label='Prediction')
                
                # Konfidenzintervall für Vorhersage
                residuals = values - trend_values
                std_residual = np.std(residuals)
                
                ax.fill_between(future_steps, 
                               [v - std_residual for v in future_values],
                               [v + std_residual for v in future_values],
                               alpha=0.2, color='red', label='±1σ Prediction')
            
            # Performance-Zonen
            mean_val = values.mean()
            std_val = values.std()
            
            ax.axhspan(mean_val + std_val, values.max(), alpha=0.1, color='green', 
                      label='High Performance')
            ax.axhspan(values.min(), mean_val - std_val, alpha=0.1, color='red', 
                      label='Low Performance')
            ax.axhline(mean_val, color='gray', linestyle=':', alpha=0.7, 
                      label=f'Mean: {mean_val:.4f}')
            
            # Titel und Labels
            clean_name = metric.replace('_', ' ').title()
            ax.set_title(f'{clean_name} Trend Analysis', fontweight='bold')
            ax.set_xlabel('Global Training Step', fontweight='bold')
            ax.set_ylabel(clean_name, fontweight='bold')
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
            
            # Trend-Statistiken
                        # Trend-Statistiken
            if len(values) >= 3:
                # Berechne Trend-Stärke und -Richtung
                recent_trend = np.polyfit(range(len(values)//2, len(values)), 
                                        values[len(values)//2:], 1)[0]
                overall_trend = np.polyfit(range(len(values)), values, 1)[0]
                
                trend_direction = "↗" if recent_trend > 0 else "↘" if recent_trend < 0 else "→"
                trend_strength = abs(recent_trend)
                
                # Volatilität
                volatility = np.std(np.diff(values)) / np.mean(np.abs(values))
                
                stats_text = f'{trend_direction} Recent Trend: {recent_trend:.4f}\n'
                stats_text += f'Volatility: {volatility:.3f}\n'
                stats_text += f'R²: {np.corrcoef(range(len(values)), values)[0,1]**2:.3f}'
                
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                       verticalalignment='top', fontsize=9,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Verstecke leere Subplots
        for idx in range(len(trend_metrics), 4):
            row = idx // 2
            col = idx % 2
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        save_path = os.path.join(self.plots_dir, f'performance_trends_step_{current_step}.png')
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved performance trends analysis: {save_path}")

    def _plot_training_dynamics(self, current_step):
        """Training-Dynamik mit erweiterten Lernkurven-Analysen"""
        
        if len(self.all_metrics_history) < 3:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'🚀 Training Dynamics Analysis (Step: {current_step:,})', 
                    fontsize=16, fontweight='bold')
        
        # Sammle alle verfügbaren Metriken über Zeit
        metrics_over_time = {}
        for metrics_dict in self.all_metrics_history:
            for key, value in metrics_dict.items():
                if key not in metrics_over_time:
                    metrics_over_time[key] = []
                try:
                    metrics_over_time[key].append(float(value))
                except (ValueError, TypeError):
                    metrics_over_time[key].append(np.nan)
        
        # Filtere Metriken mit genügend validen Werten
        valid_metrics = {k: v for k, v in metrics_over_time.items() 
                        if sum(1 for x in v if not np.isnan(x)) >= 3}
        
        if not valid_metrics:
            return
        
        steps = list(range(len(list(valid_metrics.values())[0])))
        
        # 1. Learning Rate vs Performance Correlation
        ax1 = axes[0, 0]
        
        # Suche nach Learning Rate Metriken
        lr_metrics = [k for k in valid_metrics.keys() if 'lr' in k.lower() or 'learning_rate' in k.lower()]
        perf_metrics = [k for k in valid_metrics.keys() if any(term in k.lower() 
                       for term in ['loss', 'score', 'accuracy', 'mae'])]
        
        if lr_metrics and perf_metrics:
            lr_metric = lr_metrics[0]
            perf_metric = perf_metrics[0]
            
            lr_values = [v for v in valid_metrics[lr_metric] if not np.isnan(v)]
            perf_values = [v for v in valid_metrics[perf_metric] if not np.isnan(v)]
            
            if len(lr_values) == len(perf_values) and len(lr_values) > 2:
                # Dual-axis Plot
                ax1_twin = ax1.twinx()
                
                line1 = ax1.plot(steps[:len(lr_values)], lr_values, 'b-o', 
                               linewidth=2, markersize=6, alpha=0.8, label=lr_metric)
                line2 = ax1_twin.plot(steps[:len(perf_values)], perf_values, 'r-s', 
                                    linewidth=2, markersize=6, alpha=0.8, label=perf_metric)
                
                ax1.set_xlabel('Training Step', fontweight='bold')
                ax1.set_ylabel(lr_metric.replace('_', ' ').title(), color='blue', fontweight='bold')
                ax1_twin.set_ylabel(perf_metric.replace('_', ' ').title(), color='red', fontweight='bold')
                
                # Korrelation berechnen
                correlation = np.corrcoef(lr_values, perf_values)[0, 1]
                ax1.set_title(f'📊 LR vs Performance\nCorrelation: {correlation:.3f}', fontweight='bold')
                
                # Legende kombinieren
                lines = line1 + line2
                labels = [l.get_label() for l in lines]
                ax1.legend(lines, labels, loc='upper left')
                
                ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'No LR or Performance\nmetrics found', 
                    ha='center', va='center', transform=ax1.transAxes, fontsize=12)
        
        # 2. Training Stability Analysis
        ax2 = axes[0, 1]
        
        # Berechne Stabilität für verschiedene Metriken
        stability_scores = {}
        
        for metric_name, values in valid_metrics.items():
            clean_values = [v for v in values if not np.isnan(v)]
            
            if len(clean_values) >= 5:
                # Rolling Standard Deviation als Stabilitätsmaß
                window_size = min(5, len(clean_values) // 2)
                rolling_std = []
                
                for i in range(window_size, len(clean_values)):
                    window = clean_values[i-window_size:i]
                    rolling_std.append(np.std(window))
                
                if rolling_std:
                    avg_stability = np.mean(rolling_std)
                    stability_scores[metric_name] = 1 / (1 + avg_stability)  # Inverse für höhere = stabiler
        
        if stability_scores:
            # Top 8 stabilste Metriken
            sorted_stability = sorted(stability_scores.items(), key=lambda x: x[1], reverse=True)[:8]
            
            names = [name.replace('_', '\n')[:15] for name, _ in sorted_stability]
            scores = [score for _, score in sorted_stability]
            
            bars = ax2.bar(range(len(names)), scores, 
                          color=plt.cm.RdYlGn(np.array(scores)), alpha=0.8,
                          edgecolor='black', linewidth=0.5)
            
            ax2.set_xticks(range(len(names)))
            ax2.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
            ax2.set_ylabel('Stability Score', fontweight='bold')
            ax2.set_title('🎯 Training Stability Ranking', fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Werte auf Balken
            for bar, score in zip(bars, scores):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{score:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # 3. Gradient/Update Magnitude Analysis (falls verfügbar)
        ax3 = axes[0, 2]
        
        grad_metrics = [k for k in valid_metrics.keys() if any(term in k.lower() 
                       for term in ['grad', 'update', 'norm'])]
        
        if grad_metrics:
            # Zeige bis zu 3 Gradient-Metriken
            colors = ['blue', 'red', 'green']
            
            for i, metric in enumerate(grad_metrics[:3]):
                values = [v for v in valid_metrics[metric] if not np.isnan(v)]
                
                if len(values) > 2:
                    ax3.plot(steps[:len(values)], values, 'o-', 
                            color=colors[i], linewidth=2, markersize=6, alpha=0.8,
                            label=metric.replace('_', ' ')[:20])
                    
                    # Exponential Moving Average für Trend
                    ema = []
                    alpha = 0.3
                    ema.append(values[0])
                    for val in values[1:]:
                        ema.append(alpha * val + (1 - alpha) * ema[-1])
                    
                    ax3.plot(steps[:len(ema)], ema, '--', 
                            color=colors[i], alpha=0.6, linewidth=1.5)
            
            ax3.set_xlabel('Training Step', fontweight='bold')
            ax3.set_ylabel('Magnitude', fontweight='bold')
            ax3.set_title('📈 Gradient/Update Magnitudes', fontweight='bold')
            ax3.legend(fontsize=9)
            ax3.grid(True, alpha=0.3)
            ax3.set_yscale('log')  # Log-Scale für bessere Sichtbarkeit
        else:
            ax3.text(0.5, 0.5, 'No gradient/update\nmetrics found', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        
        # 4. Loss Landscape Smoothness (approximiert)
        ax4 = axes[1, 0]
        
        loss_metrics = [k for k in valid_metrics.keys() if 'loss' in k.lower()]
        
        if loss_metrics:
            loss_metric = loss_metrics[0]
            loss_values = [v for v in valid_metrics[loss_metric] if not np.isnan(v)]
            
            if len(loss_values) >= 5:
                # Berechne Smoothness als inverse der zweiten Ableitung
                second_derivatives = []
                
                for i in range(2, len(loss_values)):
                    second_deriv = loss_values[i] - 2*loss_values[i-1] + loss_values[i-2]
                    second_derivatives.append(abs(second_deriv))
                
                # Smoothness Plot
                smoothness_steps = steps[2:len(second_derivatives)+2]
                ax4.plot(smoothness_steps, second_derivatives, 'o-', 
                        linewidth=2, markersize=6, alpha=0.8, color='purple')
                
                # Moving Average für Trend
                if len(second_derivatives) >= 3:
                    window = min(3, len(second_derivatives))
                    ma = np.convolve(second_derivatives, np.ones(window)/window, mode='valid')
                    ma_steps = smoothness_steps[window-1:]
                    ax4.plot(ma_steps, ma, '--', alpha=0.7, color='orange', 
                            linewidth=2, label=f'MA({window})')
                
                ax4.set_xlabel('Training Step', fontweight='bold')
                ax4.set_ylabel('Loss Curvature (|2nd Derivative|)', fontweight='bold')
                ax4.set_title('🌊 Loss Landscape Smoothness', fontweight='bold')
                ax4.grid(True, alpha=0.3)
                ax4.set_yscale('log')
                
                # Durchschnittliche Smoothness
                avg_smoothness = np.mean(second_derivatives)
                ax4.axhline(avg_smoothness, color='red', linestyle=':', alpha=0.7,
                           label=f'Avg: {avg_smoothness:.2e}')
                ax4.legend()
        else:
            ax4.text(0.5, 0.5, 'No loss metrics\nfound for smoothness', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        
        # 5. Convergence Analysis
        ax5 = axes[1, 1]
        
        # Wähle Hauptmetrik für Konvergenz-Analyse
        main_metrics = [k for k in valid_metrics.keys() if any(term in k.lower() 
                       for term in ['discrimination_score', 'loss', 'accuracy'])]
        
        if main_metrics:
            main_metric = main_metrics[0]
            values = [v for v in valid_metrics[main_metric] if not np.isnan(v)]
            
            if len(values) >= 5:
                # Konvergenz-Kriterium: Relative Änderung
                convergence_threshold = 0.01  # 1% Änderung
                relative_changes = []
                
                for i in range(1, len(values)):
                    if abs(values[i-1]) > 1e-8:
                        rel_change = abs(values[i] - values[i-1]) / abs(values[i-1])
                    else:
                        rel_change = abs(values[i] - values[i-1])
                    relative_changes.append(rel_change)
                
                change_steps = steps[1:len(relative_changes)+1]
                
                # Plot relative Änderungen
                ax5.plot(change_steps, relative_changes, 'o-', 
                        linewidth=2, markersize=6, alpha=0.8, color='blue')
                
                # Konvergenz-Schwelle
                ax5.axhline(convergence_threshold, color='red', linestyle='--', 
                           alpha=0.7, linewidth=2, label=f'Convergence Threshold ({convergence_threshold:.1%})')
                
                # Konvergenz-Bereiche markieren
                converged_mask = np.array(relative_changes) < convergence_threshold
                if np.any(converged_mask):
                    ax5.fill_between(change_steps, 0, convergence_threshold, 
                                    where=converged_mask, alpha=0.3, color='green',
                                    label='Converged Regions')
                
                ax5.set_xlabel('Training Step', fontweight='bold')
                ax5.set_ylabel('Relative Change', fontweight='bold')
                ax5.set_title(f'🎯 Convergence Analysis\n({main_metric})', fontweight='bold')
                ax5.set_yscale('log')
                ax5.grid(True, alpha=0.3)
                ax5.legend()
                
                # Konvergenz-Statistiken
                converged_pct = np.mean(converged_mask) * 100
                ax5.text(0.02, 0.98, f'Converged: {converged_pct:.1f}%', 
                        transform=ax5.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        
        # 6. Training Efficiency Analysis
        ax6 = axes[1, 2]
        
        # Effizienz = Performance-Verbesserung pro Trainingsschritt
        if main_metrics:
            main_metric = main_metrics[0]
            values = [v for v in valid_metrics[main_metric] if not np.isnan(v)]
            
            if len(values) >= 3:
                # Performance-Verbesserung über Zeit
                improvements = []
                efficiency_window = min(5, len(values) // 3)
                
                for i in range(efficiency_window, len(values)):
                    current_avg = np.mean(values[i-efficiency_window:i])
                    previous_avg = np.mean(values[max(0, i-2*efficiency_window):i-efficiency_window])
                    
                    if abs(previous_avg) > 1e-8:
                        improvement = (current_avg - previous_avg) / abs(previous_avg)
                    else:
                        improvement = current_avg - previous_avg
                    
                    improvements.append(improvement)
                
                if improvements:
                    eff_steps = steps[efficiency_window:efficiency_window+len(improvements)]
                    
                    # Effizienz-Plot
                    colors = ['green' if imp > 0 else 'red' for imp in improvements]
                    bars = ax6.bar(range(len(improvements)), improvements, 
                                  color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
                    
                    ax6.set_xticks(range(0, len(improvements), max(1, len(improvements)//5)))
                    ax6.set_xticklabels([f'{eff_steps[i]:.0f}' for i in range(0, len(improvements), max(1, len(improvements)//5))],
                                       rotation=45)
                    ax6.set_xlabel('Training Step', fontweight='bold')
                    ax6.set_ylabel('Relative Improvement', fontweight='bold')
                    ax6.set_title('⚡ Training Efficiency\n(Performance Improvement Rate)', fontweight='bold')
                    ax6.axhline(0, color='black', linestyle='-', alpha=0.5)
                    ax6.grid(True, alpha=0.3, axis='y')
                    
                    # Durchschnittliche Effizienz
                    avg_efficiency = np.mean(improvements)
                    efficiency_color = 'green' if avg_efficiency > 0 else 'red'
                    ax6.text(0.02, 0.98, f'Avg Efficiency: {avg_efficiency:.3f}', 
                            transform=ax6.transAxes, verticalalignment='top',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor=efficiency_color, alpha=0.3))
        
        plt.tight_layout()
        save_path = os.path.join(self.plots_dir, f'training_dynamics_step_{current_step}.png')
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved training dynamics analysis: {save_path}")

    def _create_detailed_pertubation_analyse(self, combined_agg_results): 
        """Creates a detailed analysis of perturbation effects"""        
        text = ("=== DETAILLIERTE PERTURBATIONS-ANALYSE ===\r\n")

        # Top 5 und Bottom 5 Perturbationen
        top_5 = combined_agg_results.nlargest(5, 'discrimination_score_l1')
        bottom_5 = combined_agg_results.nsmallest(5, 'discrimination_score_l1')

        text +=("\n🏆 TOP 5 PERTURBATIONEN (Höchste Discrimination Score):\n")
        text +=("=" * 70)
        for idx, row in top_5.iterrows():
            text +=(f"📍 {row['perturbation']}")
            text +=(f"   Discrimination Score: {row['discrimination_score_l1']:.4f}\n")
            text +=(f"   MAE: {row['mae']:.4f}\n")
            text +=(f"   Overlap at N: {row['overlap_at_N']:.4f}\n")
            text +=()

        text +=("\n📉 BOTTOM 5 PERTURBATIONEN (Niedrigste Discrimination Score):\n")
        text +=("=" * 70)
        for idx, row in bottom_5.iterrows():
            text +=(f"\n📍 {row['perturbation']}\n")
            text +=(f"   Discrimination Score: {row['discrimination_score_l1']:.4f}\n")
            text +=(f"   MAE: {row['mae']:.4f}\n")
            text +=(f"   Overlap at N: {row['overlap_at_N']:.4f}\n")
            text +=()

        # Korrelationsanalyse
        text +=("\n🔗 KORRELATIONS-ANALYSE:\n")
        text +=("=" * 40)
        correlation_matrix = combined_agg_results[['overlap_at_N', 'mae', 'discrimination_score_l1']].corr()
        text +=(f"\n {correlation_matrix.round(4)}\n")

        # Spezielle Fälle identifizieren
        text +=("\n🎯 SPEZIELLE FÄLLE:\n")
        text +=("=" * 30)

        # Perturbationen mit perfektem Discrimination Score
        perfect_disc = combined_agg_results[combined_agg_results['discrimination_score_l1'] == 1.0]
        text +=(f"\nPerturbationen mit perfektem Discrimination Score (1.0): {len(perfect_disc)}\n")
        if len(perfect_disc) > 0:
            for idx, row in perfect_disc.iterrows():
                text +=(f"  - {row['perturbation']}: MAE={row['mae']:.4f}, Overlap={row['overlap_at_N']:.4f}\n")

        # Perturbationen mit hohem Overlap
        high_overlap = combined_agg_results[combined_agg_results['overlap_at_N'] > 0.01]
        text +=(f"\nPerturbationen mit hohem Overlap (>0.01): {len(high_overlap)}\n")
        if len(high_overlap) > 0:
            for idx, row in high_overlap.iterrows():
                text +=(f"  - {row['perturbation']}: Overlap={row['overlap_at_N']:.4f}, Disc={row['discrimination_score_l1']:.4f}\n")

        # Statistiken
        text +=(f"\n📈 ZUSAMMENFASSUNG:\n")
        text +=(f"Anzahl Perturbationen: {len(combined_agg_results)}\n")
        text +=(f"Durchschnittliche Discrimination Score: {combined_agg_results['discrimination_score_l1'].mean():.4f}\n")
        text +=(f"Durchschnittliche MAE: {combined_agg_results['mae'].mean():.4f}\n")
        text +=(f"Durchschnittlicher Overlap: {combined_agg_results['overlap_at_N'].mean():.4f}\n")

        return text

    def generate_final_report(self, current_step):
        """Generiert einen finalen HTML-Bericht mit allen Analysen"""
        
        report_path = os.path.join(self.plots_dir, f'analysis_report_step_{current_step}.html')
        
        # HTML Template
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Training Analysis Report - Step {step}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                .header {{ background-color: #2c3e50; color: white; padding: 20px; text-align: center; }}
                .section {{ background-color: white; margin: 20px 0; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .plot-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }}
                .plot-item {{ text-align: center; }}
                .plot-item img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }}
                .metrics-table {{ width: 100%; border-collapse: collapse; }}
                .metrics-table th, .metrics-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .metrics-table th {{ background-color: #f2f2f2; }}
                .summary-stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
                .stat-card {{ background-color: #ecf0f1; padding: 15px; border-radius: 6px; text-align: center; }}
                .stat-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
                .stat-label {{ font-size: 14px; color: #7f8c8d; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚀 Training Analysis Report</h1>
                <h2>Step: {step:,} | Generated: {timestamp}</h2>
            </div>
            
            <div class="section">
                <h2>📊 Summary Statistics</h2>
                <div class="summary-stats">
                    {summary_stats}
                </div>
            </div>
            
            <div class="section">
                <h2>📈 Generated Visualizations</h2>
                <div class="plot-grid">
                    {plot_images}
                </div>
            </div>
            
            <div class="section">
                <h2>🔍 Pertubation Insights</h2>
                <ul>
                    {pert_insights}
                </ul>
            </div>

            <div class="section">
                <h2>📋 Latest Metrics</h2>
                {metrics_table}
            </div>
            
            <div class="section">
                <h2>🔍 Key Insights</h2>
                <ul>
                    {key_insights}
                </ul>
            </div>
        </body>
        </html>
        """
        #print(f"Type of self.agg_results_history: {type(self.agg_results_history)}")
        #pert_insights_html = self._generate_problem_analysis_content(self.agg_results_history)
        pert_insights_html = "Not implemented yet"
        # Sammle Daten für den Bericht
        timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Summary Statistics
        summary_stats_html = ""
        #if self.all_metrics_history:
        latest_metrics = self.all_metrics_history[-1]
        
        # Wichtige Statistiken
        stats = [
            ("Total Evaluations", len(self.all_metrics_history)),
            ("Current Step", current_step),
            ("Plots Generated", len(list(self.plots_dir.glob(f'*step_{current_step}.png')))),
        ]
        
        # Füge wichtige Metriken hinzu
        for key, value in latest_metrics.items():
            if any(term in key.lower() for term in ['discrimination_score', 'mae', 'overlap_at_N', 'avg_score']):
                try:
                    stats.append((key.replace('_', ' ').title(), f"{float(value):.4f}"))
                except (ValueError, TypeError):
                    pass
        
        for label, value in stats:
            summary_stats_html += f"""
            <div class="stat-card">
                <div class="stat-value">{value}</div>
                <div class="stat-label">{label}</div>
            </div>
            """
    
        # Plot Images
        plot_images_html = ""
        plot_files = sorted(self.plots_dir.glob(f'*step_{current_step}.png'))
        
        for plot_file in plot_files:
            plot_name = plot_file.stem.replace(f'_step_{current_step}', '').replace('_', ' ').title()
            plot_images_html += f"""
            <div class="plot-item">
                <h3>{plot_name}</h3>
                <img src="{plot_file.name}" alt="{plot_name}">
            </div>
            """
        
        # Metrics Table
        #metrics_table_html = "<p>No metrics available</p>"
        
        #if self.all_metrics_history:
        latest_metrics = self.all_metrics_history[-1]
        
        metrics_table_html = '<table class="metrics-table"><tr><th>Metric</th><th>Value</th></tr>'
        for key, value in latest_metrics.items():
            metrics_table_html += f'<tr><td>{key}</td><td>{value}</td></tr>'
        metrics_table_html += '</table>'
        
        # Key Insights
        key_insights_html = ""
        insights = self._generate_key_insights(current_step)
        for insight in insights:
            key_insights_html += f"<li>{insight}</li>"
        
        # HTML generieren
        html_content = html_template.format(
            step=current_step,
            timestamp=timestamp,
            summary_stats=summary_stats_html,
            plot_images=plot_images_html,
            metrics_table=metrics_table_html,
            key_insights=key_insights_html,
            pert_insights=pert_insights_html,
            )
        
        # Speichere HTML-Bericht
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Generated final HTML report: {report_path}")
        return report_path

    def _generate_key_insights(self, current_step):
        """Generiert automatische Key Insights basierend auf den Daten"""
        
        insights = []
        
        if not self.all_metrics_history:
            return ["No training data available for insights generation."]
        
        # Aktuelle vs. vorherige Performance
        if len(self.all_metrics_history) >= 2:
            current_metrics = self.all_metrics_history[-1]
            previous_metrics = self.all_metrics_history[-2]
            
            for key in current_metrics:
                if key in previous_metrics:
                    try:
                        current_val = float(current_metrics[key])
                        previous_val = float(previous_metrics[key])
                        
                        if abs(previous_val) > 1e-8:
                            change_pct = ((current_val - previous_val) / abs(previous_val)) * 100
                            
                            if abs(change_pct) > 5:  # Signifikante Änderung
                                direction = "improved" if change_pct > 0 else "decreased"
                                insights.append(f"{key.replace('_', ' ').title()} {direction} by {abs(change_pct):.1f}% since last evaluation")
                    except (ValueError, TypeError):
                        continue
        
        # Training Stabilität
        if len(self.all_metrics_history) >= 5:
            for key in self.all_metrics_history[-1]:
                values = []
                for metrics_dict in self.all_metrics_history[-5:]:
                    if key in metrics_dict:
                        try:
                            values.append(float(metrics_dict[key]))
                        except (ValueError, TypeError):
                            continue
                
                if len(values) >= 3:
                    cv = np.std(values) / (abs(np.mean(values)) + 1e-8)
                    if cv < 0.1:
                        insights.append(f"{key.replace('_', ' ').title()} shows high stability (CV: {cv:.3f})")
                    elif cv > 0.5:
                        insights.append(f"{key.replace('_', ' ').title()} shows high volatility (CV: {cv:.3f})")
        
        # Trend-Analyse
        if len(self.all_metrics_history) >= 3:
            for key in self.all_metrics_history[-1]:
                values = []
                for metrics_dict in self.all_metrics_history:
                    if key in metrics_dict:
                        try:
                            values.append(float(metrics_dict[key]))
                        except (ValueError, TypeError):
                            continue
                
                if len(values) >= 3:
                    # Einfache Trend-Erkennung
                    recent_trend = np.polyfit(range(len(values)//2, len(values)), 
                                            values[len(values)//2:], 1)[0]
                    
                    if abs(recent_trend) > 1e-6:
                        trend_direction = "upward" if recent_trend > 0 else "downward"
                        insights.append(f"{key.replace('_', ' ').title()} shows {trend_direction} trend (slope: {recent_trend:.6f})")
        
        # Fallback falls keine Insights gefunden
        if not insights:
            insights.append("Training is progressing. Continue monitoring for more detailed insights.")
            insights.append(f"Current training step: {current_step:,}")
            insights.append(f"Total evaluations completed: {len(self.all_metrics_history)}")
        
        return insights[:10]  # Maximal 10 Insights 
    
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

    def _create_metric_volatility_analysis(self, df_history, current_step):
        """Analyzes the volatility of different metrics over time"""
        
        if df_history is None or len(df_history) < 5:
            logger.warning("Not enough data for volatility analysis")
            return
        
        # Identify numeric columns for volatility analysis
        numeric_columns = df_history.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove step-related columns
        metric_columns = [col for col in numeric_columns 
                        if not any(term in col.lower() for term in ['step', 'epoch', 'iteration'])]
        
        if len(metric_columns) < 2:
            logger.warning("Not enough metrics for volatility analysis")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'📊 Metric Volatility Analysis (Step: {current_step:,})', 
                    fontsize=16, fontweight='bold')
        
        # 1. Rolling Volatility Heatmap
        ax1 = axes[0, 0]
        
        # Calculate Rolling Volatility (Rolling Standard Deviation)
        window_size = min(10, len(df_history) // 3)
        volatility_data = {}
        
        for col in metric_columns[:10]:  # Limit to 10 metrics for better visualization
            if col in df_history.columns:
                values = df_history[col].dropna()
                if len(values) >= window_size:
                    rolling_vol = values.rolling(window=window_size, min_periods=1).std()
                    volatility_data[col] = rolling_vol
        
        if volatility_data:
            # Create DataFrame for heatmap
            vol_df = pd.DataFrame(volatility_data)
            
            # Normalize for better visualization
            vol_df_normalized = (vol_df - vol_df.min()) / (vol_df.max() - vol_df.min())
            vol_df_normalized = vol_df_normalized.fillna(0)
            
            # Heatmap
            im = ax1.imshow(vol_df_normalized.T, cmap='YlOrRd', aspect='auto', 
                        interpolation='nearest')
            
            # Configure axes
            ax1.set_yticks(range(len(vol_df_normalized.columns)))
            ax1.set_yticklabels([col.replace('_', '\n')[:15] for col in vol_df_normalized.columns], 
                            fontsize=9)
            
            # X-axis: Show only every nth step
            n_ticks = min(10, len(vol_df_normalized))
            tick_indices = np.linspace(0, len(vol_df_normalized)-1, n_ticks, dtype=int)
            ax1.set_xticks(tick_indices)
            ax1.set_xticklabels([f'{i}' for i in tick_indices], rotation=45)
            
            ax1.set_xlabel('Time Steps', fontweight='bold')
            ax1.set_ylabel('Metrics', fontweight='bold')
            ax1.set_title('🔥 Rolling Volatility Heatmap\n(Normalized)', fontweight='bold')
            
            # Colorbar
            cbar = plt.colorbar(im, ax=ax1, shrink=0.8)
            cbar.set_label('Normalized Volatility', fontsize=9)
        
        # 2. Volatility Ranking
        ax2 = axes[0, 1]
        
        # Calculate average volatility for each metric
        avg_volatilities = {}
        
        for col in metric_columns:
            if col in df_history.columns:
                values = df_history[col].dropna()
                if len(values) >= 3:
                    # Coefficient of Variation as volatility measure
                    cv = values.std() / (abs(values.mean()) + 1e-8)
                    avg_volatilities[col] = cv
        
        if avg_volatilities:
            # Sort by volatility
            sorted_volatilities = sorted(avg_volatilities.items(), 
                                    key=lambda x: x[1], reverse=True)[:12]
            
            names = [name.replace('_', '\n')[:15] for name, _ in sorted_volatilities]
            volatilities = [vol for _, vol in sorted_volatilities]
            
            # Color coding based on volatility
            colors = plt.cm.Reds(np.linspace(0.3, 1.0, len(volatilities)))
            
            bars = ax2.barh(range(len(names)), volatilities, color=colors, 
                        alpha=0.8, edgecolor='black', linewidth=0.5)
            
            ax2.set_yticks(range(len(names)))
            ax2.set_yticklabels(names, fontsize=9)
            ax2.set_xlabel('Coefficient of Variation', fontweight='bold')
            ax2.set_title('📈 Volatility Ranking\n(Higher = More Volatile)', fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='x')
            
            # Values on bars
            for i, (bar, vol) in enumerate(zip(bars, volatilities)):
                ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{vol:.3f}', ha='left', va='center', fontsize=8, fontweight='bold')
            
            # Mark volatility categories
            if volatilities:
                high_vol_threshold = np.percentile(volatilities, 75)
                low_vol_threshold = np.percentile(volatilities, 25)
                
                ax2.axvline(high_vol_threshold, color='red', linestyle='--', alpha=0.7,
                        label=f'High Vol. (>{high_vol_threshold:.3f})')
                ax2.axvline(low_vol_threshold, color='green', linestyle='--', alpha=0.7,
                        label=f'Low Vol. (<{low_vol_threshold:.3f})')
                ax2.legend(fontsize=8)
        
        # 3. Volatility vs Performance Scatter
        ax3 = axes[1, 0]
        
        # Search for performance metrics
        perf_metrics = [col for col in metric_columns 
                    if any(term in col.lower() for term in 
                            ['score', 'accuracy', 'loss', 'error', 'mae', 'mse'])]
        
        if perf_metrics and avg_volatilities:
            # Choose best performance metric
            perf_metric = perf_metrics[0]
            
            if perf_metric in df_history.columns:
                perf_values = df_history[perf_metric].dropna()
                
                if len(perf_values) > 0:
                    # Collect data for scatter plot
                    x_volatilities = []
                    y_performances = []
                    labels = []
                    
                    for metric_name, volatility in avg_volatilities.items():
                        if metric_name in df_history.columns and metric_name != perf_metric:
                            # Use last value as performance proxy
                            last_perf = perf_values.iloc[-1] if len(perf_values) > 0 else 0
                            
                            x_volatilities.append(volatility)
                            y_performances.append(last_perf)
                            labels.append(metric_name)
                    
                    if x_volatilities and y_performances:
                        # Scatter plot
                        scatter = ax3.scatter(x_volatilities, y_performances, 
                                            alpha=0.7, s=80, c=range(len(x_volatilities)),
                                            cmap='viridis', edgecolors='black', linewidth=0.5)
                        
                        # Labels for interesting points
                        for i, (x, y, label) in enumerate(zip(x_volatilities, y_performances, labels)):
                            if i < 5:  # Only first 5 labels
                                ax3.annotate(label[:10], (x, y), xytext=(5, 5), 
                                        textcoords='offset points', fontsize=8,
                                        bbox=dict(boxstyle='round,pad=0.3', 
                                                facecolor='yellow', alpha=0.7))
                        
                        # Correlation line
                        if len(x_volatilities) > 2:
                            z = np.polyfit(x_volatilities, y_performances, 1)
                            p = np.poly1d(z)
                            x_line = np.linspace(min(x_volatilities), max(x_volatilities), 100)
                            ax3.plot(x_line, p(x_line), 'r--', alpha=0.8, linewidth=2)
                            
                            # Calculate correlation
                            correlation = np.corrcoef(x_volatilities, y_performances)[0, 1]
                            ax3.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                                transform=ax3.transAxes, fontsize=10,
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                        
                        ax3.set_xlabel('Metric Volatility (CV)', fontweight='bold')
                        ax3.set_ylabel(f'{perf_metric.replace("_", " ").title()}', fontweight='bold')
                        ax3.set_title('🎯 Volatility vs Performance\n(Current Values)', fontweight='bold')
                        ax3.grid(True, alpha=0.3)
        
        # 4. Volatility Time Series for Top Metrics
        ax4 = axes[1, 1]
        
        if volatility_data:
            # Choose top 5 most volatile metrics
            top_volatile = sorted(avg_volatilities.items(), key=lambda x: x[1], reverse=True)[:5]
            
            colors = plt.cm.tab10(np.linspace(0, 1, len(top_volatile)))
            
            for i, (metric_name, _) in enumerate(top_volatile):
                if metric_name in volatility_data:
                    rolling_vol = volatility_data[metric_name]
                    
                    # Plot rolling volatility
                    ax4.plot(range(len(rolling_vol)), rolling_vol, 'o-', 
                            color=colors[i], linewidth=2, markersize=4, alpha=0.8,
                            label=metric_name.replace('_', ' ')[:15])
                    
                    # Trend line
                    if len(rolling_vol) > 3:
                        z = np.polyfit(range(len(rolling_vol)), rolling_vol, 1)
                        trend_line = np.poly1d(z)
                        ax4.plot(range(len(rolling_vol)), trend_line(range(len(rolling_vol))), 
                                '--', color=colors[i], alpha=0.5, linewidth=1.5)
            
            ax4.set_xlabel('Time Steps', fontweight='bold')
            ax4.set_ylabel('Rolling Volatility', fontweight='bold')
            ax4.set_title(f'📊 Volatility Evolution\n(Top 5 Most Volatile)', fontweight='bold')
            ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.plots_dir, f'metric_volatility_analysis_step_{current_step}.png')
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved metric volatility analysis: {save_path}")

    def _create_pareto_frontier_analysis(self, df_history, current_step):
        """Creates Pareto frontier analysis for multi-objective optimization"""
        
        if df_history is None or len(df_history) < 3:
            logger.warning("Not enough data for Pareto frontier analysis")
            return
        
        # Identify numeric metrics
        numeric_columns = df_history.select_dtypes(include=[np.number]).columns.tolist()
        metric_columns = [col for col in numeric_columns 
                        if not any(term in col.lower() for term in ['step', 'epoch', 'iteration'])]
        
        if len(metric_columns) < 2:
            logger.warning("Need at least 2 metrics for Pareto frontier analysis")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'🎯 Pareto Frontier Analysis (Step: {current_step:,})', 
                    fontsize=16, fontweight='bold')
        
        # 1. 2D Pareto Frontier (best 2 metrics)
        ax1 = axes[0, 0]
        
        # Choose 2 most important metrics (based on variability)
        metric_importance = {}
        for col in metric_columns:
            if col in df_history.columns:
                values = df_history[col].dropna()
                if len(values) > 1:
                    # Combine variability and trend as importance
                    variability = values.std() / (abs(values.mean()) + 1e-8)
                    trend = abs(np.polyfit(range(len(values)), values, 1)[0]) if len(values) > 2 else 0
                    importance = variability + trend
                    metric_importance[col] = importance
        
        if len(metric_importance) >= 2:
            # Top 2 metrics
            top_metrics = sorted(metric_importance.items(), key=lambda x: x[1], reverse=True)[:2]
            metric1, metric2 = top_metrics[0][0], top_metrics[1][0]
            
            # Data for Pareto analysis
            data1 = df_history[metric1].dropna()
            data2 = df_history[metric2].dropna()
            
            # Ensure both series have same length
            min_len = min(len(data1), len(data2))
            if min_len > 0:
                data1 = data1.iloc[:min_len]
                data2 = data2.iloc[:min_len]
                
                # Calculate Pareto frontier
                pareto_points = []
                pareto_indices = []
                
                for i in range(len(data1)):
                    is_pareto = True
                    for j in range(len(data1)):
                        if i != j:
                            # Assumption: Higher values are better for both metrics
                            # (can be adjusted depending on metric)
                            if (data1.iloc[j] >= data1.iloc[i] and data2.iloc[j] >= data2.iloc[i] and
                                (data1.iloc[j] > data1.iloc[i] or data2.iloc[j] > data2.iloc[i])):
                                is_pareto = False
                                break
                    
                    if is_pareto:
                        pareto_points.append((data1.iloc[i], data2.iloc[i]))
                        pareto_indices.append(i)
                
                # Plot all points
                ax1.scatter(data1, data2, alpha=0.6, s=50, c='lightblue', 
                        edgecolors='black', linewidth=0.5, label='All Points')
                
                # Highlight Pareto points
                if pareto_points:
                    pareto_x, pareto_y = zip(*pareto_points)
                    ax1.scatter(pareto_x, pareto_y, alpha=0.9, s=100, c='red', 
                            marker='*', edgecolors='darkred', linewidth=1.5,
                            label=f'Pareto Frontier ({len(pareto_points)} points)')
                    
                    # Connect Pareto points
                    sorted_pareto = sorted(pareto_points, key=lambda x: x[0])
                    if len(sorted_pareto) > 1:
                        pareto_x_sorted, pareto_y_sorted = zip(*sorted_pareto)
                        ax1.plot(pareto_x_sorted, pareto_y_sorted, 'r--', alpha=0.7, 
                                linewidth=2, label='Pareto Line')
                
                # Current point
                current_x, current_y = data1.iloc[-1], data2.iloc[-1]
                ax1.scatter(current_x, current_y, alpha=1.0, s=150, c='gold', 
                        marker='D', edgecolors='black', linewidth=2,
                        label='Current Point')
                
                ax1.set_xlabel(metric1.replace('_', ' ').title(), fontweight='bold')
                ax1.set_ylabel(metric2.replace('_', ' ').title(), fontweight='bold')
                ax1.set_title(f'🎯 2D Pareto Frontier\n{metric1} vs {metric2}', fontweight='bold')
                ax1.legend(fontsize=9)
                ax1.grid(True, alpha=0.3)
                
                # Pareto efficiency of current point
                is_current_pareto = len(pareto_indices) > 0 and (len(data1) - 1) in pareto_indices
                efficiency_text = "✅ Pareto Efficient" if is_current_pareto else "❌ Not Pareto Efficient"
                ax1.text(0.02, 0.98, efficiency_text, transform=ax1.transAxes, 
                        verticalalignment='top', fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='lightgreen' if is_current_pareto else 'lightcoral', 
                                alpha=0.8))
        
        # 2. Pareto Frontier Evolution Over Time
        ax2 = axes[0, 1]
        
        if len(metric_importance) >= 2:
            # Calculate Pareto efficiency over time
            window_size = min(20, len(df_history) // 2)
            pareto_efficiency_over_time = []
            time_steps = []
            
            for i in range(window_size, len(df_history)):
                window_data1 = data1.iloc[i-window_size:i]
                window_data2 = data2.iloc[i-window_size:i]
                
                # Pareto points in this window
                window_pareto_count = 0
                for j in range(len(window_data1)):
                    is_pareto = True
                    for k in range(len(window_data1)):
                        if j != k:
                            if (window_data1.iloc[k] >= window_data1.iloc[j] and 
                                window_data2.iloc[k] >= window_data2.iloc[j] and
                                (window_data1.iloc[k] > window_data1.iloc[j] or 
                                window_data2.iloc[k] > window_data2.iloc[j])):
                                is_pareto = False
                                break
                    if is_pareto:
                        window_pareto_count += 1
                
                pareto_ratio = window_pareto_count / len(window_data1)
                pareto_efficiency_over_time.append(pareto_ratio)
                time_steps.append(i)
            
            if pareto_efficiency_over_time:
                ax2.plot(time_steps, pareto_efficiency_over_time, 'o-', 
                        linewidth=3, markersize=6, alpha=0.8, color='purple')
                
                # Trend line
                if len(pareto_efficiency_over_time) > 2:
                    z = np.polyfit(time_steps, pareto_efficiency_over_time, 1)
                    trend_line = np.poly1d(z)
                    ax2.plot(time_steps, trend_line(time_steps), '--', 
                            alpha=0.7, color='orange', linewidth=2, label='Trend')
                    
                    trend_slope = z[0]
                    trend_text = f"Trend: {'↗' if trend_slope > 0 else '↘'} {trend_slope:.4f}"
                    ax2.text(0.02, 0.98, trend_text, transform=ax2.transAxes, 
                            verticalalignment='top', fontsize=10,
                            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
                
                ax2.set_xlabel('Time Steps', fontweight='bold')
                ax2.set_ylabel('Pareto Efficiency Ratio', fontweight='bold')
                ax2.set_title(f'📈 Pareto Efficiency Evolution\n(Window Size: {window_size})', fontweight='bold')
                ax2.grid(True, alpha=0.3)
                ax2.set_ylim(0, 1)
        
        # 3. Multi-Objective Trade-off Analysis
        ax3 = axes[1, 0]
        
        if len(metric_columns) >= 3:
            # Use top 3 metrics for trade-off analysis
            top_3_metrics = sorted(metric_importance.items(), key=lambda x: x[1], reverse=True)[:3]
            
            # Normalize metrics for better comparison
            normalized_data = {}
            for metric_name, _ in top_3_metrics:
                values = df_history[metric_name].dropna()
                if len(values) > 0:
                    # Min-Max normalization
                    normalized = (values - values.min()) / (values.max() - values.min() + 1e-8)
                    normalized_data[metric_name] = normalized
            
            if len(normalized_data) >= 3:
                # Parallel Coordinates Plot
                metric_names = list(normalized_data.keys())
                
                # Create data matrix
                data_matrix = []
                min_len = min(len(data) for data in normalized_data.values())
                
                for i in range(min_len):
                    row = [normalized_data[metric].iloc[i] for metric in metric_names]
                    data_matrix.append(row)
                
                # Plot Parallel Coordinates
                for i, row in enumerate(data_matrix):
                    alpha = 0.1 if i < len(data_matrix) - 1 else 1.0  # Highlight last point
                    color = 'gray' if i < len(data_matrix) - 1 else 'red'
                    linewidth = 0.5 if i < len(data_matrix) - 1 else 3
                    
                    ax3.plot(range(len(metric_names)), row, color=color, 
                            alpha=alpha, linewidth=linewidth)
                
                # Highlight last values
                if data_matrix:
                    last_row = data_matrix[-1]
                    ax3.plot(range(len(metric_names)), last_row, 'ro-', 
                            linewidth=3, markersize=8, label='Current State')
                
                ax3.set_xticks(range(len(metric_names)))
                ax3.set_xticklabels([name.replace('_', '\n')[:10] for name in metric_names], 
                                rotation=45, ha='right', fontsize=9)
                ax3.set_ylabel('Normalized Value', fontweight='bold')
                ax3.set_title('🔄 Multi-Objective Trade-offs\n(Parallel Coordinates)', fontweight='bold')
                ax3.grid(True, alpha=0.3, axis='y')
                ax3.legend()
                ax3.set_ylim(0, 1)
        
        # 4. Hypervolume Indicator (approximated)
        ax4 = axes[1, 1]
        
        if len(metric_importance) >= 2:
            # Calculate approximated hypervolume over time
            hypervolumes = []
            time_points = []
            
            # Use sliding windows
            window_size = min(10, len(data1) // 3)
            
            for i in range(window_size, len(data1)):
                window_data1 = data1.iloc[i-window_size:i]
                window_data2 = data2.iloc[i-window_size:i]
                
                # Simple hypervolume approximation (area under Pareto curve)
                # Normalize data for this window
                norm_data1 = (window_data1 - window_data1.min()) / (window_data1.max() - window_data1.min() + 1e-8)
                norm_data2 = (window_data2 - window_data2.min()) / (window_data2.max() - window_data2.min() + 1e-8)
                
                # Find Pareto points
                pareto_points_norm = []
                for j in range(len(norm_data1)):
                    is_pareto = True
                    for k in range(len(norm_data1)):
                        if j != k:
                            if (norm_data1.iloc[k] >= norm_data1.iloc[j] and 
                                norm_data2.iloc[k] >= norm_data2.iloc[j] and
                                (norm_data1.iloc[k] > norm_data1.iloc[j] or 
                                norm_data2.iloc[k] > norm_data2.iloc[j])):
                                is_pareto = False
                                break
                    if is_pareto:
                        pareto_points_norm.append((norm_data1.iloc[j], norm_data2.iloc[j]))
                
                # Approximate hypervolume as sum of coordinates of Pareto points
                if pareto_points_norm:
                    hypervolume = sum(x * y for x, y in pareto_points_norm) / len(pareto_points_norm)
                else:
                    hypervolume = 0
                
                hypervolumes.append(hypervolume)
                time_points.append(i)
            
            if hypervolumes:
                # Plot hypervolume evolution
                ax4.plot(time_points, hypervolumes, 'o-', linewidth=3, markersize=6, 
                        alpha=0.8, color='darkgreen')
                
                # Trend analysis
                if len(hypervolumes) > 2:
                    z = np.polyfit(time_points, hypervolumes, 1)
                    trend_line = np.poly1d(z)
                    ax4.plot(time_points, trend_line(time_points), '--', 
                            alpha=0.7, color='orange', linewidth=2, label='Trend')
                    
                    # R² for trend quality
                    r_squared = np.corrcoef(time_points, hypervolumes)[0, 1]**2
                    ax4.text(0.02, 0.98, f'R² = {r_squared:.3f}', transform=ax4.transAxes, 
                            verticalalignment='top', fontsize=10,
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                # Mark best hypervolume
                best_hv_idx = np.argmax(hypervolumes)
                best_hv = hypervolumes[best_hv_idx]
                best_time = time_points[best_hv_idx]
                
                ax4.scatter(best_time, best_hv, color='gold', s=150, marker='*', 
                        edgecolors='black', linewidth=2, zorder=5,
                        label=f'Best HV: {best_hv:.4f}')
                
                ax4.set_xlabel('Time Steps', fontweight='bold')
                ax4.set_ylabel('Hypervolume Indicator', fontweight='bold')
                ax4.set_title('📊 Hypervolume Evolution\n(Multi-Objective Performance)', fontweight='bold')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.plots_dir, f'pareto_frontier_analysis_step_{current_step}.png')
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved Pareto frontier analysis: {save_path}")

    def _generate_problem_analysis_content(data, metrics=['overlap_at_N', 'mae', 'discrimination_score_l1', 'avg_score']):
        """
        Generates HTML content for perturbation data structure problem analysis.
        Returns only the content part (without full HTML structure) for embedding.
        
        Parameters:
        -----------
        data : List[dict] or pd.DataFrame
            Data to analyze - can be list of dictionaries or DataFrame
        metrics : list
            List of metrics to analyze
            
        Returns:
        --------
        str: HTML content ready for embedding
        """
        
        # Convert List[dict] to DataFrame if needed
        #if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        df = pd.concat(data.agg_results_history, ignore_index=True)
        print(f"Data type for problem analysis: {type(df)}")
        print(f"Dan data: {df}")
        #elif hasattr(data, 'to_dict'):  # Handle other data structures that can convert to dict
        #    df = pd.DataFrame(data.to_dict())
        #else:
        #    df = data  # Assume it's already a DataFrame
        
        # Get available metrics from the actual data
        available_metrics = [metric for metric in metrics if metric in df.columns]
        all_columns = list(df.columns)
        
        html_content = """
                <div style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6;">
                    <h1 style="color: #2c3e50; text-align: center; border-bottom: 3px solid #3498db; padding-bottom: 10px;">
                        🔬 Perturbation Data Analysis
                    </h1>
                    
                    <div style="background-color: #fff5f5; border: 1px solid #fed7d7; border-radius: 8px; padding: 20px; margin: 20px 0;">
                        <h2 style="color: #34495e; border-left: 4px solid #3498db; padding-left: 15px; margin-top: 0;">
                            <span style="font-size: 1.2em; margin-right: 8px;">🚨</span>Problem Analysis
                        </h2>
                        <h3>Your code expects:</h3>
                        <ul style="padding-left: 20px;">
                            <li style="margin: 8px 0;">A column <span style="background-color: #fff3cd; padding: 2px 6px; border-radius: 4px; font-weight: bold;">'global_step'</span> (training steps)</li>
                            <li style="margin: 8px 0;">Temporal evolution of metrics</li>
                            <li style="margin: 8px 0;">Multiple time points for trend analysis</li>
                        </ul>
                    </div>
                    
                    <div style="background-color: #f0fff4; border: 1px solid #c6f6d5; border-radius: 8px; padding: 20px; margin: 20px 0;">
                        <h2 style="color: #34495e; border-left: 4px solid #3498db; padding-left: 15px; margin-top: 0;">
                            <span style="font-size: 1.2em; margin-right: 8px;">📊</span>Your data contains:
                        </h2>
                        <ul style="padding-left: 20px;">
                            <li style="margin: 8px 0;">Static perturbation results</li>
                            <li style="margin: 8px 0;"><span style="background-color: #fff3cd; padding: 2px 6px; border-radius: 4px; font-weight: bold;">{num_samples}</span> different genes/proteins</li>
                            <li style="margin: 8px 0;">No temporal dimension</li>
                            <li style="margin: 8px 0;">Available columns: <span style="background-color: #e3f2fd; padding: 2px 6px; border-radius: 4px; font-family: monospace;">{available_columns}</span></li>
                        </ul>
                    </div>
                    
                    <div style="background-color: #fffbf0; border: 1px solid #fbd38d; border-radius: 8px; padding: 20px; margin: 20px 0;">
                        <h2 style="color: #34495e; border-left: 4px solid #3498db; padding-left: 15px; margin-top: 0;">
                            <span style="font-size: 1.2em; margin-right: 8px;">💡</span>What's likely happening:
                        </h2>
                        <ul style="padding-left: 20px;">
                            <li style="margin: 8px 0;">Code tries to find <span style="background-color: #fff3cd; padding: 2px 6px; border-radius: 4px; font-weight: bold;">'global_step'</span> → not present</li>
                            <li style="margin: 8px 0;">Possibly using index as 'steps'</li>
                            <li style="margin: 8px 0;">Shows random/incorrect temporal evolution</li>
                        </ul>
                    </div>
                    
                    <h2 style="color: #34495e; border-left: 4px solid #3498db; padding-left: 15px; margin-top: 30px;">
                        <span style="font-size: 1.2em; margin-right: 8px;">📈</span>Actual Data Distribution
                    </h2>
        """
        
        # Only show metrics table if we have available metrics
        if available_metrics:
            html_content += """
                    <table style="width: 100%; border-collapse: collapse; margin: 20px 0; background-color: white; border-radius: 8px; overflow: hidden; box-shadow: 0 0 10px rgba(0,0,0,0.1);">
                        <thead>
                            <tr>
                                <th style="background-color: #3498db; color: white; padding: 12px; text-align: left;">Metric</th>
                                <th style="background-color: #3498db; color: white; padding: 12px; text-align: left;">Minimum</th>
                                <th style="background-color: #3498db; color: white; padding: 12px; text-align: left;">Maximum</th>
                                <th style="background-color: #3498db; color: white; padding: 12px; text-align: left;">Mean</th>
                                <th style="background-color: #3498db; color: white; padding: 12px; text-align: left;">Unique Values</th>
                            </tr>
                        </thead>
                        <tbody>
            """
            
            # Fill metrics table
            row_counter = 0
            for metric in available_metrics:
                min_val = df[metric].min()
                max_val = df[metric].max()
                mean_val = df[metric].mean()
                unique_count = df[metric].nunique()
                
                bg_color = "#f8f9fa" if row_counter % 2 == 1 else "white"
                
                html_content += f"""
                                <tr style="background-color: {bg_color};">
                                    <td style="padding: 12px; border-bottom: 1px solid #ecf0f1;"><strong>{metric}</strong></td>
                                    <td style="padding: 12px; border-bottom: 1px solid #ecf0f1;">{min_val:.4f}</td>
                                    <td style="padding: 12px; border-bottom: 1px solid #ecf0f1;">{max_val:.4f}</td>
                                    <td style="padding: 12px; border-bottom: 1px solid #ecf0f1;">{mean_val:.4f}</td>
                                    <td style="padding: 12px; border-bottom: 1px solid #ecf0f1;">{unique_count}</td>
                                </tr>
                """
                row_counter += 1
            
            html_content += """
                        </tbody>
                    </table>
            """
        else:
            html_content += """
                    <div style="background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px; padding: 20px; margin: 20px 0;">
                        <p><strong>⚠️ Warning:</strong> None of the expected metrics ({expected_metrics}) were found in your data.</p>
                        <p><strong>Available columns:</strong> {available_columns}</p>
                    </div>
            """.format(
                expected_metrics=', '.join(metrics),
                available_columns=', '.join(all_columns)
            )
        
        # Complete HTML content
        html_content += """
                    <div style="background-color: #fffbf0; border: 1px solid #fbd38d; border-radius: 8px; padding: 20px; margin: 20px 0;">
                        <h2 style="color: #34495e; border-left: 4px solid #3498db; padding-left: 15px; margin-top: 0;">
                            <span style="font-size: 1.2em; margin-right: 8px;">🎯</span>Recommendations
                        </h2>
                        <ul style="padding-left: 20px;">
                            <li style="margin: 8px 0;"><strong>Adjust data structure:</strong> Create a 'global_step' column if temporal analysis is desired</li>
                            <li style="margin: 8px 0;"><strong>Static analysis:</strong> Use perturbation names instead of time steps for X-axis</li>
                            <li style="margin: 8px 0;"><strong>Ranking-based:</strong> Sort by performance metrics for better visualization</li>
                            <li style="margin: 8px 0;"><strong>Categorization:</strong> Group similar perturbations for clearer representation</li>
                            <li style="margin: 8px 0;"><strong>Data conversion:</strong> If using List[dict], convert to DataFrame first: <code style="background-color: #f8f9fa; padding: 2px 4px; border-radius: 3px;">pd.DataFrame(your_list)</code></li>
                        </ul>
                    </div>
                    
                    <div style="text-align: center; margin-top: 30px; padding: 20px; background-color: #ecf0f1; border-radius: 8px;">
                        <p><strong>Generated on:</strong> {timestamp}</p>
                        <p><em>Automated perturbation data structure analysis</em></p>
                    </div>
                </div>
        """
        
        # Replace placeholders
        html_content = html_content.format(
            num_samples=len(df),
            available_columns=', '.join(all_columns[:10]) + ('...' if len(all_columns) > 10 else ''),
            timestamp=datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S")
        )
        
        return html_content


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, Any
import os
import logging

logger = logging.getLogger(__name__)


class ModelVisualizer:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        plt.style.use('default')

    def plot_model_comparison(self, model_results: Dict[str, Dict[str, Any]]):
        models = list(model_results.keys())
        train_accs = [results['train_accuracy'] for results in model_results.values()]
        val_accs = [results['val_accuracy'] for results in model_results.values()]
        cv_scores = [results['cv_score'] for results in model_results.values()]
        gaps = [results['overfitting_gap'] for results in model_results.values()]


        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')


        x = np.arange(len(models))
        width = 0.35

        ax1.bar(x - width / 2, train_accs, width, label='Training', alpha=0.8, color='skyblue')
        ax1.bar(x + width / 2, val_accs, width, label='Validation', alpha=0.8, color='lightcoral')

        ax1.set_xlabel('Models')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Training vs Validation Accuracy')
        ax1.set_xticks(x)
        ax1.set_xticklabels([m.upper() for m in models], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        for i, (train_acc, val_acc) in enumerate(zip(train_accs, val_accs)):
            ax1.text(i - width / 2, train_acc + 0.01, f'{train_acc:.3f}',
                     ha='center', va='bottom', fontsize=8)
            ax1.text(i + width / 2, val_acc + 0.01, f'{val_acc:.3f}',
                     ha='center', va='bottom', fontsize=8)


        colors = ['green' if gap < 0.03 else 'orange' for gap in gaps]
        bars = ax2.bar(models, cv_scores, alpha=0.8, color=colors)
        ax2.set_xlabel('Models')
        ax2.set_ylabel('CV Score')
        ax2.set_title('Cross-Validation Scores')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)

        for bar, score in zip(bars, cv_scores):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f'{score:.3f}', ha='center', va='bottom', fontsize=9)


        colors = ['green' if gap < 0.03 else 'red' for gap in gaps]
        bars = ax3.bar(models, gaps, alpha=0.8, color=colors)
        ax3.axhline(y=0.03, color='red', linestyle='--', alpha=0.7, label='Overfitting Threshold')
        ax3.set_xlabel('Models')
        ax3.set_ylabel('Overfitting Gap')
        ax3.set_title('Overfitting Analysis (Train - Val Accuracy)')
        ax3.tick_params(axis='x', rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        for bar, gap in zip(bars, gaps):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                     f'{gap:.3f}', ha='center', va='bottom', fontsize=9)


        df_metrics = pd.DataFrame({
            'Model': [m.upper() for m in models],
            'CV Score': cv_scores,
            'Val Accuracy': val_accs,
            'Overfitting Gap': gaps
        })


        df_metrics['Rank'] = df_metrics['CV Score'].rank(ascending=False)
        df_sorted = df_metrics.sort_values('Rank')


        ax4.axis('tight')
        ax4.axis('off')
        table = ax4.table(cellText=df_sorted.round(4).values,
                          colLabels=df_sorted.columns,
                          cellLoc='center',
                          loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        ax4.set_title('Model Performance Summary', pad=20)

        plt.tight_layout()


        plot_path = os.path.join(self.output_dir, 'model_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Model comparison plot saved to {plot_path}")

    def plot_feature_importance(self, model, feature_names, top_n=15):
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=True).tail(top_n)

            plt.figure(figsize=(10, 8))
            plt.barh(importance_df['feature'], importance_df['importance'])
            plt.xlabel('Feature Importance')
            plt.title(f'Top {top_n} Feature Importances')
            plt.tight_layout()

            plot_path = os.path.join(self.output_dir, 'feature_importance.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"Feature importance plot saved to {plot_path}")

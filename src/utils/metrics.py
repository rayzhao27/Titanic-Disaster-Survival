from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Class for comprehensive model evaluation"""

    def __init__(self):
        pass

    def generate_detailed_report(self, y_true, y_pred, model_name: str, y_pred_proba=None) -> Dict[str, Any]:
        """Generate comprehensive evaluation metrics"""

        logger.info(f"Generating detailed evaluation report for {model_name}")

        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Classification report
        class_report = classification_report(y_true, y_pred, output_dict=True)

        # Advanced metrics with probabilities
        roc_auc = None
        pr_auc = None
        optimal_threshold = 0.5
        
        if y_pred_proba is not None:
            try:
                # ROC AUC
                roc_auc = roc_auc_score(y_true, y_pred_proba)
                
                # PR AUC (Average Precision Score)
                pr_auc = average_precision_score(y_true, y_pred_proba)
                
                # Optimal threshold using Youden J statistic
                optimal_threshold = self.find_optimal_threshold_youden(y_true, y_pred_proba)
                
                logger.info(f"ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}, Optimal Threshold: {optimal_threshold:.4f}")
                
            except Exception as e:
                logger.warning(f"Could not calculate advanced metrics: {e}")
                # Fallback to using predictions as probabilities (not ideal but works)
                try:
                    roc_auc = roc_auc_score(y_true, y_pred)
                except:
                    roc_auc = None

        report = {
            'model_name': model_name,
            'metrics': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'roc_auc': float(roc_auc) if roc_auc else None,
                'pr_auc': float(pr_auc) if pr_auc else None,
                'optimal_threshold': float(optimal_threshold)
            },
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report,
            'summary': {
                'total_samples': len(y_true),
                'correct_predictions': int(accuracy * len(y_true)),
                'survival_rate_actual': float(y_true.mean()),
                'survival_rate_predicted': float(y_pred.mean())
            }
        }

        return report

    def calculate_business_metrics(self, y_true, y_pred) -> Dict[str, float]:
        """Calculate business-relevant metrics"""

        # For Titanic: False positives = predicted survived but died (family false hope)
        # False negatives = predicted died but survived (missed rescue opportunity)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        metrics = {
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),  # Predicted survival but died
            'false_negatives': int(fn),  # Predicted death but survived
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,  # Recall for survivors
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,  # Recall for non-survivors
            'precision_survivors': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'precision_non_survivors': tn / (tn + fn) if (tn + fn) > 0 else 0
        }

        return metrics

    def find_optimal_threshold_youden(self, y_true, y_pred_proba) -> float:
        """Find optimal threshold using Youden J statistic (sensitivity + specificity - 1)"""
        
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        
        # Youden J statistic = sensitivity + specificity - 1 = tpr + (1 - fpr) - 1 = tpr - fpr
        youden_j = tpr - fpr
        
        # Find threshold that maximizes Youden J
        optimal_idx = np.argmax(youden_j)
        optimal_threshold = thresholds[optimal_idx]
        
        logger.info(f"Youden J optimal threshold: {optimal_threshold:.4f} (J={youden_j[optimal_idx]:.4f})")
        
        return optimal_threshold

    def find_optimal_threshold_f1(self, y_true, y_pred_proba) -> float:
        """Find optimal threshold that maximizes F1 score"""
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        
        # Calculate F1 scores for each threshold
        f1_scores = 2 * (precision * recall) / (precision + recall)
        f1_scores = np.nan_to_num(f1_scores)  # Handle division by zero
        
        # Find threshold that maximizes F1
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        
        logger.info(f"F1 optimal threshold: {optimal_threshold:.4f} (F1={f1_scores[optimal_idx]:.4f})")
        
        return optimal_threshold

    def evaluate_threshold_performance(self, y_true, y_pred_proba, threshold: float) -> Dict[str, float]:
        """Evaluate model performance at a specific threshold"""
        
        y_pred_thresh = (y_pred_proba >= threshold).astype(int)
        
        return {
            'threshold': threshold,
            'accuracy': accuracy_score(y_true, y_pred_thresh),
            'precision': precision_score(y_true, y_pred_thresh, zero_division=0),
            'recall': recall_score(y_true, y_pred_thresh, zero_division=0),
            'f1_score': f1_score(y_true, y_pred_thresh, zero_division=0)
        }

    def threshold_analysis(self, y_true, y_pred_proba) -> Dict[str, Any]:
        """Comprehensive threshold analysis with multiple optimization criteria"""
        
        # Find optimal thresholds using different criteria
        youden_threshold = self.find_optimal_threshold_youden(y_true, y_pred_proba)
        f1_threshold = self.find_optimal_threshold_f1(y_true, y_pred_proba)
        
        # Evaluate performance at different thresholds
        thresholds_to_test = [0.3, 0.4, 0.5, 0.6, 0.7, youden_threshold, f1_threshold]
        threshold_results = []
        
        for thresh in thresholds_to_test:
            result = self.evaluate_threshold_performance(y_true, y_pred_proba, thresh)
            threshold_results.append(result)
        
        return {
            'youden_optimal_threshold': youden_threshold,
            'f1_optimal_threshold': f1_threshold,
            'threshold_performance': threshold_results
        }

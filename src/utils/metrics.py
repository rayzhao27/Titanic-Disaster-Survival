from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import numpy as np
import pandas as pd
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Class for comprehensive model evaluation"""

    def __init__(self):
        pass

    def generate_detailed_report(self, y_true, y_pred, model_name: str) -> Dict[str, Any]:
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

        # ROC AUC if probabilities available
        try:
            auc_score = roc_auc_score(y_true, y_pred)
        except:
            auc_score = None

        report = {
            'model_name': model_name,
            'metrics': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'auc_score': float(auc_score) if auc_score else None
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

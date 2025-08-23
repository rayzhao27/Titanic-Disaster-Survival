from typing import Dict, List, Tuple, Any
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import lightgbm as lgb
import logging

logger = logging.getLogger(__name__)


class ModelFactory:
    """Factory class for creating and managing ML models"""

    def __init__(self, config: 'ModelConfig'):
        self.config = config

    def create_base_models(self) -> List[Tuple[str, Any]]:
        models = [
            ('rf', RandomForestClassifier(**self.config.rf_params)),
            ('xgb', xgb.XGBClassifier(**self.config.xgb_params)),
            ('lgb', lgb.LGBMClassifier(**self.config.lgb_params)),
            ('lr', LogisticRegression(**self.config.lr_params))
        ]

        return models

    def evaluate_models(self, models: List[Tuple[str, Any]],
                        X_train, y_train, X_val, y_val) -> Dict[str, Dict[str, float]]:
        results = {}

        for name, model in models:
            logger.info(f"Training and evaluating {name}...")

            # Fit model
            model.fit(X_train, y_train)

            # Predictions
            y_pred_train = model.predict(X_train)
            y_pred_val = model.predict(X_val)

            # Metrics
            train_acc = accuracy_score(y_train, y_pred_train)
            val_acc = accuracy_score(y_val, y_pred_val)

            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=self.config.cv_folds,
                scoring='accuracy'
            )

            results[name] = {
                'model': model,
                'train_accuracy': train_acc,
                'val_accuracy': val_acc,
                'cv_score': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'overfitting_gap': train_acc - val_acc,
                'is_overfitting': train_acc - val_acc > self.config.overfitting_threshold
            }

            logger.info(f"{name} - CV: {cv_scores.mean():.4f}, Gap: {train_acc - val_acc:.4f}")

        return results

    def create_ensemble(self, model_results: Dict[str, Dict[str, float]],
                        X_train, y_train, X_val, y_val) -> Tuple[Any, Dict[str, float]]:

        # Select models that aren't overfitting
        good_models = [
            (name, results['model'])
            for name, results in model_results.items()
            if not results['is_overfitting']
        ]

        if len(good_models) < 2:
            logger.warning("Not enough good models for ensemble, using best single model")
            best_model_name = max(model_results.keys(),
                                  key=lambda k: model_results[k]['cv_score'])
            return model_results[best_model_name]['model'], model_results[best_model_name]

        # Create voting classifier
        ensemble = VotingClassifier(
            estimators=good_models,
            voting=self.config.ensemble_voting
        )

        logger.info(f"Creating ensemble with models: {[name for name, _ in good_models]}")

        ensemble.fit(X_train, y_train)

        y_pred_train = ensemble.predict(X_train)
        y_pred_val = ensemble.predict(X_val)

        train_acc = accuracy_score(y_train, y_pred_train)
        val_acc = accuracy_score(y_val, y_pred_val)

        cv_scores = cross_val_score(
            ensemble, X_train, y_train,
            cv=self.config.cv_folds
        )

        ensemble_results = {
            'model': ensemble,
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'cv_score': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'overfitting_gap': train_acc - val_acc,
            'is_overfitting': train_acc - val_acc > self.config.overfitting_threshold
        }

        return ensemble, ensemble_results

    def select_best_model(self, model_results: Dict[str, Dict[str, float]]) -> Tuple[str, Any, Dict[str, float]]:
        # For compatibility with original version, prefer RF if it's close to the best
        rf_results = model_results.get('rf')
        best_model_name = max(model_results.keys(),
                              key=lambda k: model_results[k]['cv_score'])
        best_results = model_results[best_model_name]
        
        # If RF is within 0.002 of the best CV score, use RF for consistency with original
        if rf_results and abs(rf_results['cv_score'] - best_results['cv_score']) <= 0.002:
            logger.info(f"Using RF model for consistency (CV: {rf_results['cv_score']:.4f}) instead of {best_model_name} (CV: {best_results['cv_score']:.4f})")
            return 'rf', rf_results['model'], rf_results
        
        logger.info(f"Best model: {best_model_name} (CV: {best_results['cv_score']:.4f})")
        return best_model_name, best_results['model'], best_results

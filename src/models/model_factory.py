import logging
import xgboost as xgb
import lightgbm as lgb

from typing import Dict, List, Tuple, Any
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from sklearn.model_selection import cross_val_score


logger = logging.getLogger(__name__)


class ModelFactory:
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

            model.fit(X_train, y_train)

            y_pred_train = model.predict(X_train)
            y_pred_val = model.predict(X_val)

            try:
                y_pred_proba_train = model.predict_proba(X_train)[:, 1]
                y_pred_proba_val = model.predict_proba(X_val)[:, 1]
                has_proba = True
            except:
                logger.warning(f"Model {name} doesn't support predict_proba")
                y_pred_proba_train = y_pred_train.astype(float)
                y_pred_proba_val = y_pred_val.astype(float)
                has_proba = False

            train_acc = accuracy_score(y_train, y_pred_train)
            val_acc = accuracy_score(y_val, y_pred_val)

            train_f1 = f1_score(y_train, y_pred_train)
            val_f1 = f1_score(y_val, y_pred_val)
            
            train_roc_auc = roc_auc_score(y_train, y_pred_proba_train) if has_proba else None
            val_roc_auc = roc_auc_score(y_val, y_pred_proba_val) if has_proba else None
            
            train_pr_auc = average_precision_score(y_train, y_pred_proba_train) if has_proba else None
            val_pr_auc = average_precision_score(y_val, y_pred_proba_val) if has_proba else None

            cv_accuracy = cross_val_score(model, X_train, y_train, cv=self.config.cv_folds, scoring='accuracy')
            cv_f1 = cross_val_score(model, X_train, y_train, cv=self.config.cv_folds, scoring='f1')
            cv_roc_auc = cross_val_score(model, X_train, y_train, cv=self.config.cv_folds, scoring='roc_auc') if has_proba else None

            results[name] = {
                'model': model,
                'train_accuracy': train_acc,
                'val_accuracy': val_acc,
                'train_f1': train_f1,
                'val_f1': val_f1,
                'train_roc_auc': train_roc_auc,
                'val_roc_auc': val_roc_auc,
                'train_pr_auc': train_pr_auc,
                'val_pr_auc': val_pr_auc,
                'cv_score': cv_accuracy.mean(),
                'cv_std': cv_accuracy.std(),
                'cv_f1': cv_f1.mean() if cv_f1 is not None else None,
                'cv_roc_auc': cv_roc_auc.mean() if cv_roc_auc is not None else None,
                'overfitting_gap': train_acc - val_acc,
                'is_overfitting': train_acc - val_acc > self.config.overfitting_threshold,
                'has_proba': has_proba,
                'y_pred_proba_val': y_pred_proba_val  # Store for threshold optimization
            }

            roc_auc_str = f"{val_roc_auc:.4f}" if val_roc_auc else "N/A"
            pr_auc_str = f"{val_pr_auc:.4f}" if val_pr_auc else "N/A"
            logger.info(f"{name} - CV: {cv_accuracy.mean():.4f}, F1: {val_f1:.4f}, ROC-AUC: {roc_auc_str}, PR-AUC: {pr_auc_str}")

        return results

    def create_ensemble(self, model_results: Dict[str, Dict[str, float]],
                        X_train, y_train, X_val, y_val) -> Tuple[Any, Dict[str, float]]:

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

        ensemble = VotingClassifier(
            estimators=good_models,
            voting=self.config.ensemble_voting
        )

        logger.info(f"Creating ensemble with models: {[name for name, _ in good_models]}")

        ensemble.fit(X_train, y_train)

        y_pred_train = ensemble.predict(X_train)
        y_pred_val = ensemble.predict(X_val)

        try:
            y_pred_proba_train = ensemble.predict_proba(X_train)[:, 1]
            y_pred_proba_val = ensemble.predict_proba(X_val)[:, 1]
            has_proba = True
        except:
            y_pred_proba_train = y_pred_train.astype(float)
            y_pred_proba_val = y_pred_val.astype(float)
            has_proba = False

        train_acc = accuracy_score(y_train, y_pred_train)
        val_acc = accuracy_score(y_val, y_pred_val)
        
        train_f1 = f1_score(y_train, y_pred_train)
        val_f1 = f1_score(y_val, y_pred_val)
        
        train_roc_auc = roc_auc_score(y_train, y_pred_proba_train) if has_proba else None
        val_roc_auc = roc_auc_score(y_val, y_pred_proba_val) if has_proba else None
        
        train_pr_auc = average_precision_score(y_train, y_pred_proba_train) if has_proba else None
        val_pr_auc = average_precision_score(y_val, y_pred_proba_val) if has_proba else None

        cv_accuracy = cross_val_score(ensemble, X_train, y_train, cv=self.config.cv_folds, scoring='accuracy')
        cv_f1 = cross_val_score(ensemble, X_train, y_train, cv=self.config.cv_folds, scoring='f1')
        cv_roc_auc = cross_val_score(ensemble, X_train, y_train, cv=self.config.cv_folds, scoring='roc_auc') if has_proba else None

        ensemble_results = {
            'model': ensemble,
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'train_f1': train_f1,
            'val_f1': val_f1,
            'train_roc_auc': train_roc_auc,
            'val_roc_auc': val_roc_auc,
            'train_pr_auc': train_pr_auc,
            'val_pr_auc': val_pr_auc,
            'cv_score': cv_accuracy.mean(),
            'cv_std': cv_accuracy.std(),
            'cv_f1': cv_f1.mean() if cv_f1 is not None else None,
            'cv_roc_auc': cv_roc_auc.mean() if cv_roc_auc is not None else None,
            'overfitting_gap': train_acc - val_acc,
            'is_overfitting': train_acc - val_acc > self.config.overfitting_threshold,
            'has_proba': has_proba,
            'y_pred_proba_val': y_pred_proba_val
        }

        return ensemble, ensemble_results

    def select_best_model(self, model_results: Dict[str, Dict[str, float]]) -> Tuple[str, Any, Dict[str, float]]:
        rf_results = model_results.get('rf')
        best_model_name = max(model_results.keys(),
                              key=lambda k: model_results[k]['cv_score'])
        best_results = model_results[best_model_name]
        
        logger.info("\n" + "="*60)
        logger.info("MODEL PERFORMANCE SUMMARY")
        logger.info("="*60)
        for name, results in model_results.items():
            logger.info(f"{name.upper()}:")
            logger.info(f"  CV Accuracy: {results['cv_score']:.4f}")
            logger.info(f"  Val F1: {results['val_f1']:.4f}")
            if results['val_roc_auc']:
                logger.info(f"  Val ROC-AUC: {results['val_roc_auc']:.4f}")
            if results['val_pr_auc']:
                logger.info(f"  Val PR-AUC: {results['val_pr_auc']:.4f}")
            logger.info(f"  Overfitting Gap: {results['overfitting_gap']:.4f}")
        logger.info("="*60)
        
        if rf_results and abs(rf_results['cv_score'] - best_results['cv_score']) <= 0.002:
            logger.info(f"Using RF model for consistency (CV: {rf_results['cv_score']:.4f}) instead of {best_model_name} (CV: {best_results['cv_score']:.4f})")
            return 'rf', rf_results['model'], rf_results
        
        logger.info(f"Best model: {best_model_name} (CV: {best_results['cv_score']:.4f})")
        return best_model_name, best_results['model'], best_results

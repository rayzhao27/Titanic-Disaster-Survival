import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import logging
from typing import Tuple, Dict, Any
import os

from src.data.data_loader import DataLoader
from src.features.feature_pipeline import create_feature_pipeline, create_preprocessing_pipeline
from src.models.model_factory import ModelFactory
from src.utils.visualization import ModelVisualizer
from src.utils.metrics import ModelEvaluator

logger = logging.getLogger(__name__)


class MLPipeline:
    """Main ML Pipeline orchestrator"""

    def __init__(self, config: 'Config', data_loader: DataLoader):
        self.config = config
        self.data_loader = data_loader
        self.feature_pipeline = None
        self.preprocessing_pipeline = None
        self.model = None
        self.model_results = {}
        self.best_model_name = None

        os.makedirs(self.config.data.output_dir, exist_ok=True)

    def run(self) -> Dict[str, Any]:
        """Execute the complete ML pipeline"""

        logger.info("Starting ML Pipeline execution...")

        # 1. Load data
        train_data, test_data = self.data_loader.load_data()

        # 2. Feature engineering
        X_train, X_test, y_train, test_ids = self._prepare_features(train_data, test_data)

        # 3. Preprocessing
        X_train_processed, X_val_processed, X_test_processed, y_train_split, y_val \
            = self._preprocess_data(X_train, y_train, X_test)

        # 4. Model training and evaluation
        self._train_and_evaluate_models(X_train_processed, y_train_split, X_val_processed, y_val)

        # 5. Model selection
        self.best_model_name, self.model, best_results = self._select_best_model()

        # 6. Generate predictions
        predictions = self._generate_predictions(X_test_processed, test_ids)

        # 7. Save artifacts
        self._save_artifacts()

        # 8. Generate visualizations and reports
        self._generate_reports(X_val_processed, y_val)

        logger.info("Pipeline execution completed successfully!")

        return {
            'best_model': self.best_model_name,
            'cv_score': best_results['cv_score'],
            'validation_accuracy': best_results['val_accuracy'],
            'predictions': predictions,
            'model_results': self.model_results
        }

    def _prepare_features(self, train_data: pd.DataFrame,
                          test_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Apply feature engineering pipeline - using original helper functions for exact compatibility"""

        logger.info("Applying feature engineering...")

        # Use feature pipeline for processing
        combined_data = pd.concat([train_data, test_data], sort=True).reset_index(drop=True)

        # Create and fit feature pipeline
        self.feature_pipeline = create_feature_pipeline()
        combined_processed = self.feature_pipeline.fit_transform(combined_data)

        train_processed = combined_processed.iloc[:len(train_data)]
        test_processed = combined_processed.iloc[len(train_data):]

        # Separate features and target
        X_train = train_processed.drop(['Survived', 'PassengerId'], axis=1, errors='ignore')
        y_train = train_processed['Survived'] if 'Survived' in train_processed.columns else None

        X_test = test_processed.drop(['Survived', 'PassengerId'], axis=1, errors='ignore')
        test_ids = test_data['PassengerId'].copy()
        
        # Align train and test sets to ensure same columns
        X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

        logger.info(f"Feature engineering completed. Train shape: {X_train.shape}, Test shape: {X_test.shape}")

        return X_train, X_test, y_train, test_ids

    def _preprocess_data(self, X_train: pd.DataFrame, y_train: pd.Series,
                         X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Implement preprocessing (scaling, feature selection) - exact same as original"""

        logger.info("Applying preprocessing...")

        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train,
            test_size=self.config.data.test_size,
            random_state=self.config.data.random_state,
            stratify=y_train
        )

        # Use manual preprocessing to match original exactly
        from sklearn.preprocessing import StandardScaler
        from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold

        # Feature scaling (exact same as original)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_split)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train_split.columns, index=X_train_split.index)
        X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

        # Feature selection (exact same as original)
        variance_threshold = VarianceThreshold(threshold=self.config.features.variance_threshold)
        X_train_var = variance_threshold.fit_transform(X_train_scaled)
        X_val_var = variance_threshold.transform(X_val_scaled)
        X_test_var = variance_threshold.transform(X_test_scaled)

        remaining_features = X_train_scaled.columns[variance_threshold.get_support()]
        
        X_train_var = pd.DataFrame(X_train_var, columns=remaining_features, index=X_train_scaled.index)
        X_val_var = pd.DataFrame(X_val_var, columns=remaining_features, index=X_val_scaled.index)
        X_test_var = pd.DataFrame(X_test_var, columns=remaining_features, index=X_test_scaled.index)

        # Select K best features (exact same as original)
        selector = SelectKBest(score_func=f_classif, k=min(self.config.features.k_best_features, len(remaining_features)))
        X_train_selected = selector.fit_transform(X_train_var, y_train_split)
        X_val_selected = selector.transform(X_val_var)
        X_test_selected = selector.transform(X_test_var)

        selected_features = remaining_features[selector.get_support()].tolist()
        
        X_train_processed = pd.DataFrame(X_train_selected, columns=selected_features, index=X_train_var.index)
        X_val_processed = pd.DataFrame(X_val_selected, columns=selected_features, index=X_val_var.index)
        X_test_processed = pd.DataFrame(X_test_selected, columns=selected_features, index=X_test_var.index)

        # Store preprocessing components for later use
        self.scaler = scaler
        self.variance_selector = variance_threshold
        self.feature_selector = selector
        self.selected_features = selected_features

        logger.info(f"Preprocessing completed. Final feature count: {X_train_processed.shape[1]}")
        logger.info(f"Selected features: {selected_features}")

        return X_train_processed, X_val_processed, X_test_processed, y_train_split, y_val

    def _train_and_evaluate_models(self, X_train, y_train, X_val, y_val):
        """Train and evaluate all models"""

        logger.info("Training and evaluating models...")

        model_factory = ModelFactory(self.config.model)

        # Create and evaluate base models
        base_models = model_factory.create_base_models()
        self.model_results = model_factory.evaluate_models(base_models, X_train, y_train, X_val, y_val)

        # Create ensemble if beneficial
        ensemble_model, ensemble_results = model_factory.create_ensemble(
            self.model_results, X_train, y_train, X_val, y_val
        )

        if ensemble_results and 'model' in ensemble_results:
            self.model_results['ensemble'] = ensemble_results

    def _select_best_model(self) -> Tuple[str, Any, Dict[str, float]]:
        model_factory = ModelFactory(self.config.model)
        return model_factory.select_best_model(self.model_results)

    def _generate_predictions(self, X_test, test_ids) -> pd.DataFrame:
        logger.info("Generating predictions...")

        predictions = self.model.predict(X_test)

        submission = pd.DataFrame({
            'PassengerId': test_ids,
            'Survived': predictions.astype(int)
        })

        # Save submission
        submission_path = os.path.join(self.config.data.output_dir, 'submission.csv')
        submission.to_csv(submission_path, index=False)

        logger.info(f"Predictions saved to {submission_path}")
        logger.info(f"Predicted survival rate: {submission['Survived'].mean():.3f}")

        return submission

    def _save_artifacts(self):
        """Save trained models and pipelines"""

        logger.info("Saving model artifacts...")

        # Save complete pipeline package
        pipeline_package = {
            'feature_pipeline': self.feature_pipeline,
            'preprocessing_pipeline': self.preprocessing_pipeline,
            'model': self.model,
            'best_model_name': self.best_model_name,
            'config': self.config,
            'model_results': self.model_results
        }

        package_path = os.path.join(self.config.data.output_dir, 'model_package.pkl')
        with open(package_path, 'wb') as f:
            pickle.dump(pipeline_package, f)

        logger.info(f"Model package saved to {package_path}")

    def _generate_reports(self, X_val, y_val):
        """Generate visualizations and performance reports"""

        logger.info("Generating reports and visualizations...")

        # Create visualizer and evaluator
        pic_folder = os.path.join(self.config.data.output_dir, 'pics')
        visualizer = ModelVisualizer(pic_folder)
        evaluator = ModelEvaluator()

        # Generate model comparison visualization
        visualizer.plot_model_comparison(self.model_results)

        # Generate detailed evaluation report with probabilities
        y_pred_val = self.model.predict(X_val)
        
        # Get probability predictions if available
        try:
            y_pred_proba_val = self.model.predict_proba(X_val)[:, 1]
        except:
            logger.warning("Model doesn't support predict_proba, using hard predictions")
            y_pred_proba_val = None
        
        evaluation_report = evaluator.generate_detailed_report(y_val, y_pred_val, self.best_model_name, y_pred_proba_val)
        
        # Add threshold analysis if probabilities are available
        if y_pred_proba_val is not None:
            threshold_analysis = evaluator.threshold_analysis(y_val, y_pred_proba_val)
            evaluation_report['threshold_analysis'] = threshold_analysis

        # Save evaluation report
        report_path = os.path.join(self.config.data.output_dir, 'evaluation_report.json')
        import json
        with open(report_path, 'w') as f:
            json.dump(evaluation_report, f, indent=2)

        logger.info("Reports generated successfully!")

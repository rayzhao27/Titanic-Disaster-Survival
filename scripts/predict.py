import pickle
import pandas as pd
import argparse
import logging
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.utils.logging_config import setup_logging


def load_model_package(package_path: str):
    with open(package_path, 'rb') as f:
        package = pickle.load(f)
    return package


def predict_new_data(package, new_data_path: str, output_path: str = None):
    logger = logging.getLogger(__name__)

    logger.info(f"Loading new data from {new_data_path}")
    new_data = pd.read_csv(new_data_path)

    if 'PassengerId' in new_data.columns:
        passenger_ids = new_data['PassengerId'].copy()
    else:
        passenger_ids = range(len(new_data))

    logger.info("Applying feature engineering...")
    features_processed = package['feature_pipeline'].transform(new_data)

    features_only = features_processed.drop(['Survived', 'PassengerId'],
                                            axis=1, errors='ignore')

    logger.info("Applying preprocessing...")
    features_final = package['preprocessing_pipeline'].transform(features_only)

    logger.info("Generating predictions...")
    predictions = package['model'].predict(features_final)

    results = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Survived': predictions.astype(int)
    })

    if output_path is None:
        output_path = 'new_predictions.csv'

    results.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")
    logger.info(f"Predicted survival rate: {results['Survived'].mean():.3f}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Make predictions using trained model')
    parser.add_argument('--model-package', type=str, required=True,
                        help='Path to saved model package')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to new data for prediction')
    parser.add_argument('--output', type=str, default='new_predictions.csv',
                        help='Output path for predictions')
    parser.add_argument('--log-level', type=str, default='INFO',
                        help='Logging level')

    args = parser.parse_args()

    setup_logging(log_level=args.log_level)
    logger = logging.getLogger(__name__)

    try:
        logger.info(f"Loading model package from {args.model_package}")
        package = load_model_package(args.model_package)

        logger.info(f"Loaded model: {package['best_model_name']}")
        logger.info(f"Model CV score: {package['model_results'][package['best_model_name']]['cv_score']:.4f}")

        results = predict_new_data(package, args.data, args.output)

        logger.info("Prediction completed successfully!")

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

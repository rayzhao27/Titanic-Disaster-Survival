import logging
import argparse
from pathlib import Path
import sys
import json


sys.path.append(str(Path(__file__).parent / "src"))

from config.config import Config
from src.data.data_loader import CSVDataLoader
from src.pipeline.ml_pipeline import MLPipeline
from src.utils.logging_config import setup_logging


def main():
    parser = argparse.ArgumentParser(description='ML Pipeline for Titanic Survival Prediction')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--train-data', type=str, help='Path to training data')
    parser.add_argument('--test-data', type=str, help='Path to test data')
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='Output directory for results')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')

    args = parser.parse_args()


    log_file = Path(args.output_dir) / 'pipeline.log'
    setup_logging(log_level=args.log_level, log_file=str(log_file))

    logger = logging.getLogger(__name__)
    logger.info("Starting Titanic ML Pipeline")

    try:

        config = Config()


        if args.train_data:
            config.data.train_path = args.train_data
        if args.test_data:
            config.data.test_path = args.test_data
        if args.output_dir:
            config.data.output_dir = args.output_dir


        if not Path(config.data.train_path).exists():
            raise FileNotFoundError(f"Training data not found: {config.data.train_path}")
        if not Path(config.data.test_path).exists():
            raise FileNotFoundError(f"Test data not found: {config.data.test_path}")

        data_loader = CSVDataLoader(config.data.train_path, config.data.test_path)

        pipeline = MLPipeline(config, data_loader)
        results = pipeline.run()

        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE EXECUTION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Best Model: {results['best_model']}")
        logger.info(f"Cross-Validation Score: {results['cv_score']:.4f}")
        logger.info(f"Validation Accuracy: {results['validation_accuracy']:.4f}")
        logger.info(f"Predicted Survival Rate: {results['predictions']['Survived'].mean():.3f}")

        logger.info(f"Results saved to: {config.data.output_dir}")
        logger.info("=" * 80)

        logger.info("Pipeline completed successfully!")

    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

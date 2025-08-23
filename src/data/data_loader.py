import pandas as pd
import logging
from abc import ABC, abstractmethod
from typing import Tuple

logger = logging.getLogger(__name__)


class DataLoader(ABC):
    """Abstract base class for data loading"""

    @abstractmethod
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        pass


class CSVDataLoader(DataLoader):
    """Concrete implementation for CSV data loading"""

    def __init__(self, train_path: str, test_path: str):
        self.train_path = train_path
        self.test_path = test_path

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load train and test data from CSV files"""
        try:
            logger.info(f"Loading training data from {self.train_path}")
            train_data = pd.read_csv(self.train_path)

            logger.info(f"Loading test data from {self.test_path}")
            test_data = pd.read_csv(self.test_path)

            logger.info(f"Loaded train: {train_data.shape}, test: {test_data.shape}")
            return train_data, test_data

        except FileNotFoundError as e:
            logger.error(f"Data file not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

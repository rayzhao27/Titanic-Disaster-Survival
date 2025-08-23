from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class DataConfig:
    train_path: str = "src/data/train.csv"
    test_path: str = "src/data/test.csv"
    output_dir: str = "outputs"
    random_state: int = 42
    test_size: float = 0.3


@dataclass
class FeatureConfig:
    variance_threshold: float = 0.01
    k_best_features: int = 12
    scale_features: bool = True


@dataclass
class ModelConfig:
    cv_folds: int = 10
    overfitting_threshold: float = 0.03
    ensemble_voting: str = "soft"
    random_state: int = 42

    rf_params: Dict[str, Any] = None
    xgb_params: Dict[str, Any] = None
    lgb_params: Dict[str, Any] = None
    lr_params: Dict[str, Any] = None

    def __post_init__(self):
        if self.rf_params is None:
            self.rf_params = {
                'n_estimators': 30,
                'max_depth': 4,
                'min_samples_split': 30,
                'min_samples_leaf': 15,
                'max_features': 0.4,
                'bootstrap': True,
                'max_samples': 0.7,
                'random_state': self.random_state,
                'n_jobs': -1
            }

        if self.xgb_params is None:
            self.xgb_params = {
                'n_estimators': 30,
                'max_depth': 3,
                'learning_rate': 0.02,
                'subsample': 0.6,
                'colsample_bytree': 0.6,
                'reg_alpha': 1.0,
                'reg_lambda': 3.0,
                'min_child_weight': 10,
                'gamma': 0.2,
                'scale_pos_weight': 1,
                'random_state': self.random_state,
                'n_jobs': -1,
                'eval_metric': 'logloss'
            }

        if self.lgb_params is None:
            self.lgb_params = {
                "n_estimators": 30,
                "max_depth": 3,
                "learning_rate": 0.02,
                "subsample": 0.6,
                "colsample_bytree": 0.6,
                "reg_alpha": 1.0,
                "reg_lambda": 3.0,
                "min_child_samples": 20,
                "min_split_gain": 0.1,
                "random_state": 42,
                "n_jobs": -1,
                "verbose": -1
            }

        if self.lr_params is None:
            self.lr_params = {
                "C": 0.01,
                "penalty": "l1",
                "solver": "liblinear",
                "max_iter": 1000,
                "random_state": 42
            }


@dataclass
class Config:
    data: DataConfig = DataConfig()
    features: FeatureConfig = FeatureConfig()
    model: ModelConfig = ModelConfig()

    @property
    def random_state(self):
        return self.data.random_state

<a name="readme-top"></a>

## About The Project

A comprehensive machine learning pipeline for Titanic survival prediction that combines advanced feature engineering, ensemble modeling, and robust evaluation metrics. This project provides a clean, modular architecture while maintaining compatibility with proven feature engineering techniques to achieve 83.32% cross-validation accuracy.

The pipeline includes sophisticated family survival analysis, comprehensive preprocessing, multiple model evaluation with ensemble capabilities, and production-ready prediction scripts. The `main.py` runs the complete ML pipeline from data loading to final predictions.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

* Python
* scikit-learn
* XGBoost
* LightGBM
* pandas
* matplotlib

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Project Structure -->

## Project Structure

```sh
Titanic-ML-Pipeline/
├── main.py                         # Main entry point
├── config/
│   └── config.py                   # Configuration management
├── src/
│   ├── data/
│   │   ├── data_loader.py          # CSV data loading and preprocessing
│   │   ├── train.csv               # Training dataset
│   │   └── test.csv                # Test dataset
│   ├── features/
│   │   └── feature_pipeline.py     # Feature engineering pipeline
│   ├── models/
│   │   └── model_factory.py        # Model creation and evaluation
│   ├── pipeline/
│   │   └── ml_pipeline.py          # Main ML pipeline orchestrator
│   └── utils/
│       ├── logging_config.py       # Logging configuration
│       ├── metrics.py              # Model evaluation metrics
│       └── visualization.py        # Training visualization
├── scripts/
│   └── predict.py                  # Standalone prediction script
└── outputs/                        # Training outputs and results
    ├── submission.csv              # Kaggle submission file
    ├── model_package.pkl           # Trained model package
    ├── evaluation_report.json      # Detailed evaluation metrics
    └── pics/
        └── model_comparison.png    # Model performance visualization
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Architecture Components -->

## Architecture Components

**Feature Engineering** (`src/features/`)

- **`feature_pipeline.py`**: Advanced feature engineering pipeline
  - **Missing Value Imputation**: Intelligent handling of Age, Cabin, Embarked, and Fare
  - **Title Extraction**: Extract and normalize passenger titles from names
  - **Family Survival Analysis**: Sophisticated family grouping by name/fare and ticket
  - **Binning Features**: Age and Fare quantile-based binning
  - **Family Size**: Combined SibSp and Parch features
  - **Categorical Encoding**: One-hot encoding with proper handling

**Configuration Management** (`config/`)

- **`config.py`**: Centralized configuration using dataclasses
  - `DataConfig`: Data paths, random seeds, and split ratios
  - `FeatureConfig`: Feature selection and preprocessing parameters
  - `ModelConfig`: Model hyperparameters for all algorithms

**Data Pipeline** (`src/data/`)

- **`data_loader.py`**: Robust CSV data loading with error handling
- **Features**:
  - Abstract base class for extensible data loading
  - Comprehensive logging and error handling
  - Support for different data sources

**Model Factory** (`src/models/`)

- **`model_factory.py`**: Comprehensive model management system
- **Supported Models**:
  - **Random Forest**: Optimized for feature importance and robustness
  - **XGBoost**: Gradient boosting with advanced regularization
  - **LightGBM**: Fast gradient boosting with categorical support
  - **Logistic Regression**: L1 regularized for feature selection
  - **Ensemble**: Voting classifier with overfitting detection

**ML Pipeline** (`src/pipeline/`)

- **`ml_pipeline.py`**: Complete end-to-end ML pipeline
- **Features**:
  - Feature engineering with legacy compatibility
  - Manual preprocessing for exact reproducibility
  - Cross-validation with stratified sampling
  - Model comparison and selection
  - Automated artifact saving and visualization

**Evaluation & Visualization** (`src/utils/`)

- **`metrics.py`**: Comprehensive model evaluation
  - Classification metrics (accuracy, precision, recall, F1)
  - Advanced metrics (ROC-AUC, PR-AUC)
  - Threshold optimization (Youden J statistic, F1 maximization)
  - Business metrics (survival rates, confusion analysis)

- **`visualization.py`**: Training progress and model comparison
  - Model performance comparison charts
  - Feature importance visualization
  - Training vs validation accuracy plots

- **`logging_config.py`**: Centralized logging configuration
  - Console and file logging
  - Configurable log levels
  - Library log suppression

**Prediction Scripts** (`scripts/`)

- **`predict.py`**: Standalone prediction script for new data
  - Load trained model packages
  - Apply complete preprocessing pipeline
  - Generate predictions for new datasets

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Usage -->

## Usage

**Prerequisites**

```bash
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn
```

**Run the Complete Pipeline**

```bash
python main.py
```

**Command Line Arguments**

| Argument        | Type  | Default               | Description                           |
| --------------- | ----- | --------------------- | ------------------------------------- |
| `--config`      | `str` | None                  | Path to custom config file           |
| `--train-data`  | `str` | `src/data/train.csv`  | Path to training data                 |
| `--test-data`   | `str` | `src/data/test.csv`   | Path to test data                     |
| `--output-dir`  | `str` | `outputs`             | Output directory for results         |
| `--log-level`   | `str` | `INFO`                | Logging level (DEBUG/INFO/WARNING)   |

**Example with Custom Parameters**

```bash
python main.py --output-dir results --log-level DEBUG --train-data data/train.csv
```

**Make Predictions on New Data**

```bash
python scripts/predict.py --model-package outputs/model_package.pkl --data new_data.csv --output predictions.csv
```

**Output Files**

- `outputs/submission.csv`: Kaggle submission file
- `outputs/model_package.pkl`: Complete trained pipeline
- `outputs/evaluation_report.json`: Detailed performance metrics
- `outputs/pics/model_comparison.png`: Model performance visualization
- `outputs/pipeline.log`: Execution logs

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Configuration -->

## Configuration

### Model Parameters (`config/config.py`)

```python
@dataclass
class ModelConfig:
    cv_folds: int = 10                    # Cross-validation folds
    overfitting_threshold: float = 0.03   # Overfitting detection threshold
    ensemble_voting: str = "soft"         # Ensemble voting method
    random_state: int = 42                # Random seed
    
    # Random Forest parameters
    rf_params: Dict[str, Any] = {
        'n_estimators': 30,
        'max_depth': 4,
        'min_samples_split': 30,
        'min_samples_leaf': 15,
        'max_features': 0.4,
        'bootstrap': True,
        'max_samples': 0.7
    }
```

### Feature Engineering Parameters

```python
@dataclass
class FeatureConfig:
    variance_threshold: float = 0.01      # Minimum feature variance
    k_best_features: int = 12             # Number of top features to select
    scale_features: bool = True           # Enable feature scaling
```

### Data Parameters

```python
@dataclass
class DataConfig:
    train_path: str = "src/data/train.csv"
    test_path: str = "src/data/test.csv"
    output_dir: str = "outputs"
    random_state: int = 42
    test_size: float = 0.3                # Validation split ratio
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Feature Engineering -->

## **Feature Engineering**

**Advanced Feature Creation**

1. **Title Extraction & Normalization**: 
   - Extract titles from passenger names
   - Normalize rare titles (Col, Major, etc. → Mr)
   - Use titles for intelligent age imputation

2. **Family Survival Analysis**:
   - Group families by last name and fare
   - Secondary grouping by ticket number
   - Calculate family survival rates for prediction

3. **Intelligent Missing Value Handling**:
   - Age: Imputed by passenger title median
   - Fare: Group-based imputation by class and family size
   - Embarked: Manual correction for known passengers

4. **Feature Binning**:
   - Age: Quantile-based binning (5 bins)
   - Fare: Quantile-based binning (4 bins)

5. **Categorical Encoding**:
   - One-hot encoding with drop_first=True
   - Proper handling of passenger class as categorical

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Model Performance -->

## **Model Performance**

The pipeline evaluates multiple algorithms and selects the best performer:

- **Random Forest**: Robust ensemble with feature importance
- **XGBoost**: Gradient boosting with regularization
- **LightGBM**: Fast gradient boosting
- **Logistic Regression**: L1 regularized linear model
- **Ensemble**: Voting classifier (when beneficial)

**Performance Metrics**

- **Cross-Validation Score**: 83.32%
- **Validation Accuracy**: 81.72%
- **F1-Score**: 0.81
- **ROC-AUC**: 0.87
- **PR-AUC**: 0.84
- **Optimal Threshold**: 0.358 (Youden J statistic)
- **Overfitting Detection**: Automatic gap analysis
- **Feature Selection**: Top 7 most predictive features

![Model Comparison](outputs/pics/model_comparison.png)

*Model performance comparison showing training vs validation accuracies across different algorithms.*

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Key Features -->

## **Key Features**

**Reproducibility**
- Fixed random seeds across all components
- Exact compatibility with legacy feature engineering
- Deterministic train/validation splits

**Robustness**
- Comprehensive error handling and logging
- Overfitting detection and prevention
- Cross-validation for reliable performance estimates

**Extensibility**
- Modular architecture for easy algorithm addition
- Configuration-driven hyperparameter management
- Abstract base classes for custom data loaders

**Production Ready**
- Complete artifact saving (models, scalers, selectors)
- Detailed evaluation reports with business metrics
- Automated visualization generation
- Standalone prediction scripts for deployment

**Model Management**
- Intelligent model selection with RF preference for consistency
- Ensemble creation with overfitting detection
- Comprehensive model comparison and evaluation

<p align="right">(<a href="#readme-top">back to top</a>)</p>
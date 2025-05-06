# entity classes for all components of the pipeline

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataPreprocessingConfig:
    """
    Data Preprocessing Configuration
    """

    root_dir: Path
    data_path: Path
    preprocessed_data_path: Path
    img_size: tuple


@dataclass(frozen=True)
class ModelTrainingConfig:
    """
    Model Training Configuration
    """

    root_dir: Path
    data_path: Path
    img_size: tuple
    batch_size: int
    seed: int
    num_epochs: int
    learning_rate: float
    num_workers: int
    feature_size: int


@dataclass(frozen=True)
class ModelEvaluationConfig:
    """
    Model Evaluation Configuration
    """

    root_dir: Path
    model_path: Path
    data_path: Path
    img_size: tuple
    num_workers: int
    feature_size: int

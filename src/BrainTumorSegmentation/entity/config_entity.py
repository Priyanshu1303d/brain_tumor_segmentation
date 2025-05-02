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

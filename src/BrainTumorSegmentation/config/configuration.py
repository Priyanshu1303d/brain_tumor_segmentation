from pathlib import Path

from BrainTumorSegmentation.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from BrainTumorSegmentation.entity.config_entity import (
    DataPreprocessingConfig,
    ModelEvaluationConfig,
    ModelTrainingConfig,
)
from BrainTumorSegmentation.utils.common import create_directories, read_yaml


class ConfigurationManager:
    def __init__(self):
        """
        Configuration Manager to manage the configuration of the project.
        It reads the configuration and parameters from YAML files and creates necessary directories.
        """
        self.config_path = Path(CONFIG_FILE_PATH)
        self.params_path = Path(PARAMS_FILE_PATH)
        self.config = read_yaml(self.config_path)
        self.params = read_yaml(self.params_path)

    def get_data_preprocessing_config(self) -> DataPreprocessingConfig:
        """
        Get the data preprocessing configuration.
        Returns:
            DataPreprocessingConfig: Data preprocessing configuration object.
        """
        config = self.config.data_preprocessing
        create_directories([config.root_dir, config.preprocessed_data_path])
        return DataPreprocessingConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            preprocessed_data_path=config.preprocessed_data_path,
            img_size=tuple(self.params.img_size),
        )

    def get_model_training_config(self) -> ModelTrainingConfig:
        """
        Get the model training configuration.
        Returns:
            ModelTrainingConfig: Model training configuration object.
        """
        config = self.config.model_training
        create_directories([config.root_dir])
        return ModelTrainingConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            img_size=tuple(self.params.img_size),
            batch_size=self.params.batch_size,
            seed=self.params.seed,
            num_epochs=self.params.num_epochs,
            learning_rate=float(self.params.learning_rate),
            num_workers=self.params.num_workers,
            feature_size=self.params.feature_size,
        )

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        """
        Get the model evaluation configuration.
        Returns:
            ModelEvaluationConfig: Model evaluation configuration object.
        """
        config = self.config.model_evaluation
        create_directories([config.root_dir])
        return ModelEvaluationConfig(
            root_dir=config.root_dir,
            model_path=config.model_path,
            data_path=config.data_path,
            img_size=tuple(self.params.img_size),
            num_workers=self.params.num_workers,
            feature_size=self.params.feature_size,
        )

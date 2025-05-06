from BrainTumorSegmentation import logger
from BrainTumorSegmentation.components.model_training import ModelTraining
from BrainTumorSegmentation.config.configuration import ConfigurationManager

STAGE_NAME = "MODEL TRAINING STAGE"


class ModelTrainingPipeline:
    """
    ModelTraining Pipeline to handle the entire training workflow.
    """

    def __init__(self):
        """
        Initialize the ModelTrainingPipeline class.
        """
        self.config = ConfigurationManager().get_model_training_config()
        self.trainer = ModelTraining(config=self.config)

    def run(self):
        """
        Run the training pipeline.
        """
        self.trainer.run()


if __name__ == "__main__":
    """
    Main function to run the model training pipeline.
    """
    try:
        logger.info(f"{'='*10} {STAGE_NAME} {'='*10}")
        training_pipeline = ModelTrainingPipeline()
        training_pipeline.run()
        logger.info(f"{'='*10} {STAGE_NAME} completed {'='*10}")
    except Exception as e:
        logger.exception(e)
        raise e

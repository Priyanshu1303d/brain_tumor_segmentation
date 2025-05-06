from BrainTumorSegmentation import logger
from BrainTumorSegmentation.components.model_evaluation import ModelEvaluation
from BrainTumorSegmentation.config.configuration import ConfigurationManager

STAGE_NAME = "MODEL EVALUATION STAGE"


class ModelEvaluationPipeline:
    """
    ModelEvaluation Pipeline to handle the entire evaluation workflow.
    """

    def __init__(self):
        """
        Initialize the ModelEvaluationPipeline class.
        """
        self.config = ConfigurationManager().get_model_evaluation_config()
        self.evaluator = ModelEvaluation(config=self.config)

    def run(self):
        """
        Run the evaluation pipeline.
        """
        self.evaluator.run()


if __name__ == "__main__":
    """
    Main function to run the model evaluation pipeline.
    """
    try:
        logger.info(f"{'='*10} {STAGE_NAME} {'='*10}")
        evaluation_pipeline = ModelEvaluationPipeline()
        evaluation_pipeline.run()
        logger.info(f"{'='*10} {STAGE_NAME} completed {'='*10}")
    except Exception as e:
        logger.exception(e)
        raise e

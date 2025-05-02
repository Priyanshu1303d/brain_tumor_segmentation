from BrainTumorSegmentation import logger
from BrainTumorSegmentation.components.data_preprocessing import DataPreprocessing
from BrainTumorSegmentation.config.configuration import ConfigurationManager

STAGE_NAME = "DATA PREPROCESSING STAGE"


class DataPreprocessingPipeline:
    """
    Data Preprocessing Pipeline to handle the entire preprocessing workflow.
    """

    def __init__(self):
        """
        Initialize the DataPreprocessingPipeline class.
        """
        self.config = ConfigurationManager().get_data_preprocessing_config()
        self.data_preprocessor = DataPreprocessing(config=self.config)

    def run(self):
        """
        Run the data preprocessing pipeline.
        """
        self.data_preprocessor.run()


if __name__ == "__main__":
    """
    Main function to run the data preprocessing pipeline.
    """
    try:
        logger.info(f"{'='*10} {STAGE_NAME} {'='*10}")
        data_preprocessing_pipeline = DataPreprocessingPipeline()
        data_preprocessing_pipeline.run()
        logger.info(f"{'='*10} {STAGE_NAME} completed {'='*10}")
    except Exception as e:
        logger.exception(e)
        raise e

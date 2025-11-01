from src.Chatbot.config.configration import ConfigurationManager
from src.Chatbot.components.data_transformation import DataTransformation
from src.Chatbot.logging import logger


class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def initiate_data_transformation(self):
        config=ConfigurationManager()
        data_transformation_config=config.get_data_transformation_config()
        data_transformation=DataTransformation(config=data_transformation_config)
        data_transformation.transform_and_save()
        
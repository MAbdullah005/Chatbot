from src.Chatbot.config.configration import ConfigurationManager
from src.Chatbot.components.model_trainer import ModelTrainer
from src.Chatbot.logging import logger

class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass
    def initiate_model_trainer(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer_config = ModelTrainer(config=model_trainer_config)
        model_trainer_config.train()
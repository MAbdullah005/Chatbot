from datasets import load_dataset
from src.Chatbot.entity import DataIngestionConfig
from src.Chatbot.logging import logger
from pathlib import Path

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def extract_data(self):
        """
        Loads dataset from Hugging Face Hub and saves it locally in Arrow format
        (compatible with `load_from_disk()`).
        """
        logger.info(f" Loading dataset {self.config.dataset} from Hugging Face...")
        dataset = load_dataset(self.config.dataset, token=self.config.token_id)

        # Ensure directory exists
        save_dir = Path(self.config.local_data_file)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save in Hugging Face's native Arrow format
        logger.info(f" Saving dataset to {save_dir} in Arrow format...")
        dataset.save_to_disk(str(save_dir))

        logger.info(" Dataset downloaded and saved successfully in Arrow format!")
        return dataset

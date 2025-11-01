from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Trainer
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from src.Chatbot.entity import DataIngestionConfig
from src.Chatbot.logging import logger
from src.Chatbot.config.configration import ConfigurationManager

from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def extract_data(self):
      """
      Loads dataset from Hugging Face Hub, splits into train/test, 
      and saves both locally in Arrow format (compatible with load_from_disk()).
      """
      logger.info(f" Loading dataset {self.config.dataset} from Hugging Face Hub...")
      dataset = load_dataset(self.config.dataset)

    #  Create train/test split if not provided
      if "train" not in dataset or "test" not in dataset:
        logger.info(" No explicit train/test found — creating 90/10 split...")
        dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
      else:
        logger.info(" Dataset already contains train/test splits.")

    #  Ensure directories exist
      train_dir = Path(self.config.local_data_file_train)
      test_dir = Path(self.config.local_data_file_test)
      train_dir.mkdir(parents=True, exist_ok=True)
      test_dir.mkdir(parents=True, exist_ok=True)

    #  Save both splits in Hugging Face Arrow format
      logger.info(f" Saving train split to {train_dir} ...")
      dataset["train"].save_to_disk(str(train_dir))

      logger.info(f" Saving test split to {test_dir} ...")
      dataset["test"].save_to_disk(str(test_dir))

      logger.info(" Dataset successfully downloaded, split, and saved in Arrow format.")
      return dataset

if __name__=='__main__':
    config = ConfigurationManager()
    data_ingestion_config=config.get_data_ingestion_config()
    data_ingested=DataIngestion(config=data_ingestion_config)
    data_ingested.extract_data()
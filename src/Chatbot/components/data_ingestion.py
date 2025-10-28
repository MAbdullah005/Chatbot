from datasets import load_dataset
from src.entity.chat import DataIngestionConfig
from src.logging import logger
from pathlib import Path

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def extract_data(self):
        """
        Loads dataset from Hugging Face Hub and saves it locally as parquet.
        """
        logger.info(f"🔹 Loading dataset {self.config.dataset} from Hugging Face...")
        dataset = load_dataset(self.config.dataset, token=self.config.token_id)

        # Ensure directory exists
        Path(self.config.local_dir).mkdir(parents=True, exist_ok=True)

        # Save each split locally
        for split, data in dataset.items():
            save_path = Path(self.config.local_dir) / f"{split}.parquet"
            logger.info(f"💾 Saving {split} split to {save_path}")
            data.to_parquet(save_path)

        logger.info("✅ Dataset downloaded and saved successfully!")
        return dataset

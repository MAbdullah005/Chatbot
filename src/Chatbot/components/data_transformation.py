from src.Chatbot.constants import *
from src.Chatbot.utils.common import *
from src.Chatbot.entity import DataTransformationConfig, DataIngestionConfig
from src.Chatbot.logging  import logger
from src.Chatbot.components.data_ingestion import DataIngestion
from src.Chatbot.config.configration import ConfigurationManager
from pathlib import Path
from dotenv import load_dotenv
from datasets import load_from_disk, Dataset, concatenate_datasets,load_dataset
from transformers import AutoTokenizer
import os

load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)

    def load_data(self):
      """Load dataset from data_ingested folder"""
      logger.info(f" Loading dataset from {self.config.data_path}")

      parquet_path = os.path.join(self.config.data_path, "train.parquet")
      dataset = load_dataset("parquet", data_files=parquet_path)["train"]

      logger.info(f" Dataset loaded with {len(dataset)} samples")
      return dataset
    def prepare_conversation_pairs(self, dataset, batch_size=300000):
        logger.info("Converting conversations into input-target pairs...")

        all_datasets = []
        pairs = []
        count = 0

        for conv in dataset['conversation']:
            for i in range(len(conv) - 1):
                pairs.append({
                    "input_text": conv[i],
                    "target_text": conv[i + 1]
                })
                count += 1

                # Every batch_size pairs → flush to dataset
                if count % batch_size == 0:
                    logger.info(f"Saving batch {len(all_datasets) + 1}...")
                    batch_ds = Dataset.from_list(pairs)
                    all_datasets.append(batch_ds)
                    pairs = []  # Clear memory

        # Add remaining pairs
        if pairs:
            all_datasets.append(Dataset.from_list(pairs))

        final_ds = concatenate_datasets(all_datasets)
        logger.info(f" Created total {len(final_ds)} conversation pairs")
        return final_ds

    def tokenize_function(self, examples):
        """Tokenize text for model training"""
        inputs = [str(x) if x is not None else "" for x in examples["input_text"]]
        targets = [str(x) if x is not None else "" for x in examples["target_text"]]

        model_inputs = self.tokenizer(
            inputs,
            truncation=True,
            padding="max_length",
            max_length=128
        )

        labels = self.tokenizer(
            targets,
            truncation=True,
            padding="max_length",
            max_length=128
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def transform_and_save(self):
        """Main pipeline for data transformation with memory-safe batching"""
        dataset = self.load_data()
        clean_dataset = self.prepare_conversation_pairs(dataset)

        logger.info(" Applying tokenization in batches (memory safe)...")

        os.makedirs(self.config.transformed_data_path, exist_ok=True)
        total_samples = len(clean_dataset)
        batch_size = 50000  # Adjust based on RAM
        num_batches = (total_samples // batch_size) + 1

        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, total_samples)
            if start >= total_samples:
                break

            logger.info(f"Processing batch {i+1}/{num_batches}: samples {start}-{end}")

            subset = clean_dataset.select(range(start, end))

            tokenized_subset = subset.map(
                self.tokenize_function,
                batched=True,
                remove_columns=subset.column_names,
                num_proc=2  # Parallel processing
            )

            batch_save_path = os.path.join(self.config.transformed_data_path, f"tokenized_batch_{i+1}")
            tokenized_subset.save_to_disk(batch_save_path)
            logger.info(f" Saved tokenized batch {i+1} → {batch_save_path}")

        logger.info(" All batches tokenized and saved successfully!")

if __name__ == "__main__":
    config = ConfigurationManager()
    #data_ingestion_config=config.get_data_ingestion_config()
    #data_ingested=DataIngestion(config=data_ingestion_config)
    #data_ingested.extract_data()
    data_transformation_config = config.get_data_transformation_config()
    transformer = DataTransformation(config=data_transformation_config)
    transformer.transform_and_save()

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.Chatbot.constants import *
from src.Chatbot.utils.common import *
from src.Chatbot.entity import DataTransformationConfig
from src.Chatbot.logging import logger
from src.Chatbot.config.configration import ConfigurationManager
from datasets import load_from_disk, Dataset, concatenate_datasets
from transformers import AutoTokenizer
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)
        os.makedirs(self.config.root_dir, exist_ok=True)


    def load_dataset(self, split_name):
        """
        Load dataset split (train or test) from data_ingestion folder.
        """
        dataset_path = os.path.join(self.config.data_path, split_name)
        logger.info(f" Loading {split_name} dataset from {dataset_path} ...")

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"{split_name} dataset not found at {dataset_path}")

        dataset = load_from_disk(dataset_path)
        logger.info(f" {split_name.capitalize()} dataset loaded with {len(dataset)} samples")
        return dataset

    def prepare_conversation_pairs(self, dataset, batch_size=300000):
        """
        Convert multi-turn conversations into input-target pairs.
        """
        logger.info(" Converting conversations into input-target pairs...")

        all_datasets = []
        pairs = []
        count = 0

        for conv in dataset["conversation"]:
            for i in range(len(conv) - 1):
                pairs.append({
                    "input_text": conv[i],
                    "target_text": conv[i + 1]
                })
                count += 1

                # Save intermediate batches to prevent memory overflow
                if count % batch_size == 0:
                    batch_ds = Dataset.from_list(pairs)
                    all_datasets.append(batch_ds)
                    pairs = []
                    logger.info(f" Processed {count} samples so far...")

        # Add any remaining pairs
        if pairs:
            all_datasets.append(Dataset.from_list(pairs))

        final_ds = concatenate_datasets(all_datasets)
        logger.info(f" Created {len(final_ds)} total conversation pairs")
        return final_ds

    def tokenize_function(self, examples):
        """
        Tokenize text pairs for model training.
        """
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

    def process_and_save(self, dataset, save_dir, name):
        """
        Tokenize dataset in memory-safe batches and save each batch.
        """
        total_samples = len(dataset)
        batch_size = 50000
        num_batches = (total_samples // batch_size) + 1

        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, total_samples)
            if start >= total_samples:
                break

            logger.info(f" Processing {name} batch {i+1}/{num_batches}: samples {start}-{end}")

            subset = dataset.select(range(start, end))

            tokenized_subset = subset.map(
                self.tokenize_function,
                batched=True,
                remove_columns=subset.column_names,
                num_proc=2
            )

            batch_save_path = os.path.join(save_dir, f"{name}_tokenized_batch_{i+1}")
            tokenized_subset.save_to_disk(batch_save_path)
            logger.info(f" Saved {name} tokenized batch {i+1} → {batch_save_path}")

    def transform_and_save(self):
        """
        Main pipeline for transforming and saving both train and test datasets.
        """
        # Load train and test datasets
        train_dataset = self.load_dataset("train")
        test_dataset = self.load_dataset("test")

        # Prepare conversation pairs
        logger.info(" Preparing train conversation pairs...")
        train_clean = self.prepare_conversation_pairs(train_dataset)

        logger.info(" Preparing test conversation pairs...")
        test_clean = self.prepare_conversation_pairs(test_dataset)

        # Create directories for transformed data
        os.makedirs(self.config.root_dir, exist_ok=True)
       # train_save_dir = os.path.join(self.config.transformed_data_path, "train_batches")
        #test_save_dir = os.path.join(self.config.transformed_data_path, "test_batches")
        os.makedirs(self.config.transformed_data_path_train, exist_ok=True)
        os.makedirs(self.config.transformed_data_path_test, exist_ok=True)

        # Process and save both splits
        logger.info(" Starting tokenization for train data...")
        self.process_and_save(train_clean, self.config.transformed_data_path_train, "train")

        logger.info(" Starting tokenization for test data...")
        self.process_and_save(test_clean, self.config.transformed_data_path_test, "test")

        logger.info(" All train and test batches stokenized and saved successfully!")


if __name__ == "__main__":
    config = ConfigurationManager()
    data_transformation_config = config.get_data_transformation_config()
    transformer = DataTransformation(config=data_transformation_config)
    transformer.transform_and_save()

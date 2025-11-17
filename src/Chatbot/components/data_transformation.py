import sys
import os
import ast
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.Chatbot.constants import *
from src.Chatbot.utils.common import *
from src.Chatbot.entity import DataTransformationConfig
from src.Chatbot.logging import logger
from src.Chatbot.config.configration import ConfigurationManager
from datasets import load_from_disk, Dataset, concatenate_datasets
from transformers import GPT2Tokenizer
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.config.tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # GPT2 doesn’t have a PAD token
        os.makedirs(self.config.root_dir, exist_ok=True)

    # ------------------------------------------------------------------
    def clean_text(self, text):
        """Extract only 'content' field text and remove role/user wrappers."""
        if text is None:
            return ""

        try:
            if isinstance(text, dict):
                return text.get("content", "").strip()

            if isinstance(text, str):
                text = text.strip()
                if text.startswith("{") or text.startswith("["):
                    obj = ast.literal_eval(text)
                    if isinstance(obj, dict):
                        return obj.get("content", "")
                    elif isinstance(obj, list):
                        for item in obj:
                            if isinstance(item, dict) and "content" in item:
                                return item["content"]
                elif "'content':" in text:
                    part = text.split("'content':", 1)[1]
                    part = part.split("'role':")[0]
                    return part.strip().strip("'\" ")
        except Exception:
            pass

        return str(text).strip()

    # ------------------------------------------------------------------
    def load_dataset(self, split_name):
        dataset_path = os.path.join(self.config.data_path, split_name)
        logger.info(f"📥 Loading {split_name} dataset from {dataset_path} ...")

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"{split_name} dataset not found at {dataset_path}")

        dataset = load_from_disk(dataset_path)
        logger.info(f"✅ {split_name.capitalize()} dataset loaded with {len(dataset)} samples")
        return dataset

    # ------------------------------------------------------------------
    def prepare_conversation_pairs(self, dataset, batch_size=300000):
        """
        Convert multi-turn conversation into single 'User: ... Assistant: ...' text entries.
        """
        logger.info("🔄 Converting conversations into prompt-response pairs...")
        all_datasets = []
        pairs = []
        count = 0

        for sample in dataset:
            conv = sample["conversation"]
           # print(sample["conversation"][0])
            if not isinstance(conv, list):
                continue

            for i in range(len(conv) - 1):
                if conv[i]["role"] == "user" and conv[i + 1]["role"] == "assistant":
                    user_msg = self.clean_text(conv[i])
                    bot_msg = self.clean_text(conv[i + 1])
                    if not user_msg or not bot_msg:
                        continue

                    # ✅ Single text field (no input_text / target_text separation)
                    text = f"User: {user_msg}\nAssistant: {bot_msg}"
                    pairs.append({"text": text})
                    count += 1

                    if count % batch_size == 0:
                        batch_ds = Dataset.from_list(pairs)
                        all_datasets.append(batch_ds)
                        pairs = []
                        logger.info(f"🧩 Processed {count} samples so far...")

        if pairs:
            all_datasets.append(Dataset.from_list(pairs))

        final_ds = concatenate_datasets(all_datasets)
        logger.info(f" Created {len(final_ds)} total conversation samples")
        logger.info(f" Example sample:\n{final_ds[0]['text']}")
        return final_ds

    # ------------------------------------------------------------------
    def tokenize_function(self, examples):
        texts = [str(x) if x else "" for x in examples["text"]]
        tokens = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=256,
        )

        # GPT-2 uses input_ids as both input and labels
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    # ------------------------------------------------------------------
    def process_and_save(self, dataset, save_dir, name):
        total_samples = len(dataset)
        batch_size = 50000
        num_batches = (total_samples // batch_size) + 1

        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, total_samples)
            if start >= total_samples:
                break

            logger.info(f"⚙️ Processing {name} batch {i+1}/{num_batches}: samples {start}-{end}")

            subset = dataset.select(range(start, end))

            tokenized_subset = subset.map(
                self.tokenize_function,
                batched=True,
                remove_columns=subset.column_names,  # removes 'text' after tokenization
                num_proc=2
            )

            batch_save_path = os.path.join(save_dir, f"{name}_tokenized_batch_{i+1}")
            tokenized_subset.save_to_disk(batch_save_path)
            logger.info(f"💾 Saved {name} tokenized batch {i+1} → {batch_save_path}")

    # ------------------------------------------------------------------
    def transform_and_save(self):
        train_dataset = self.load_dataset("train")
        test_dataset = self.load_dataset("test")

        logger.info("🧠 Preparing train conversation pairs...")
        train_clean = self.prepare_conversation_pairs(train_dataset)

        logger.info("🧠 Preparing test conversation pairs...")
        test_clean = self.prepare_conversation_pairs(test_dataset)

        os.makedirs(self.config.transformed_data_path_train, exist_ok=True)
        os.makedirs(self.config.transformed_data_path_test, exist_ok=True)

        logger.info("🚀 Starting tokenization for train data...")
        self.process_and_save(train_clean, self.config.transformed_data_path_train, "train")

        logger.info("🚀 Starting tokenization for test data...")
        self.process_and_save(test_clean, self.config.transformed_data_path_test, "test")

        logger.info("✅ All train and test batches tokenized and saved successfully!")



if __name__ == "__main__":
    config = ConfigurationManager()
    data_transformation_config = config.get_data_transformation_config()
    transformer = DataTransformation(config=data_transformation_config)
    transformer.transform_and_save()
    

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Trainer
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from datasets import load_from_disk, concatenate_datasets
from src.Chatbot.utils.common import *
from src.Chatbot.entity import ModelTrainerConfig
from src.Chatbot.config.configration import ConfigurationManager
from src.Chatbot.logging import logger
from pathlib import Path


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        root=self.config.root_dir
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_ckpt)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(config.model_ckpt)

    def load_batches_from_dir(self, batch_dirs,dir_path):
        """
        Load and combine all batches from a given directory.
        """
        logger.info(f" Loading batches from: {dir_path}")
        dataset_list = []

        # batch_dirs = sorted(Path(dir_path).glob("**/tokenized_batch_*"))
        if not batch_dirs:
            raise FileNotFoundError(f"No tokenized batches found in {dir_path}")

        for batch_dir in batch_dirs:
            dataset = load_from_disk(batch_dir)
            dataset_list.append(dataset)
            logger.info(f" Loaded {batch_dir} with {len(dataset)} samples")

        full_dataset = concatenate_datasets(dataset_list)
        logger.info(f" Combined total samples: {len(full_dataset)}")
        return full_dataset

    def train(self):
        logger.info(" Starting model training pipeline...")

        # === Load Train & Test Datasets ===
       # train_dir = os.path.join(self.config.data_path_train, "train_batches")
       # test_dir = os.path.join(self.config.data_path_test, "test_batches")
        create_directories([self.config.root_dir])
        
        batch_dirs_train = sorted(Path(self.config.data_path_train).glob("**/train_tokenized_batch_*"))
        batch_dirs_test=sorted(Path(self.config.data_path_test).glob("**/test_tokenized_batch_*"))

        train_data = self.load_batches_from_dir(batch_dirs_train,self.config.data_path_train)
        eval_data = self.load_batches_from_dir(batch_dirs_test,self.config.data_path_test)
        train_data=train_data.select(range(1000))
        eval_data=eval_data.select(range(500))

        logger.info(f" Train samples: {len(train_data)} | Eval samples: {len(eval_data)}")

        # === Training Arguments ===
        training_args = TrainingArguments(
            output_dir=self.config.root_dir,
            eval_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=3,
            weight_decay=0.01,
            save_total_limit=2,
            logging_dir=f"{self.config.root_dir}/logs",
            logging_steps=100,
            push_to_hub=False
        )

        # === Trainer ===
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=eval_data,
            tokenizer=self.tokenizer
        )

        # === Training Start ===
        logger.info(" Training model...")
        trainer.train()

        # === Save Model ===
        trainer.save_model(self.config.model_save_path)
        logger.info(f" Model saved at: {self.config.model_save_path}")


if __name__ == "__main__":
    config = ConfigurationManager()
    trainer_config = config.get_model_trainer_config()
    trainer = ModelTrainer(config=trainer_config)
    trainer.train()

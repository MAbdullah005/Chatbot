from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Trainer
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from datasets import load_from_disk, concatenate_datasets
from src.Chatbot.entity import ModelTrainerConfig
from src.Chatbot.config.configration import ConfigurationManager
from src.Chatbot.logging import logger
from pathlib import Path


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_ckpt)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(config.model_ckpt)

    def load_all_batches(self):
        logger.info(" Loading all transformed dataset batches...")
        dataset_list = []

        for batch_dir in Path(self.config.data_path).glob("tokenized_batch_*"):
            dataset = load_from_disk(batch_dir)
            dataset_list.append(dataset)

        full_dataset = concatenate_datasets(dataset_list)
        logger.info(f" Total samples loaded: {len(full_dataset)}")
        return full_dataset

    def train(self):
        logger.info(" Starting model training...")
        dataset = self.load_all_batches()
        dataset = dataset.select(range(1000))  # just first 20K samples


        # Split dataset
        dataset = dataset.train_test_split(test_size=0.1)
        train_data = dataset["train"]
        eval_data = dataset["test"]

        # Training arguments
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

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=eval_data,
            tokenizer=self.tokenizer
        )

        trainer.train()
        trainer.save_model(self.config.model_save_path)
        logger.info(f" Model saved at {self.config.model_save_path}")


if __name__ == "__main__":
    config =ConfigurationManager()
    trainer_config = config.get_model_trainer_config()
    trainer = ModelTrainer(config=trainer_config)
    trainer.train()
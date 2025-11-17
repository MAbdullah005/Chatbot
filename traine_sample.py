from transformers import AutoModelForCausalLM, GPT2Tokenizer, Trainer, TrainingArguments, TrainerCallback
import math
import sys
import os
from pathlib import Path
from datasets import load_from_disk, concatenate_datasets

#Add project root to sys.path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.Chatbot.utils.common import create_directories
from src.Chatbot.entity import ModelTrainerConfig
from src.Chatbot.config.configration import ConfigurationManager
from src.Chatbot.logging import logger

class ChatbotMonitorCallback(TrainerCallback):
#"""Custom callback to monitor loss, perplexity, and generate sample replies."""
 def init(self, tokenizer, model, log_path):
   self.tokenizer = tokenizer
   self.model = model
   self.log_path = log_path

 def on_epoch_end(self, args, state, control, **kwargs):
    # Get current losses
    train_loss = state.log_history[-1].get("loss", None)
    eval_loss = state.log_history[-1].get("eval_loss", None)

    # Compute perplexity
    ppl = math.exp(eval_loss) if eval_loss else None

    # Generate a sample reply
    prompt = "Hello, how are you today?"
    inputs = self.tokenizer(prompt, return_tensors="pt")
    output = self.model.generate(
        **inputs,
        max_new_tokens=50,
        temperature=0.8,
        top_p=0.9,
        do_sample=True,
        pad_token_id=self.tokenizer.eos_token_id
    )
    reply = self.tokenizer.decode(output[0], skip_special_tokens=True)

    # Log info to console
    logger.info(f"Epoch {state.epoch:.0f} Summary:")
    logger.info(f"  Train loss: {train_loss}")
    logger.info(f"  Eval loss: {eval_loss}")
    logger.info(f"  Perplexity: {ppl}")
    logger.info(f"  Sample reply: {reply}")

    # Save to file
    with open(self.log_path, "a", encoding="utf-8") as f:
        f.write(f"\n--- Epoch {state.epoch:.0f} ---\n")
        f.write(f"Train loss: {train_loss}\n")
        f.write(f"Eval loss: {eval_loss}\n")
        f.write(f"Perplexity: {ppl}\n")
        f.write(f"Sample reply: {reply}\n")

class ModelTrainer:
  def init(self, config: ModelTrainerConfig):
   self.config = config
   self.tokenizer = GPT2Tokenizer.from_pretrained(config.model_ckpt)
   self.model = AutoModelForCausalLM.from_pretrained(config.model_ckpt)   
   self.tokenizer.pad_token = self.tokenizer.eos_token
   self.model.config.pad_token_id = self.tokenizer.eos_token_id

  def load_batches_from_dir(self, batch_dirs, dir_path):
      logger.info(f" Loading batches from: {dir_path}")
      dataset_list = []

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
      create_directories([self.config.root_dir])

      batch_dirs_train = sorted(Path(self.config.data_path_train).glob("**/train_tokenized_batch_*"))
      batch_dirs_test = sorted(Path(self.config.data_path_test).glob("**/test_tokenized_batch_*"))

      train_data = self.load_batches_from_dir(batch_dirs_train, self.config.data_path_train)
      eval_data = self.load_batches_from_dir(batch_dirs_test, self.config.data_path_test)

      logger.info(f" Train samples: {len(train_data)} | Eval samples: {len(eval_data)}")

      train_data = train_data.select(range(min(30000, len(train_data))))
      eval_data = eval_data.select(range(min(3000, len(eval_data))))

    # Training arguments
      training_args = TrainingArguments(
        output_dir=self.config.root_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=2,
        logging_dir=f"{self.config.root_dir}/logs",
        logging_steps=50,
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False
    )

      log_path = Path(self.config.root_dir) / "training_progress.txt"

      trainer = Trainer(
        model=self.model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        tokenizer=self.tokenizer,
        callbacks=[ChatbotMonitorCallback(self.tokenizer, self.model, log_path)]
    )

      logger.info(" Training model...")
      trainer.train()

      model_save_path = Path(self.config.model_save_path)
      model_save_path.mkdir(parents=True, exist_ok=True)
      trainer.save_model(str(model_save_path))
      logger.info(f" Model saved at: {model_save_path}")


if __name__ == "__main__":
  config = ConfigurationManager()
  trainer_config = config.get_model_trainer_config()
  trainer = ModelTrainer(config=trainer_config)
  trainer.train()
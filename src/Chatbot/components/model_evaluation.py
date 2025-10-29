import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_from_disk,concatenate_datasets
from evaluate import load as load_metric
from src.Chatbot.logging import logger
from src.Chatbot.config.configration import ConfigurationManager
from src.Chatbot.entity import ModelEvaluationConfig



class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

        # Automatically detect latest checkpoint (if inside model_trainer)
        model_path = Path(self.config.model_path)
        checkpoints = list(model_path.glob("checkpoint-*"))
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=os.path.getmtime)
            model_dir = latest_checkpoint
            logger.info(f" Found latest checkpoint: {latest_checkpoint}")
        else:
            model_dir = model_path
            logger.info(f" No checkpoint found, loading model from: {model_dir}")

        logger.info(" Loading model and tokenizer...")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

    def load_data(self):
        """Load and combine all tokenized batches"""
        logger.info(f"Loading tokenized batches from {self.config.data_path}")

        dataset_list = []
        for batch_dir in Path(self.config.data_path).glob("tokenized_batch_*"):
            dataset = load_from_disk(batch_dir)
            dataset_list.append(dataset)

        full_dataset = concatenate_datasets(dataset_list)
        logger.info(f"✅ Loaded total {len(full_dataset)} samples.")
        return full_dataset

    def evaluate(self):
        """Generate predictions and compute BLEU & ROUGE scores"""
        eval_data = self.load_data()
        dataset_split = eval_data.train_test_split(test_size=0.1, seed=42)
        eval_data = dataset_split["test"]

        preds = []
        refs = []
        eval_data = eval_data.select(range(1000))


        logger.info(" Generating predictions...")
        for example in eval_data:
          # input_text = example["input_text"]
            input_text = self.tokenizer.decode(example["input_ids"], skip_special_tokens=True)
            target_text = self.tokenizer.decode(example["labels"], skip_special_tokens=True)

            #target_text = example["target_text"]

            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            )

            outputs = self.model.generate(
                **inputs,
                max_length=128,
                num_beams=4,
                early_stopping=True
            )

            pred_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            preds.append(pred_text)
            refs.append(target_text)

        # Compute BLEU and ROUGE metrics
        logger.info(" Calculating BLEU and ROUGE scores...")
        rouge = load_metric("rouge")
        bleu = load_metric("bleu")

        bleu_score = bleu.compute(
            predictions=[p.split() for p in preds],
            references=[[r.split()] for r in refs]
        )["bleu"]

        rouge_score = rouge.compute(
            predictions=preds,
            references=refs,
            rouge_types=["rouge1", "rouge2", "rougeL"]
        )

        logger.info(f" BLEU Score: {bleu_score:.4f}")
        logger.info(f" ROUGE Scores: {rouge_score}")

        # Save metrics
        os.makedirs(Path(self.config.root_dir), exist_ok=True)
        metrics_path = Path(self.config.metric_file_name)
        with open(metrics_path, "w", encoding="utf-8") as f:
            f.write("metric,value\n")
            f.write(f"BLEU,{bleu_score:.4f}\n")
            for key, value in rouge_score.items():
                f.write(f"{key},{value.mid.fmeasure:.4f}\n")

        logger.info(f" Metrics saved to: {metrics_path}")
        return {"bleu": bleu_score, "rouge": rouge_score}


if __name__ == "__main__":
    config = ConfigurationManager()
    eval_config = config.get_model_evaluation_config()

    evaluator = ModelEvaluation(config=eval_config)
    results = evaluator.evaluate()

    print(" Evaluation complete.")
    print(results)

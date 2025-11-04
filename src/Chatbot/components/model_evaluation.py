import os
import sys
import csv
import ast
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_from_disk, concatenate_datasets
import sacrebleu
import numpy as np
from evaluate import load as load_metric

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from src.Chatbot.logging import logger
from src.Chatbot.config.configration import ConfigurationManager
from src.Chatbot.entity import ModelEvaluationConfig


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        model_path = Path(self.config.model_path)

        # Detect latest checkpoint if exists
        checkpoints = list(model_path.glob("checkpoint-*"))
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=os.path.getmtime)
            model_dir = latest_checkpoint
            logger.info(f" Found latest checkpoint: {latest_checkpoint}")
        else:
            model_dir = model_path
            logger.info(f" No checkpoint found, loading model from: {model_dir}")

        # Load model and tokenizer
        logger.info(" Loading model and tokenizer...")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

    def load_data(self):
        """Load and combine all tokenized data batches."""
        logger.info(f" Loading tokenized batches from {self.config.data_path}")
        dataset_list = []
        for batch_dir in Path(self.config.data_path).glob("test_tokenized_batch_*"):
            dataset = load_from_disk(batch_dir)
            dataset_list.append(dataset)

        if not dataset_list:
            raise ValueError(f" No tokenized batches found in {self.config.data_path}")

        full_dataset = concatenate_datasets(dataset_list)
        logger.info(f" Loaded total {len(full_dataset)} samples.")
        return full_dataset

    def _clean_text(self, text):
        """Extract 'content' field cleanly and remove noise."""
        if text is None:
            return ""
        try:
            if isinstance(text, str):
                text = text.strip()
                if text.startswith("{") or text.startswith("["):
                    obj = ast.literal_eval(text)
                    if isinstance(obj, dict) and "content" in obj:
                        return obj["content"]
                    if isinstance(obj, list):
                        # find first dict with 'content'
                        for item in obj:
                            if isinstance(item, dict) and "content" in item:
                                return item["content"]
                        return " ".join(map(str, obj))
                elif "'content':" in text:
                    # Extract between quotes after 'content':
                    parts = text.split("'content':")
                    if len(parts) > 1:
                        segment = parts[1]
                        # remove potential role/user fragments
                        segment = segment.split("'role':")[0]
                        return segment.strip().strip("'\" ")
            elif isinstance(text, dict):
                return text.get("content", "")
        except Exception:
            pass
        return str(text).strip()
    


    def compute_metrics(self,eval_pred):
      predictions, labels = eval_pred

      if isinstance(predictions, tuple):
        predictions = predictions[0]

      if predictions.ndim == 3:
          predictions = np.argmax(predictions, axis=-1)

      labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)

      decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
      decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

      decoded_preds = [pred.strip() for pred in decoded_preds]
      decoded_labels = [label.strip() for label in decoded_labels]

      bleu = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels]).score
      rouge = load_metric("rouge")
      rouge_scores = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    # ✅ handle both float and object return types
      if isinstance(list(rouge_scores.values())[0], float):
          result = {"bleu": bleu, **{k: v * 100 for k, v in rouge_scores.items()}}
      else:
          result = {"bleu": bleu, **{k: v.mid.fmeasure * 100 for k, v in rouge_scores.items()}}

      return result






    def evaluate(self):
        """Generate predictions, compute metrics, and save outputs."""
        eval_data = self.load_data()
        eval_data = eval_data.select(range(min(20, len(eval_data))))

        preds, refs = [], []

        logger.info(" Generating predictions...")
        for i, example in enumerate(eval_data):
            if (i + 1) % 10 == 0:
                logger.info(f"Generated {i + 1} samples...")

            input_text = self.tokenizer.decode(example["input_ids"], skip_special_tokens=True)
            target_text = self.tokenizer.decode(example["labels"], skip_special_tokens=True)

            input_clean = self._clean_text(input_text)
            target_clean = self._clean_text(target_text)

            if not input_clean.strip() or not target_clean.strip():
                continue

            inputs = self.tokenizer(
                input_clean,
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
            refs.append(target_clean)

        # Compute metrics
         # Compute metrics
        logger.info(" Calculating BLEU and ROUGE scores...")
        bleu = load_metric("bleu")
        rouge = load_metric("rouge")

        # ✅ Correct BLEU format (no token split)
        bleu_input = {"predictions": preds, "references": [[r] for r in refs]}
        bleu_score = bleu.compute(**bleu_input)["bleu"]

        rouge_score = rouge.compute(
            predictions=preds,
            references=refs,
            rouge_types=["rouge1", "rouge2", "rougeL"]
        )

        logger.info(f" BLEU Score: {bleu_score:.4f}")
        logger.info(f" ROUGE Scores: {rouge_score}")
        logger.info(f" BLEU Score: {bleu_score:.4f}")
        logger.info(f" ROUGE Scores: {rouge_score}")

        # Save metrics
        os.makedirs(Path(self.config.root_dir), exist_ok=True)
        metrics_path = Path(self.config.metric_file_name)
        with open(metrics_path, "w", encoding="utf-8") as f:
            f.write("metric,value\n")
            f.write(f"BLEU,{bleu_score:.4f}\n")
            for key, value in rouge_score.items():
                score = value.mid.fmeasure if hasattr(value, "mid") else float(value)
                f.write(f"{key},{score:.4f}\n")
        logger.info(f" Metrics saved to: {metrics_path}")

        # Save predictions
        predictions_path = Path(self.config.root_dir) / "predictions.csv"
        with open(predictions_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Input", "Target", "Prediction"])
            for i in range(min(10, len(preds))):
                writer.writerow([
                    self._clean_text(self.tokenizer.decode(eval_data[i]["input_ids"], skip_special_tokens=True)),
                    refs[i],
                    preds[i]
                ])
        logger.info(f" Sample predictions saved to: {predictions_path}")

        return {"bleu": bleu_score, "rouge": rouge_score}


if __name__ == "__main__":
    config = ConfigurationManager()
    eval_config = config.get_model_evaluation_config()
    evaluator = ModelEvaluation(config=eval_config)
    results = evaluator.evaluate()

    print("\n Evaluation complete.")
    print(results)

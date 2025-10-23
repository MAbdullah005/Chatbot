from datasets import load_dataset, Dataset
import pandas as pd

# ✅ FIX: Use the preprocessed Parquet version
dataset = load_dataset("Amod/daily_dialog_parquet")

def extract_conversations(split):
    dialogs = []
    for dialog in split:
        utterances = dialog["dialog"]
        for i in range(len(utterances) - 1):
            dialogs.append({
                "input_text": utterances[i],
                "target_text": utterances[i + 1]
            })
    return dialogs

train_pairs = extract_conversations(dataset["train"])
validation_pairs = extract_conversations(dataset["validation"])
test_pairs = extract_conversations(dataset["test"])

train_df = pd.DataFrame(train_pairs)
val_df = pd.DataFrame(validation_pairs)
test_df = pd.DataFrame(test_pairs)

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

train_dataset.save_to_disk("data/chatbot/train")
val_dataset.save_to_disk("data/chatbot/validation")
test_dataset.save_to_disk("data/chatbot/test")

print("✅ DailyDialog dataset preprocessed and saved to disk!")

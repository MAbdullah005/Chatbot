import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class ChatbotPredictor:
    def __init__(self, model_path):
        print(f"Loading model from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    def predict(self, user_input):
        inputs = self.tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
        outputs = self.model.generate(
            **inputs,
            max_length=128,
            num_beams=4,
            early_stopping=True
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    # Path to your fine-tuned model folder (checkpoint)
    model_path = "artifacts\model_trainer\checkpoint-675"  # replace with your actual checkpoint folder name

    bot = ChatbotPredictor(model_path)

    print("🤖 Chatbot is ready! Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break
        response = bot.predict(user_input)
        print("Bot:", response)

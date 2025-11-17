from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from transformers import T5Tokenizer, AutoModelForCausalLM,GPT2Tokenizer
import torch

app = FastAPI()

# for get static 
app.mount("/static", StaticFiles(directory="static"), name="static")

# Template directory (for HTML)
templates = Jinja2Templates(directory="templates")

# Load trained model
MODEL_DIR = "artifacts/model_trainer/gpt2-trained"  # your saved model folder
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)

class ChatRequest(BaseModel):
    text: str

@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(request: ChatRequest):
    user_input = request.text.strip()
    if not user_input:
        return {"reply": "Please say something."}

    input_ids = tokenizer.encode(user_input, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=100,
            num_beams=4,
            early_stopping=True,
            temperature=0.8,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id

        )

    reply = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"reply": reply}

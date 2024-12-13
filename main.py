import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi import FastAPI
from pydantic import BaseModel
import bitsandbytes as bnb

app = FastAPI()

class Input(BaseModel):
    prompt: str

model_path = "C:\\Users\\ADMIN\\Desktop\\Llama3_8B_test\\Llama3_8B_test_model"

# Load the model and tokenizer
try:
    loaded_tokenizer = AutoTokenizer.from_pretrained(model_path)

    if loaded_tokenizer.pad_token is None:
        loaded_tokenizer.pad_token = loaded_tokenizer.eos_token

    loaded_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        load_in_8bit=True,
        device_map="auto"
    )
    print("Model loaded successfully (in 8-bit)!")
except Exception as e:
    print(f"An error occurred: {e}")

@app.post("/generate")
async def generate(input: Input):
    batch_prompts = [input.prompt]

    inputs = loaded_tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(loaded_model.device)

    with torch.no_grad():
        outputs = loaded_model.generate(
            **inputs,
            max_new_tokens=1500,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            top_k=40,
            early_stopping=True
        )

    decoded_output = loaded_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": decoded_output}

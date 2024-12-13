import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import pandas as pd
from tqdm.auto import tqdm
from peft import LoraConfig, PeftModel

# Paths to your CSV files
train_csv = "C:\\Users\\ADMIN\\Desktop\\ielts-bots\\train.csv"
test_csv = "C:\\Users\\ADMIN\\Desktop\\ielts-bots\\test.csv"

# Load your CSV files into Pandas DataFrames
train_data = pd.read_csv(train_csv)
test_data = pd.read_csv(test_csv)

# Convert the DataFrames to Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_data)
test_dataset = Dataset.from_pandas(test_data)

# Path to your model files
model_path = "C:\\Users\\ADMIN\\Desktop\\Llama3_8B_test\\Llama3_8B_test_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    load_in_8bit=True,
    device_map="auto"
)

# Configure LORA with correct target modules
lora_config = LoraConfig(
    target_modules=[
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.v_proj",
        "model.layers.0.self_attn.o_proj"
    ], 
    r=4,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none"
)

# Attach LORA to the model
model = PeftModel(model, lora_config)

# Preprocess function to tokenize inputs and targets
def preprocess_function(examples):
    inputs = [f"{p} {e}" for p, e in zip(examples['prompt'], examples['essay'])]
    targets = [f"{ev} {b}" for ev, b in zip(examples['evaluation'], examples['band'])]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply the preprocess function to the datasets
print("Tokenizing train dataset...")
tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True, desc="Tokenizing train dataset")
print("Tokenizing test dataset...")
tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True, desc="Tokenizing test dataset")

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    save_total_limit=1,
    save_steps=500,
    fp16=True
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
)

# Fine-tuning the model with progress bar
print("Starting training...")
trainer.train()

# Save the fine-tuned model
print("Saving the fine-tuned model...")
model.save_pretrained("./Fine_tuned_Llama3_8B")
tokenizer.save_pretrained("./Fine_tuned_Llama3_8B")

print("Model fine-tuning complete and saved successfully!")

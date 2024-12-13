import os
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "meta-llama/Llama-3.1-8B-Instruct"

# 1. Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# 2. Get the current directory
current_directory = os.getcwd()

# 3. Create the save directory (if it doesn't exist)
save_directory = os.path.join(current_directory, "Llama3_8B_test_model")
os.makedirs(save_directory, exist_ok=True) # exist_ok avoids errors if the directory exists

# 4. Save the model and tokenizer
tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)

print(f"Model and tokenizer saved to: {save_directory}")

# --- Loading from the same directory ---
loaded_tokenizer = AutoTokenizer.from_pretrained("./my_local_llama_model") #relative path
loaded_model = AutoModelForCausalLM.from_pretrained("./my_local_llama_model") #relative path

print("Model loaded successfully from current directory!")

#Example usage
prompt = "What is the capital of France?"
inputs = loaded_tokenizer(prompt, return_tensors="pt")
outputs = loaded_model.generate(**inputs)
print(loaded_tokenizer.decode(outputs[0], skip_special_tokens=True))
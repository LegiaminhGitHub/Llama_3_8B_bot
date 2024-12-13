from llama import LLaMA

# Initialize model
model = LLaMA("Llama3.1-8B-Instruct")

# Generate text
output = model("Your prompt here")

print(output)
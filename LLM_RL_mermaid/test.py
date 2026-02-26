import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 1. Configuration
base_model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
adapter_path = "./llama-3-8b-custom-adapter" # Where your trained skills are saved

print("Loading tokenizer and base model...")
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

# 2. Load the base model in 4-bit to save memory
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    load_in_4bit=True,
    device_map="auto",
)

# 3. Merge the base model with your newly trained Mermaid adapter
print("Applying trained adapters...")
model = PeftModel.from_pretrained(base_model, adapter_path)

# 4. Create your test prompt using the exact format we trained on
print("Generating Mermaid chart...\n")
system_prompt = "You are an expert system architect. Convert the user's description into valid Mermaid.js code."
user_request = "A user logs into the website. The system validates the password. If valid, the user goes to the dashboard. If invalid, the user goes to an error page."

prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_request}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

# 5. Generate the output!
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.1) # Low temperature for accurate code

# Print the result and strip out the prompt we fed it
full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(full_response)
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer, SFTConfig
import wandb
import random
import os
from dotenv import load_dotenv
import math
from transformers import TrainerCallback

# Load variables from .env file
load_dotenv()

# ==========================================
# 1. CONFIGURATION & WANDB SETUP
# ==========================================
# TODO: Put your actual HF token here, or log in via CLI later
HF_TOKEN = os.getenv("HF_TOKEN")

# Initialize Weights & Biases
wandb.init(
    project="llama-3-post-training", 
    name="mermaid-bidirectional-finetune" # Updated name!
)

# ==========================================
# 2. LOAD DATASET & FORMATTING
# ==========================================
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    token=HF_TOKEN
)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("Loading Mermaid dataset...")
# Load the Mermaid dataset (using the first 2000 rows for a solid test run)
dataset = load_dataset("Celiadraw/text-to-mermaid", split="train[:2000]")
# The Bidirectional Formatting Magic
def format_chat_template(row):
    # The dataset has 'text' (the plain english) and 'mermaid' (the code) columns.
    values = list(row.values())
    
    # Usually: Column 0 is English, Column 1 is Mermaid
    # (If your CSV is flipped, just swap 0 and 1 here)
    english_text = str(values[0])
    mermaid_code = str(values[1])
    
    if random.choice([0, 1]) == 0:
        # Task: Generate code from description
        sys_prompt = "You are an expert system architect. Convert the user's description into valid Mermaid.js code."
        user_input = english_text
        assistant_output = f"```mermaid\n{mermaid_code}\n```"
    else:
        # Task: Explain code in English
        sys_prompt = "You are an expert system architect. Explain the logic of the provided Mermaid.js code in well explained plain English."
        user_input = f"```mermaid\n{mermaid_code}\n```"
        assistant_output = english_text

    # Apply the Llama-3 Chat Template format
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": assistant_output}
    ]
    
    # Tokenize the conversation
    return {
        "prompt": tokenizer.apply_chat_template(
            messages[:-1],  # system + user
            tokenize=False,
            add_generation_prompt=True
        ),
        "completion": messages[-1]["content"]
    }

dataset = dataset.map(format_chat_template)

# ==========================================
# 3. LOAD MODEL IN 4-BIT (QLoRA)
# ==========================================
print("Loading model in 4-bit...")
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# This config is the secret to fitting 8B parameters on a 24GB GPU
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    token=HF_TOKEN
)
model.config.use_cache = False
model.gradient_checkpointing_enable()

# ==========================================
# 4. SET UP PEFT / LoRA ADAPTERS
# ==========================================
print("Setting up LoRA...")
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16, # The rank (size) of the adapters. 16 is a great default.
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], # Target the attention mechanisms
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)


# ==========================================
# 6. START TRAINING
# ==========================================
print("Starting SFT Trainer with Perplexcity logging...")
class PerplexityLoggerCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            try:
                # Perplexity = e^Loss
                logs["perplexity"] = math.exp(logs["loss"])
            except OverflowError:
                logs["perplexity"] = float("inf")

sft_config = SFTConfig(
    output_dir="./output",
    max_length=1024,  # <--- Change 'max_seq_length' to 'max_length'
    packing=False,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    report_to="wandb",
    # ... any other args ...
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    args=sft_config,
    processing_class=tokenizer, # Note: Latest TRL also prefers 'processing_class' over 'tokenizer'
    callbacks=[PerplexityLoggerCallback()]
)
trainer.train()

# ==========================================
# 7. SAVE THE RESULTS
# ==========================================
print("Training complete! Saving adapters...")
# 1. Define your final name
final_model_path = "./mermaid-llama-v1"

# 2. Save the specialized LoRA adapters
trainer.save_model(final_model_path)

# 3. Save the tokenizer (crucial for Hugging Face to know how to read your model)
tokenizer.save_pretrained(final_model_path)

print(f"✨ Training complete! Final model saved to: {final_model_path}")
wandb.finish()
print("All done. You can now terminate the RunPod GPU!")


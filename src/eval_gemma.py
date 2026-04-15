import torch
import json
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import os

# -------------------------
# CONFIGURATION
# -------------------------
BASE_MODEL = "google/gemma-2-2b-it"      # HuggingFace path or local folder
LORA_CHECKPOINT = "./gemma-lora-checkpoints/checkpoint-3500" 
USE_LORA = False 

EVAL_SAMPLES = 2000                  # Stop after this many samples
SAVE_INTERVAL = 100                  # Calculate means and save every 100 iterations
OUTPUT_FILE = "doktorsitesi_gemma_eval_results.jsonl"
SUMMARY_FILE = "doktorsitesi_gemma_eval_summary.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# LOAD TOKENIZER & APPLY GEMMA TEMPLATE
# -------------------------
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

# Gemma does not natively support the "system" role. 
# This custom template dynamically merges the system prompt into the first user turn.
tokenizer.chat_template = "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% set messages = messages[1:] %}{% else %}{% set system_message = '' %}{% endif %}{% for message in messages %}{% if loop.first %}<start_of_turn>user\n{{ system_message + '\n\n' if system_message }}{{ message['content'] }}<end_of_turn>\n{% elif message['role'] == 'user' %}<start_of_turn>user\n{{ message['content'] }}<end_of_turn>\n{% elif message['role'] == 'assistant' %}<start_of_turn>model\n{{ message['content'] }}<end_of_turn>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}"

# -------------------------
# LOAD BASE MODEL
# -------------------------
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",      # High-quality 4-bit format
    bnb_4bit_compute_dtype=torch.bfloat16, # Speeds up computation
    bnb_4bit_use_double_quant=True, # Saves even more VRAM
)

print(f"Loading base model ({BASE_MODEL}) in 4-bit NF4...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, 
    trust_remote_code=True,
    quantization_config=quant_config,
    device_map="auto", # Automatically handles GPU placement (Do NOT use .to(device))
    attn_implementation="sdpa"
)

if USE_LORA:
    print(f"Loading LoRA adapter from {LORA_CHECKPOINT}...")
    model = PeftModel.from_pretrained(base_model, LORA_CHECKPOINT)
else:
    print("Skipping LoRA, using base model only...")
    model = base_model

model.eval()

# -------------------------
# LOAD DATASET
# -------------------------
print("Loading dataset 'alibayram/doktorsitesi'...")
dataset = load_dataset("alibayram/doktorsitesi", split="train", streaming=True)

# -------------------------
# EVALUATION LOOP
# -------------------------
print(f"\n=== Starting Evaluation on {EVAL_SAMPLES} samples ===")

results = []
losses = []

# Open file in append mode to save incrementally
with open(OUTPUT_FILE, "a", encoding="utf-8") as f_out:
    
    # Iterate through the dataset
    for i, data in enumerate(tqdm(dataset, total=EVAL_SAMPLES)):
        if i >= EVAL_SAMPLES:
            break
            
        # Extract using the exact column names
        question = data.get("question_content", "")
        answer = data.get("question_answer", "")
        specialty = data.get("doctor_specialty", "")
        title = data.get("doctor_title", "")
        
        if not question or not answer:
            continue

        system_prompt = f"Sen bir {title} {specialty} uzmanısın. Hasta sorusunu profesyonelce yanıtla."

        # 1. Format the Prompt (System + Question) to find its length
        prompt_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        
        # 2. Format the Full Text (System + Question + Answer)
        full_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ]
        full_text = tokenizer.apply_chat_template(full_messages, tokenize=False)
        
        # 3. Tokenize with strict truncation
        MAX_LEN = 1536 # Adjust based on your VRAM (1536 or 2048 are safe bets)
        
        prompt_ids = tokenizer(
            prompt_text, return_tensors="pt", truncation=True, max_length=MAX_LEN
        )["input_ids"]
        
        full_ids = tokenizer(
            full_text, return_tensors="pt", truncation=True, max_length=MAX_LEN
        )["input_ids"]
        
        # 4. Create Labels for Loss Calculation
        labels = full_ids.clone()
        prompt_len = prompt_ids.shape[1]
        
        # Ensure we don't out-of-bounds if tokenization gets weird
        min_len = min(prompt_len, labels.shape[1])
        labels[0, :min_len] = -100 
        
        # Move to GPU (device map places model automatically, but inputs need manual mapping)
        full_ids = full_ids.to(model.device)
        labels = labels.to(model.device)
        
        # 5. Forward Pass
        with torch.no_grad():
            outputs = model(input_ids=full_ids, labels=labels)
            loss = outputs.loss.item() 
            
        losses.append(loss)
        
        # --- THE FIX: Delete the massive tensor here and flush ---
        del outputs
        torch.cuda.empty_cache()
        
        # Save individual result
        record = {
            "iteration": i + 1,
            "loss": loss,
            "perplexity": float(np.exp(loss)),
            "question_preview": question[:50] + "..."
        }
        f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        # 6. Save State Every 100 Points
        if (i + 1) % SAVE_INTERVAL == 0:
            current_mean_loss = np.mean(losses)
            current_mean_ppl = np.exp(current_mean_loss)
            
            # Print to console
            tqdm.write(f"--- Checkpoint {i + 1} --- Mean Loss: {current_mean_loss:.4f} | Mean PPL: {current_mean_ppl:.4f}")
            
            # Save summary to disk
            summary = {
                "total_evaluated": i + 1,
                "mean_loss": float(current_mean_loss),
                "mean_perplexity": float(current_mean_ppl)
            }
            with open(SUMMARY_FILE, "w", encoding="utf-8") as f_sum:
                json.dump(summary, f_sum, indent=4)
                
            # REMOVED the old 'del outputs' and 'empty_cache' from here!

# -------------------------
# FINAL REPORT
# -------------------------
final_mean_loss = np.mean(losses)
final_mean_ppl = np.exp(final_mean_loss)
print("\n=== Evaluation Complete ===")
print(f"Final Mean Loss: {final_mean_loss:.4f}")
print(f"Final Mean Perplexity: {final_mean_ppl:.4f}")
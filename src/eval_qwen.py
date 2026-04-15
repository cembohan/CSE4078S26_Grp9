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
BASE_MODEL = "Qwen/Qwen3-1.7B"       # HuggingFace path or local folder
LORA_CHECKPOINT = "./qwen-lora-checkpoints/checkpoint-3500" 
USE_LORA = False 

EVAL_SAMPLES = 2000                  # Stop after this many samples
SAVE_INTERVAL = 100                  # Calculate means and save every 100 iterations
OUTPUT_FILE = "doktorsitesi_eval_results.jsonl"
SUMMARY_FILE = "doktorsitesi_eval_summary.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# LOAD MODEL & TOKENIZER
# -------------------------
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",      # High-quality 4-bit format
    bnb_4bit_compute_dtype=torch.bfloat16, # Speeds up computation
    bnb_4bit_use_double_quant=True, # Saves even more VRAM
)
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, 
    trust_remote_code=True,
    quantization_config=quant_config,
    device_map="auto"
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
# Streaming=True means we don't download the entire dataset at once
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

        # Optional: Inject the specialty into the system prompt for better context
        # If the model knows it is acting as a "Kardiyoloji" (Cardiology) doctor, 
        # it will predict the medical terms much better.
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
        
        # 3. Tokenize
        prompt_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"]
        full_ids = tokenizer(full_text, return_tensors="pt")["input_ids"]
        
        # 4. Create Labels for Loss Calculation
        # We clone the full input IDs, but set the prompt tokens to -100.
        # PyTorch's CrossEntropyLoss ignores targets with a value of -100.
        labels = full_ids.clone()
        prompt_len = prompt_ids.shape[1]
        
        # Ensure we don't out-of-bounds if tokenization gets weird
        min_len = min(prompt_len, labels.shape[1])
        labels[0, :min_len] = -100 
        
        # Move to GPU
        full_ids = full_ids.to(DEVICE)
        labels = labels.to(DEVICE)
        
        # 5. Forward Pass
        with torch.no_grad():
            outputs = model(input_ids=full_ids, labels=labels)
            loss = outputs.loss.item() # This is the average PPL/Loss for the answer tokens
            
        losses.append(loss)
        
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
                
            # --- ADD THESE LINES TO STOP THE VRAM LEAK ---
            # Delete intermediate tensors that might be lingering
            del outputs
            del loss
            # Force PyTorch to release the fragmented memory cache back to the OS
            torch.cuda.empty_cache()

# -------------------------
# FINAL REPORT
# -------------------------
final_mean_loss = np.mean(losses)
final_mean_ppl = np.exp(final_mean_loss)
print("\n=== Evaluation Complete ===")
print(f"Final Mean Loss: {final_mean_loss:.4f}")
print(f"Final Mean Perplexity: {final_mean_ppl:.4f}")
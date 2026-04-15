import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# -------------------------
# CONFIGURATION
# -------------------------
BASE_MODEL = "Qwen/Qwen3-1.7B"       # HuggingFace path or local folder
LORA_CHECKPOINT = "./qwen-lora-checkpoints/checkpoint-3500"  # Your LoRA adapter -- change to the recent lora path
USE_LORA = False  # Set to False if you want to use only the base model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SYSTEM_PROMPT = ""#"You are an assistant that converts math problems into formal Lean statements. Whenever you see a math problem you must respond with a formal lean statement that represents the problem and nothing else. You turn questions into questions and do not solve the questions."

# -------------------------
# LOAD BASE MODEL + TOKENIZER
# -------------------------
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",      # High-quality 4-bit format
    bnb_4bit_compute_dtype=torch.bfloat16, # Speeds up computation
    bnb_4bit_use_double_quant=True, # Saves even more VRAM
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, trust_remote_code=True, quantization_config=quant_config)
base_model.to(DEVICE)
base_model.eval()

# -------------------------
# OPTIONAL: LOAD LoRA
# -------------------------
if USE_LORA:
    print(f"Loading LoRA adapter from {LORA_CHECKPOINT}...")
    model = PeftModel.from_pretrained(base_model, LORA_CHECKPOINT)
else:
    print("Skipping LoRA, using base model only...")
    model = base_model

model.eval()

# -------------------------
# CLI LOOP
# -------------------------
print("\n=== Chat CLI ===")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    # Combine system prompt + user input
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input}
    ]
    
    text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False
    )
    
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=True,
            top_p=0.9,
            temperature=0.7
        )

    response = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"Model: {response}\n")

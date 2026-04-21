import torch
import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# -------------------------
# CONFIGURATION
# -------------------------
MODEL_ID = "google/gemma-2-2b-it"
# Using your specific dataset from Hugging Face
DATASET_NAME = "alibayram/doktorsitesi" 
OUTPUT_DIR = "./gemma-2-2b-it-med-lora"
MAX_SEQ_LENGTH = 1024

# LoRA Parameters
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Training Hyperparameters
LEARNING_RATE = 2e-4
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
EPOCHS = 3

# -------------------------
# 1. LOAD DATASET & PREPROCESSING
# -------------------------
print(f"Loading dataset: {DATASET_NAME}...")
dataset = load_dataset(DATASET_NAME, split="train")

# Data Cleaning based on your dataset columns
# question_content -> user question
# question_answer -> doctor answer
dataset = dataset.filter(lambda x: x['question_answer'] is not None and x['question_content'] is not None)
dataset = dataset.filter(lambda x: len(x['question_answer']) > 20 and len(x['question_content']) > 10)

def format_instruction(sample):
    """
    Formats the sample into the Gemma-2 chat template.
    Matches the Turkish medical Q&A structure.
    """
    messages = [
        {"role": "user", "content": sample['question_content']},
        {"role": "assistant", "content": sample['question_answer']}
    ]
    # Use the tokenizer's template for consistency
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}

# -------------------------
# 2. LOAD TOKENIZER & MODEL (4-BIT)
# -------------------------
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Handle Gemma's system role limitation via custom template
tokenizer.chat_template = "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% set messages = messages[1:] %}{% else %}{% set system_message = '' %}{% endif %}{% for message in messages %}{% if loop.first and system_message != '' %}{{ '<start_of_turn>user\\n' + system_message + '\\n\\n' + message['content'] + '<end_of_turn>\\n' }}{% else %}{{ '<start_of_turn>' + message['role'] + '\\n' + message['content'] + '<end_of_turn>\\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<start_of_turn>model\\n' }}{% endif %}"

# Format the dataset
dataset = dataset.map(format_instruction)

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=quant_config,
    device_map="auto",
    trust_remote_code=True
)

# Prepare for PEFT
model = prepare_model_for_kbit_training(model)

# -------------------------
# 3. CONFIGURE LORA
# -------------------------
peft_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# -------------------------
# 4. TRAINING ARGUMENTS & SFT
# -------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    logging_steps=10,
    num_train_epochs=EPOCHS,
    save_steps=500,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_is_bf16_supported(),
    optim="paged_adamw_8bit",
    report_to="none"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    tokenizer=tokenizer,
    args=training_args,
    packing=False,
)

# -------------------------
# 5. EXECUTE TRAINING
# -------------------------
print("Starting training...")
trainer.train()

# Save final adapter
print(f"Saving fine-tuned adapter to {OUTPUT_DIR}...")
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Training Complete!")
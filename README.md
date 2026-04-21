# CSE4078S26_Grp9
NLP project on evaluating LLMs on Turkish benchmarks

dataset to use: alibayram/doktorsitesi
https://huggingface.co/datasets/alibayram/doktorsitesi

qwen3-1.7B: https://huggingface.co/Qwen/Qwen3-1.7B
gemma2-2b-it: https://huggingface.co/google/gemma-2-2b-it


# CSE4078 S2026 - Group 9 Progress Presentation

## Project Objective
- Evaluate Large Language Models on Turkish medical benchmarks
- Compare `google/gemma-2-2b-it` and `Qwen/Qwen3-1.7B`
- Use the `alibayram/doktorsitesi` dataset for clinical Q&A evaluation

## Dataset
- Dataset: `alibayram/doktorsitesi`
- Split: `train`
- Fields used:
  - `question_content`
  - `question_answer`
  - `doctor_specialty`
  - `doctor_title`

## Models and Setup
- `google/gemma-2-2b-it` with a custom Gemma chat template
- `Qwen/Qwen3-1.7B` with standard chat template
- 4-bit quantized inference using BitsAndBytes
- Optional LoRA adapter support available, currently disabled

## Evaluation Method
- Construct system prompt using doctor title and specialty
- Tokenize prompt + answer; mask prompt tokens during loss computation
- Compute per-sample loss and perplexity
- Stream dataset without downloading the whole split
- Save both detailed JSONL and summary JSON files

## Current Results
- Gemma evaluation summary:
  - `total_evaluated`: 2000
  - `mean_loss`: 4.7263
  - `mean_perplexity`: 112.8773
- Qwen evaluation summary:
  - `total_evaluated`: 2000
  - `mean_loss`: 4.8244
  - `mean_perplexity`: 124.5154

## Implementation Progress
- Data loading and streaming evaluation pipeline is implemented
- Prompt templates and answer masking logic are in place
- Incremental summary saving every 100 samples is implemented
- Results are recorded in `tests/` as JSON summary files



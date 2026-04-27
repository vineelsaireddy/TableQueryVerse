# TableQueryVerse: SemEval 2025 - Task 8
Question Answering over Tabular Data (Text-to-SQL)

## Overview
This project tackles the SemEval 2025 Task 8 benchmark by leveraging Large Language Models to answer natural language questions over tabular data. We implement two core approaches:
1. **Zero-Shot Baseline:** Using **Ollama** with `CodeLlama-7b` to execute zero-shot SQL generation and DataFrame operations.
2. **Fine-Tuned LLM:** Performing **QLoRA** (Quantized Low-Rank Adaptation) fine-tuning on **CodeLlama-7b-hf** using the **Spider dataset** to create a highly accurate Text-to-SQL specialist model.

## Features
- **Parameter-Efficient Fine-Tuning (PEFT):** Utilizes `peft` and `bitsandbytes` for 8-bit quantization and LoRA adapters, allowing CodeLlama to be fine-tuned efficiently on a single consumer GPU.
- **Local Inference with Ollama:** A seamless baseline implementation that communicates with a local Ollama server to keep inference private and fast.
- **Robust Evaluation Pipeline:** Built-in integration with `databench_eval`. Extracts generative python/SQL responses and executes them dynamically to verify outputs against DataBench and DataBench-lite.
- **WandB Tracking:** Automated Weights & Biases logging for the supervised fine-tuning loop (`spider-sql-finetuning` workspace).

## Repository Structure
- `installing_ollama.sh`: Script to automatically install Ollama locally.
- `ollama_codellama.py`: The zero-shot baseline using Local Ollama inference. Generates predictions on the DataBench benchmarks.
- `finetune.py`: The QLoRA training script. Prepares the Spider dataset, initializes 8-bit CodeLlama-7b, and trains the adapters.
- `load_model.py`: Utility script demonstrating how to load the base model and merge the trained LoRA adapters (`finetuned_model/`) for inference.
- `calculate_accuracy.py`: Script to compute final benchmark accuracy.
- `databench_eval/`: The official DataBench evaluation modules used to score outputs.
- `spider_data/`: Contains the `tables.json` schemas necessary for context-aware prompt generation during training.

## Running Instructions

### 1. Zero-Shot Baseline (Ollama)
First, ensure you have the `codellama:7b` model available on a running Ollama server.
```bash
# Install Ollama (Linux)
bash installing_ollama.sh

# Pull the model and start the server in a separate terminal
ollama run codellama:7b

# Run the databench baseline script
python ollama_codellama.py
```
This will generate `predictions.txt` and `predictions_lite.txt` containing the outputs of the evaluation benchmark, along with a `submission.zip`.

### 2. QLoRA Fine-Tuning
To run the fine-tuning process yourself, ensure you have a CUDA-capable GPU (VRAM >= 12GB recommended for 8-bit).
```bash
# Install dependencies
pip install torch transformers peft datasets accelerate bitsandbytes wandb

# Run the training script
python finetune.py
```
*Note: Make sure your `spider_data/tables.json` is correctly placed. The model will automatically save LoRA weights to `spider-code-llama/finetuned_model`.*

### 3. Running Evaluator
You can run the accuracy calculations directly using the provided utility:
```bash
python calculate_accuracy.py
```

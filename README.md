# ANLP-Project - SemEval 2025 - Task 8: Question Answering over Tabular Data

## File structure:
- `installing_ollama.sh:` Installs ollama.
- `ollama_codellama.py:` This is the implementation of the baseline scripts.
- `predictions.txt:` This is the file where we are saving the outputs from DataBench.
- `predictions_lite.txt:` This is the file where we are saving the outputs from DataBench_lite.
- `spider_data:` Contains the data from spider dataset
- `finetune.py:` Code to finetune and codellama-7b on spider dataset for the task text-to-sql with QLoRA
- `load_model.py:` Sample code on how to load the saved model
- `finetuned_model:` Saved Adapter weights of the QLoRA finetuned model
- `run.sh:` Bash script that was used to run the code on Ada

from datetime import datetime
import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq, LlamaForCausalLM
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
    PeftModel
)
import sys
from datasets import load_dataset   

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Load the Spider dataset
dataset = load_dataset("spider")
train_dataset = dataset["train"].select(range(3000))
eval_dataset = dataset["validation"].select(range(300))

print("Loading Model")
base_model = "codellama/CodeLlama-7b-hf"
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token = tokenizer.eos_token

# Load the tables.json file
with open("./spider_data/tables.json", "r") as f:
    tables_info = json.load(f)

# Create a dictionary that maps db_id to the corresponding database information
db_info = {db["db_id"]: db for db in tables_info}

def tokenize(prompt, response):
    # Tokenize the input prompt
    prompt_tokens = tokenizer(
        prompt,
        truncation=True,
        max_length=512,
        padding=False,
        return_tensors=None,
    )
    
    # Tokenize the response (SQL query)
    response_tokens = tokenizer(
        response,
        truncation=True,
        max_length=512,
        padding=False,
        return_tensors=None,
    )
    
    # Combine input_ids
    result = {
        "input_ids": prompt_tokens["input_ids"] + response_tokens["input_ids"],
        "attention_mask": prompt_tokens["attention_mask"] + response_tokens["attention_mask"],
    }
    
    # Create labels: -100 for prompt tokens (ignored in loss), actual ids for response tokens
    result["labels"] = [-100] * len(prompt_tokens["input_ids"]) + response_tokens["input_ids"]
    
    return result

def get_schema(db_id):
    return db_info[db_id]

def generate_and_tokenize_prompt(data_point):
    db_id = data_point.get('db_id')
    question = data_point.get('question', '')
    query = data_point.get('query', '')
    
    # Get the schema for the database ID
    schema = get_schema(db_id)

    # Create the prompt without including the response
    prompt = f"""You are a powerful text-to-SQL model. Your job is to answer questions about a database. You are given a question and context regarding one or more tables.

    You must output the SQL query that answers the question.

    ### Input:
    {question}

    ### Context:
    The original column names of the database are {[item[1] for item in schema["column_names_original"]]}. The column names in general natural language are {[item[1] for item in schema["column_names"]]}

    ### Response:"""
    
    return tokenize(prompt, query)
# Apply the processing functions to the dataset
tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt)

print("Preparing for training")
# Prepare the model for LoRA
model.train()
model = prepare_model_for_kbit_training(model)
config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)

# Optional: resume from a checkpoint
resume_from_checkpoint = ""  # specify path to checkpoint if available
if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
    print(f"Resuming from {resume_from_checkpoint}")
    adapters_weights = torch.load(resume_from_checkpoint)
    set_peft_model_state_dict(model, adapters_weights)

print("Initialising wandb")
# Setup wandb project
wandb_project = "spider-sql-finetuning"
if wandb_project:
    os.environ["WANDB_PROJECT"] = wandb_project


batch_size = 1
per_device_train_batch_size = 1
gradient_accumulation_steps = 16
output_dir = "spider-code-llama"

training_args = TrainingArguments(
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    num_train_epochs=4,  # Use epochs instead of steps
    learning_rate=3e-4,
    fp16=True,
    logging_steps=10,
    optim="adamw_torch",
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=100,
    save_steps=100,
    output_dir=output_dir,
    group_by_length=True,
    report_to="wandb",
    run_name=f"codellama-{datetime.now().strftime('%Y-%m-%d-%H-%M')}",
)
print(f"Current device: {next(model.parameters()).device}")
# Define Trainer
trainer = Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    args=training_args,
    data_collator=DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    ),
)

model.config.use_cache = False
old_state_dict = model.state_dict
model.state_dict = (lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())).__get__(
    model, type(model)
)
if torch.__version__ >= "2" and sys.platform != "win32":
    print("Compiling the model")
    model = torch.compile(model)

print("Starting training")
# Train the model
trainer.train()

# Save model and tokenizer

# Save the finetuned model
model_dir = os.path.join(output_dir, "finetuned_model")
# os.makedirs(model_dir, exist_ok=True)
# torch.save(model.state_dict(), os.path.join(model_dir, "model.pth"))
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)
print(f"Finetuned model saved to {model_dir}")

# Load the finetuned model
loaded_model = LlamaForCausalLM.from_pretrained(base_model)
loaded_tokenizer = AutoTokenizer.from_pretrained(model_dir)
loaded_model = PeftModel.from_pretrained(loaded_model, "./spider-code-llama/finetuned_model")

eval_prompt = """You are a powerful text-to-SQL model. Your job is to answer questions about a database. You are given a question and context regarding one or more tables.

You must output the SQL query that answers the question.
### Input:
Which Class has a Frequency MHz larger than 91.5, and a City of license of hyannis, nebraska?

### Context:
CREATE TABLE table_name_12 (class VARCHAR, frequency_mhz VARCHAR, city_of_license VARCHAR)

### Response:
"""
model_input = tokenizer(eval_prompt, return_tensors="pt")

loaded_model.eval()
with torch.no_grad():
    print(tokenizer.decode(loaded_model.generate(**model_input, max_new_tokens=100)[0], skip_special_tokens=True))

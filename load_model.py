import torch
from transformers import AutoTokenizer, LlamaForCausalLM
from peft import PeftModel
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

loaded_model = LlamaForCausalLM.from_pretrained("codellama/CodeLlama-7b-hf")
loaded_tokenizer = AutoTokenizer.from_pretrained("./spider-code-llama/finetuned_model")
loaded_model = PeftModel.from_pretrained(loaded_model, "./spider-code-llama/finetuned_model")

eval_prompt = """You are a powerful text-to-SQL model. Your job is to answer questions about a database. You are given a question and context regarding one or more tables.

You must output the SQL query that answers the question.
### Input:
Which Class has a Frequency MHz larger than 91.5, and a City of license of hyannis, nebraska?

### Context:
CREATE TABLE table_name_12 (class VARCHAR, frequency_mhz VARCHAR, city_of_license VARCHAR)

### Response:
"""
model_input = loaded_tokenizer(eval_prompt, return_tensors="pt")

loaded_model.eval()
with torch.no_grad():
    print(loaded_tokenizer.decode(loaded_model.generate(**model_input, max_new_tokens=100)[0], skip_special_tokens=True))

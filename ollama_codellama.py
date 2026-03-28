import pandas as pd
import requests
import json
import zipfile
import sys

from datasets import Dataset
from databench_eval import Runner, Evaluator, utils

def call_ollama_model(prompts):
    results = []
    for p in prompts:
        try:
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': 'codellama:7b',
                    'prompt': p,
                    'stream': False,
                    'options': {
                        'num_predict': 128,
                    }
                }
            )
            response.raise_for_status()
            result = response.json()['response']
            results.append(result)
        except Exception as e:
            results.append(f"__CODE_GEN_ERROR__: {str(e)}")
    
    return results

def example_generator(row: dict) -> str:
    """IMPORTANT:
    **Only the question and dataset keys will be available during the actual competition**.
    You can, however, try to predict the answer type or columns used
    with another modeling task if that helps, then use them here.
    """
    dataset = row["dataset"]
    question = row["question"]
    df = utils.load_table(dataset)
    return f"""
# TODO: complete the following function in one line. It should give the answer to: How many rows are there in this dataframe? 
def example(df: pd.DataFrame) -> int:
    df.columns=["A"]
    return df.shape[0]

# TODO: complete the following function in one line. It should give the answer to: {question}
def answer(df: pd.DataFrame) -> {row["type"]}:
    df.columns = {list(df.columns)}
    return"""

def example_postprocess(response: str, dataset: str, loader):
    try:
        print(response)
        df = loader(dataset)
        
        # First, try to find the last 'return' statement
        return_parts = response.split("return")
        if len(return_parts) < 2:
            return "__CODE_ERROR__: No return statement found in response"
        
        # Get the last return statement
        last_return = return_parts[-1].strip()
        
        # If there are multiple lines, get just the first one
        code_to_execute = last_return.split("\n")[0].strip()
        
        # Remove any potential "[end of text]" or similar tokens
        code_to_execute = code_to_execute.replace("[end of text]", "").strip()
        
        # Prepare the complete code to execute
        full_code = f"""
def answer(df):
    return {code_to_execute}

global ans
ans = answer(df)
"""
        
        # Execute the code
        exec(full_code)
        
        # Handle potential multi-line results
        return str(ans).split("\n")[0] if "\n" in str(ans) else ans
        
    except Exception as e:
        return f"__CODE_ERROR__: {str(e)}"

N = 10
qa = utils.load_qa(name="semeval", split="dev")
subset_qa = qa[:N]  # Replace N with the number of examples you want to use
qa = Dataset.from_pandas(pd.DataFrame(subset_qa))

# sys.exit()
evaluator = Evaluator(qa=qa)

runner = Runner(
    model_call=call_ollama_model,
    prompt_generator=example_generator,
    postprocess=lambda response, dataset: example_postprocess(
        response, dataset, utils.load_table
    ),
    qa=qa,
    batch_size=2,
)

runner_lite = Runner(
    model_call=call_ollama_model,
    prompt_generator=example_generator,
    postprocess=lambda response, dataset: example_postprocess(
        response, dataset, utils.load_sample
    ),
    qa=qa,
    batch_size=2,
)

responses = runner.run(save="predictions.txt")
responses_lite = runner_lite.run(save="predictions_lite.txt")
print(f"DataBench accuracy is {evaluator.eval(responses)}")
print(f"DataBench_lite accuracy is {evaluator.eval(responses_lite, lite=True)}")

with zipfile.ZipFile("submission.zip", "w") as zipf:
    zipf.write("predictions.txt")
    zipf.write("predictions_lite.txt")

print("Created submission.zip containing predictions.txt and predictions_lite.txt")
import pandas as pd
import requests
import json
import zipfile
import sys

from datasets import Dataset
from databench_eval import Runner, Evaluator, utils

N = 320
qa = utils.load_qa(name="semeval", split="dev")
subset_qa = qa[:N]  # Replace N with the number of examples you want to use
qa = Dataset.from_pandas(pd.DataFrame(subset_qa))

# sys.exit()
evaluator = Evaluator(qa=qa)

if __name__ == "__main__":
    with open("predictions_2.txt", 'r') as f:
        responses = f.read().splitlines()
        print(responses)
        print(f"DataBench accuracy is {evaluator.eval(responses)}")
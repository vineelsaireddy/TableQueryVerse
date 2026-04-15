from typing import Callable, List, Union, Optional
from .utils import load_qa
from datasets import Dataset
from tqdm import tqdm
import ast


class Evaluator:
    def __init__(
        self,
        compare: Optional[Callable] = None,
        qa: Optional[Dataset] = None,
        batch_size: int = 10,
        **kwargs,
    ):
        self.qa = qa if qa is not None else load_qa(**kwargs)

    def compute_accuracy(self, value, truth, semantic=None):
        return str(value).strip() == str(truth).strip()
    
    def is_exact_match(self, value, truth, lite: str=None):
        return str(value) == str(truth)

    def eval(
        self,
        responses: Union[List[str], str],
        lite: bool = False,
    ) -> float:
        if isinstance(responses, str):
            with open(responses, "r") as f:
                responses = f.read().splitlines()

        correct = 0
        exact_match = 0

        truths = self.qa["answer"] if not lite else self.qa["sample_answer"]

        with open(f"comparision_lite_{lite}.txt", "w") as f:
            for response, truth in tqdm(zip(responses, truths), total=len(truths)):
                temp_list = ast.literal_eval(response)
                print(temp_list)

                if self.compute_accuracy(response, truth, "lite" if lite else None):
                    correct += 1
                    f.write(f"Correct\tTruth:{truth}\tPredicted:{response}\n")
                else:
                    f.write(f"Incorrect\tTruth:{truth}\tPredicted:{response}\n")

                if self.is_exact_match(response, truth, "lite" if lite else None):
                    exact_match += 1
        
        accuracy = correct / len(truths)
        exact_match_accuracy = exact_match / len(truths)


        return accuracy, exact_match_accuracy

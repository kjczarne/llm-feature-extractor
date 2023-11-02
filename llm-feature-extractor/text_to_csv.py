from transformers import pipeline, AutoModelForCausalLM
from dataclasses import dataclass
from pathlib import Path
from typing import List
import pandas as pd
import torch
import argparse
from .instruct_pipeline import InstructionTextGenerationPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer



@dataclass
class Config:
    model_name: str
    text_file_path: Path
    features: List[str]
    save_model: bool = False


def construct_prompt(text: str, features: List[str]) -> str:
    return f"Here is a list of features to extract from a text snippet: [{','.join(features)}].\n" \
           f"Here is the text snippet: {text}\n\n" \
           "Please construct a CSV table row of values for each feature extracted from the text." \
           "Please keep the header row and include feature values in a new line." \
           "based on the list of features and the text provided."


def text_to_csv(config: Config) -> str:
    tokenizer = AutoTokenizer.from_pretrained(config.model_name,
                                              padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(config.model_name,
                                                 device_map="auto",
                                                 torch_dtype=torch.bfloat16)

    instruct_pipeline = InstructionTextGenerationPipeline(model=model,
                                                          tokenizer=tokenizer)

    responses = []

    if config.save_model:
        instruct_pipeline.save_pretrained("model")

    with open(config.text_file_path, "r") as f:
        text = f.read()

    responses.append(instruct_pipeline(construct_prompt(text, config.features)))
    return responses


def main():
    parser = argparse.ArgumentParser("csv-to-text")
    parser.add_argument("-m", "--model-name",
                        type=str,
                        help="Name of the model to use",
                        # default="mistralai/Mistral-7B-Instruct-v0.1")
                        # default="databricks/dolly-v2-12b")
                        # default="databricks/dolly-v2-3b")
                        default="./model")
    parser.add_argument("-t", "--text-file-path",
                        default="text.txt",
                        help="Path to the text file from which we want to extract features")
    parser.add_argument("-f", "--features",
                        nargs="+",
                        help="Features to be extracted from the test sample")
    parser.add_argument("--save-model",
                        action="store_true",
                        default=False,
                        help="Whether to save the downloaded model or not")
    args = parser.parse_args()

    features = [] if args.features is None else args.features
    text = text_to_csv(Config(args.model_name,
                              Path(args.text_file_path),
                              features,
                              args.save_model))
    print(text)


if __name__ == "__main__":
    main()
 
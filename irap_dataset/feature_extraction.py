from transformers import pipeline, AutoModelForCausalLM
from dataclasses import dataclass
from pathlib import Path
from typing import List
import pandas as pd
import torch
import argparse
from llm_feature_extractor.instruct_pipeline import InstructionTextGenerationPipeline
from llm_feature_extractor.text_to_csv import Config, text_to_csv, construct_prompt
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser("csv-to-text")
    parser.add_argument("-m", "--model-name",
                        type=str,
                        help="Name of the model to use",
                        default="mistralai/Mistral-7B-Instruct-v0.1")
                        # default="databricks/dolly-v2-12b")
                        # default="databricks/dolly-v2-3b")
                        # default="./model")
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
 
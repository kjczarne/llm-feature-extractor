from transformers import pipeline
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import torch
import argparse


@dataclass
class Config:
    model_name: str
    csv_file_path: Path
    batching: bool
    batch_size: int

    @property
    def df(self):
        return pd.read_csv(self.csv_file_path, header=0)

    @property
    def table_csv(self):
        return self.df.to_csv(index=False)

    def table_csv_slice(self, start: int, end: int):
        return self.df[start:end].to_csv(index=False)


def construct_prompt(table_csv: str) -> str:
    return "Given the following table in CSV format, " \
            f"generate a short description of the company: \n\n {table_csv}"


def csv_to_text(config: Config) -> str:
    instruct_pipeline = pipeline(model=config.model_name,
                                 torch_dtype=torch.bfloat16,
                                 trust_remote_code=True,
                                 device_map="auto")
    
    responses = []
     
    match config.batching:
        case True:
            for i in range(0, len(config.df), config.batch_size):
                table_csv = config.table_csv_slice(i, i + config.batch_size)
                responses.append(instruct_pipeline(construct_prompt(table_csv)))
        case False:
            table_csv = config.table_csv
            responses.append(instruct_pipeline(construct_prompt(table_csv)))

    return "\n\n".join(responses) if config.batching else responses[0]


def main():
    parser = argparse.ArgumentParser("csv-to-text")
    parser.add_argument("-m", "--model-name",
                        type=str,
                        help="Name of the model to use",
                        default="databricks/dolly-v2-12b")
    parser.add_argument("-d", "--data-file",
                        type=str,
                        help="Path to the CSV with the data",
                        default="data.csv")
    parser.add_argument("-b", "--batching", action="store_true",
                        help="Whether to process the data in batches (recommended)")
    parser.add_argument("--batch-size",
                        type=int,
                        help="How many samples from the dataset are we taking",
                        default=10)
    args = parser.parse_args()

    text = csv_to_text(Config(args.model_name,
                              Path(args.data_file),
                              args.batching,
                              args.batch_size))
    print(text)


if __name__ == "__main__":
    main()
 
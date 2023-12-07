import argparse
import pandas as pd
from typing import List
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    data_file: str | Path
    num_of_sentences: int
    num_of_samples: int
    start_with: int
    _drop_columns: List[str] | None

    def __post_init__(self):
        self.df = pd.read_csv(self.data_file, header=0)
        self.drop_columns = self._drop_columns or []

    @property
    def table(self):
        return self.df

    @property
    def table_csv(self):
        return self.df[self.start_with:self.start_with+self.num_of_samples].drop(self.drop_columns, axis=1).to_csv(index=False)


def construct_prompt(config: Config) -> str:
    prompt = f"""For each line in the following CSV file generate a short description of the company.
    Attempt to generate at least {config.num_of_sentences} sentences.
    \n\n{config.table_csv}
    """
    return prompt


def main():

    parser = argparse.ArgumentParser("construct-prompt")
    parser.add_argument("-d", "--data-file",
                        type=str,
                        help="Path to the CSV with the data",
                        default="data.csv")
    parser.add_argument("-s", "--start-with",
                        type=int,
                        help="The first sample index to start with. Convenient for processing the dataset in batches",
                        default=0)
    parser.add_argument("--num-samples",
                        type=int,
                        help="How many samples from the dataset are we taking",
                        default=10)
    parser.add_argument("--num-sentences",
                        type=int,
                        help="How many sentences are we generating per sample",
                        default=3)
    parser.add_argument("--drop",
                        nargs="+",
                        type=str,
                        help="Features to drop from the table",
                        default=None)

    args = parser.parse_args()
    
    config = Config(args.data_file,
                    args.num_sentences,
                    args.num_samples,
                    args.start_with,
                    args.drop)

    prompt = construct_prompt(config)

    print(prompt)


if __name__ == "__main__":
    main()

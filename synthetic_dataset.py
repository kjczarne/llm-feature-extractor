"""This script can be used to construct a synthetic dataset.
The data frame can be fed into BingAI or chatGPT. Running this
script directly will print a prompt that can be simply pasted
into either of those conversational models in order to produce
text snippets. The text snippets will then be used by the
LLM Feature Extractor to reconstruct the table.
"""

from typing import List, Any
from dataclasses import dataclass

from numpy.typing import NDArray

import numpy as np
import pandas as pd
import argparse


def _literal_to_type(literal: str):
    pass


def _literal_to_distribution(literal: str):
    pass


@dataclass
class RandomDataGenerator:
    seed: int

    def __post_init__(self):
        np.random.seed(self.seed)

    def _rand_select(self,
                     size: int,
                     values: List[Any],
                     distribution: List[float] | NDArray[np.float32]):
        return np.random.choice(values, size=size, p=distribution)


@dataclass
class Config:
    feature_names: List[str]
    feature_types: List[str]
    target_name: str
    num_of_sentences: int
    seed: int = 42

    def __post_init__(self):
        self.rdg = RandomDataGenerator(self.seed)

    @property
    def table(self):
        return pd.DataFrame()


def construct_prompt(config: Config) -> str:
    prompt = f"""I have a CSV file that I want to describe in natural language
    using all the features collected in the table. Treat the first row as the
    header. I need {config.num_of_sentences} per sample. Here is the CSV file:
    \n\n{config.table}
    """
    return prompt


def main():

    default_feature_names = ["serial_entrepreneur",
                             "comm_strategy_rating",
                             "revenue_2021",
                             "revenue_2022",
                             "subsidy_2021",
                             "industry",
                             "incorporation_year",
                             "customer_base_growth_from_last_year"]
    default_target_name = "revenue_2023"
    default_num_of_sentences = 3

    parser = argparse.ArgumentParser('llm-fe-dataset-synthesizer',
                                     description="Prepares a conversational model prompt from a table of features")

    parser.add_argument("-f", "--feature-names",
                        nargs="+",
                        type=str,
                        help="Names of features in the data frame",
                        default=default_feature_names)
    parser.add_argument("-p", "--feature-types",
                        nargs="+",
                        type=str,
                        help="Types of features in the data frame, must be Python primitives",
                        default=default_feature_names)
    parser.add_argument("-t", "--target-name",
                        type=str,
                        help="Dependent variable name",
                        default=default_target_name)
    parser.add_argument("-n", "--num-of-sentences",
                        type=int,
                        help="Number of sentences to use",
                        default=default_num_of_sentences)

    args = parser.parse_args()

    config = Config(args.feature_names,
                    args.feature_types,
                    args.target_name,
                    args.num_of_sentences)
    print(config)


if __name__ == "__main__":
    main()


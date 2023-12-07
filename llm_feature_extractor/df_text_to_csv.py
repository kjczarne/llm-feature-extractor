from llm_feature_extractor.text_to_csv import text_to_csv, Config, construct_prompt
import pandas as pd
from typing import List
from rich.pretty import pprint


def feature_list_from_file(file_path: str) -> List[str]:
    with open(file_path, "r") as f:
        features = f.read().splitlines()
    return features


def _explainability_prompt(text: str, features: List[str]) -> str:
    return "Here is a list of features to extract from a text snippet: " \
           f"[{','.join(features)}].\n" \
           f"Here is the text snippet: {text}\n\n" \
           "Please list the features extracted from the text in a new line " \
           "and provide an explanation wherever possible why you decided that said " \
           "feature was present in the dataset"


def _explainability_prompt_entire_row(row: pd.Series, features: List[str]) -> str:
    return "Here is a list of criteria for evaluation of the viability of a business: " \
           f"[{','.join(features)}].\n" \
           f"Here is a row from a table describing the company. Here's the header {row.index}" \
           f"and here is the list of values corresponding to said header: {list(row)}\n\n" \
           "Please indicate which features can be inferred to be present for this company"



def df_text_to_csv(df: pd.DataFrame,
                   desc_series_name: str,
                   config: Config,
                   limit: int = 5) -> List[str]:
    descriptions = df[desc_series_name]
    responses = []
    seen = 0
    for description in descriptions:
        config.text = description
        # response = text_to_csv(config)
        response = text_to_csv(config, _explainability_prompt)
        response = response[0][0]["generated_text"]
        responses.append(response)
        seen += 1
        if seen >= limit:
            break
    return responses


def df_row_explain(df: pd.DataFrame,
                   config: Config,
                   limit: int = 5) -> List[str]:
    responses = []
    seen = 0
    for _, row in df.iterrows():
        config.text = row
        response = text_to_csv(config, _explainability_prompt_entire_row)
        response = response[0][0]["generated_text"]
        responses.append(response)
        seen += 1
        if seen >= limit:
            break
    return responses


def main():
    df = pd.read_csv("2023_24_grants_and_contributions.csv", header=0, encoding="ISO-8859-1")
    config = Config(model_name="./model",
                    text="",
                    features=feature_list_from_file("english_terms_short.txt"))
    # responses = df_text_to_csv(df,
    #                            "Description (English)",
    #                            config,
    #                            limit=2)
    responses = df_row_explain(df,
                               config,
                               limit=2)
    pprint(responses)


if __name__ == "__main__":
    main()

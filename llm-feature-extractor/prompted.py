from typing import List
from .types import Model, Property, Prompt


class PromptableModel(Model):
    pass

    def construct_priming_prompt(self, properties: List[Property]) -> Prompt:
        return f"""Consider the following features comprising a tabular dataset: {properties}.
                You will be given a number of text inputs from which you need to extract
                said properties and present them in a CSV format. Keep the response minimal,
                produce only the tabular response and do not provide any additional explanation."""

    def construct_sample_prompt(self, text: str) -> Prompt:
        return f"""Given the aforementioned features, extract them from the following text: {text}"""

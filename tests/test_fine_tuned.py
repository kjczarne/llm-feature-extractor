from typing import Tuple, Iterable
import pytest
from transformers import AutoModelForSeq2SeqLM, AutoConfig, AutoTokenizer, Trainer, TrainingArguments
from tokenizers import Tokenizer
from datasets import Dataset
from llm_feature_extractor.types import ModelId, Model
from llm_feature_extractor.utils import accuracy_metric_trainer_api


model_ids = [
    "bigscience/bloom"
]

@pytest.fixture(scope="module", params=model_ids)
def tokenizer_and_model(model_id: ModelId) -> Tuple[Tokenizer, Model]:
    model_ = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    tokenizer = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    return tokenizer, model_


def tokenize(tokenizer: Tokenizer, column_name: str) -> Iterable[int]:
    """Default tokenize function for the `.map()` interface in Huggingface Transformers.

    Args:
        tokenizer (Tokenizer): a tokenizer instance to be used

    Returns:
        Callable[[Dataset], List[int]]: the concrete tokenization function
    """
    def tokenize(dataset: Dataset):
        return tokenizer(dataset[column_name], padding=True)
    return tokenize


def test_fine_tuned_imdb(tokenizer_and_model):
    tokenizer, model = tokenizer_and_model

    output_dir = "./training_results"
    sequence_column_name = "text"
    batch_tokenize = True

    # TODO: take a very small subset of IMDB dataset to fine-tune on
    # TODO: prepare a CSV sequence of properties as labels for the Seq2Seq task
    train_dataset
    val_dataset

    tokenized_train_dataset = train_dataset.map(tokenize(tokenizer, sequence_column_name), batched=batch_tokenize)
    tokenized_val_dataset = val_dataset.map(tokenize(tokenizer, sequence_column_name), batched=batch_tokenize)
    training_args = TrainingArguments(output_dir)

    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        tokenizer=tokenizer,
        compute_metrics=accuracy_metric_trainer_api
    )

    print(f"Fine-tuning")
    trainer.train()

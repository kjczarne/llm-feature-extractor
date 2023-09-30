from transformers import (DistilBertTokenizer,
                          DistilBertForSequenceClassification,
                          DistilBertModel,
                          TrainingArguments,
                          Trainer,
                          EvalPrediction)
from datasets import load_dataset, Dataset
from pathlib import Path
from typing import Dict
import evaluate
import numpy as np
import pandas as pd


def accuracy_metric_trainer_api(eval_pred: EvalPrediction) -> Dict[str, int | float]:
    """Calculates accuracy metric using the `evaluate` package. Compatible
    with Huggingface Trainer API.

    Args:
        eval_pred (EvalPrediction): a prediction instance from the Huggingface model

    Returns:
        Dict[str, int | float]: calculated accuracy
    """
    accuracy = evaluate.load("accuracy")
    logits, labels = eval_pred
    pred_class = np.argmax(logits, axis=-1)  # take the max-scoring logit as the predicted class ID
    return accuracy.compute(predictions=pred_class,
                            references=labels)


EPOCHS = 30
num_labels = 2
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased",
                                                            num_labels=num_labels)

model_save_dir = Path("results/model")
training_args = TrainingArguments(model_save_dir,
                                  evaluation_strategy="epoch",
                                  num_train_epochs=EPOCHS)

# Load the ground-truth labels
gt_labels = pd.read_csv("train_clf.csv", header=0)["high_revenue"].astype(int).tolist()

# Load text
with open("train_clf.txt", "r") as f:
    texts = f.read().split("\n\n")

df = pd.DataFrame({"text": texts, "labels": gt_labels})

dataset = Dataset.from_pandas(df)


# Perform a train-test split
dataset = dataset.train_test_split(test_size=0.2)

# Define the tokenization strategy

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

train_dataset = dataset["train"].map(tokenize, batched=True, batch_size=len(dataset["train"]))
val_dataset = dataset["test"].map(tokenize, batched=True, batch_size=len(dataset["test"]))


trainer = Trainer(
    model,
    training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=accuracy_metric_trainer_api
)

trainer.train()
trainer.save_model(model_save_dir)

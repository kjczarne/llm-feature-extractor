from transformers import EvalPrediction
from typing import Dict
import evaluate
import numpy as np


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

import lime
import transformers
import datasets
import torch
import numpy as np
import scipy as sp
from pathlib import Path
import pandas as pd
from datasets import Dataset, load_dataset, load_from_disk
from lime.lime_text import LimeTextExplainer

root = Path(__file__).parent.parent.parent

# load a BERT sentiment analysis model
tokenizer = transformers.DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
model = transformers.DistilBertForSequenceClassification.from_pretrained(
    "results/model"
).cuda()

# define a prediction function
def f(x):
    tv = torch.tensor([tokenizer.encode(v, padding='max_length', max_length=500, truncation=True) for v in x]).cuda()
    outputs = model(tv)[0].detach().cpu().numpy()
    scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
    val = sp.special.logit(scores[:,1]) # use one vs rest logit units
    return val

dataset = load_from_disk(root / "results/dataset")

# build an explainer using a token masker
explainer = LimeTextExplainer(class_names=["high", "low"])

# lime explanations work a little bit differently than shap
# you need to do per-samples explanations

x_idx = 2
y_pred = f(dataset['test'][x_idx]['text'])
exp = explainer.explain_instance(dataset['test'][x_idx]['text'], y_pred, num_features=10)

print(exp.as_list())

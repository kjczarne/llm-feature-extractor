import shap
import transformers
import datasets
import torch
import numpy as np
import scipy as sp
from pathlib import Path
import pandas as pd
from datasets import Dataset, load_dataset, load_from_disk

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
explainer = shap.Explainer(f, tokenizer)

shap_values = explainer(dataset['test'][:10], fixed_context=1, batch_size=2)

# plot a sentence's explanation
plot = shap.plots.text(shap_values[2], display=False)

with open("results/shap_plot.html", "w") as f:
    f.write(plot)

# Initial imports
import numpy as np

import torch
import transformers
from datasets import load_from_disk

from captum.attr import IntegratedGradients
from captum.attr import InputXGradient
from captum.attr import LayerConductance
from captum.attr import NeuronConductance

import matplotlib
import matplotlib.pyplot as plt

from scipy import stats
import pandas as pd
import scipy as sp
from pathlib import Path

root = Path(__file__).parent.parent.parent

# load a BERT sentiment analysis model
tokenizer = transformers.DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
model = transformers.DistilBertForSequenceClassification.from_pretrained(
    "results/model"
).cuda()

# define a prediction function
def f(tv):
    outputs = model(tv)[0].detach().cpu().numpy()
    scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
    val = sp.special.logit(scores[:,1]) # use one vs rest logit units
    return val

explainer = InputXGradient(f)

dataset = load_from_disk(root / "results/dataset")

x_idx = 2
x = dataset['test'][x_idx]['text']
tv = torch.tensor([tokenizer.encode(v, padding='max_length', max_length=500, truncation=True) for v in x]).cuda()
y = dataset['test'][x_idx]['labels']
attr = explainer.attribute(tv, target=y)

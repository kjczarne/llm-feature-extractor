import shap
import transformers
import datasets
import torch
import matplotlib
import numpy as np
import scipy as sp
from pathlib import Path
import pandas as pd
from datasets import Dataset, load_dataset, load_from_disk
from captum.attr import LayerIntegratedGradients, TokenReferenceBase, visualization
from functools import partial

root = Path(__file__).parent.parent.parent

# load a BERT sentiment analysis model
tokenizer = transformers.DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
model = transformers.DistilBertForSequenceClassification.from_pretrained(
    "results/model"
).cuda()
model.eval()

# define a prediction function
def f(x):
    tv = torch.tensor([tokenizer.encode(v, padding='max_length', max_length=500, truncation=True) for v in x]).cuda()
    outputs = model(tv)[0].detach().cpu().numpy()
    scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
    val = sp.special.logit(scores[:,1]) # use one vs rest logit units
    return val

lig = LayerIntegratedGradients(model, model.embedding)

single_sample_plotting_functions = [
    ("text", partial(shap.plots.text, display=False)),
]

def save_plot(plot: matplotlib.figure.Figure | str, plot_name: str):
    name = plot_name.replace(" ", "_").lower()
    match plot:
        case str(plot):
            with open(f"results/shap_plot_{name}.html", "w") as f:
                f.write(plot)
        case fig if isinstance(fig, matplotlib.figure.Figure):
            plot.savefig(f"results/shap_plot_{name}.png")
        case _:
            raise ValueError(f"Unknown plot type {type(plot)}")


SAMPLE_IDX = 2
for name, f in single_sample_plotting_functions:
    plot = f(some_input_sample_arg=shap_values[SAMPLE_IDX])

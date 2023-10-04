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
from functools import partial

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


single_sample_plotting_functions = [
    ("text", partial(shap.plots.text, display=False)),
    ("waterfall", partial(shap.plots.waterfall, show=False)),
]

multiple_instance_plotting_functions = [
    ("violin", partial(shap.plots.violin, show=False)),  # Needs an array of instances, not a single instance
    ("decision", partial(shap.plots.decision, base_value=explainer.expected_value)),  # try explainer.shap_values()
]

plotting_functions_requiring_explainer_obj = [
    # ("beeswarm", partial(shap.plots.beeswarm, show=False)),  # Needs an array of instances, not a single instance
    # ("heatmap", partial(shap.plots.heatmap, show=False)),  # Needs 2D array
    ("scatter", partial(shap.plots.scatter, show=False)),  # `too many indices for array`
]

plotting_functions_requiring_explanation_obj = [
    ("force", partial(shap.plots.force, base_value=explainer.expected_value)),  # can only display Explanation objects or arrays of them
]

plotting_functions = [
    # ("group diff", partial(shap.plots.group_difference, show=False)),  # missing group mask
    # ("embedding", partial(shap.plots.embedding, show=False)),  # missing the index of the feature to use to color the embedding
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
# for name, f in single_sample_plotting_functions:
#     plot = f(shap_values=shap_values[SAMPLE_IDX])

shap_value_vector_lengths = [j.shape[0] for j in shap_values]
shap_value_matrix = np.array([
    np.pad(shap_values.values[i], (0, np.max(shap_value_vector_lengths) - shap_values[i].shape[0]), "constant", constant_values=0)
    for i in range(len(shap_values.values))
])

# `np.pad` acts on both edges of a given axis, we only want to pad the ending of the array with zeros

for name, f in multiple_instance_plotting_functions:
    plot = f(shap_values=shap_value_matrix)

for name, f in plotting_functions_requiring_explainer_obj:
    shap_values.values = shap_value_matrix
    plot = f(shap_values=shap_values)

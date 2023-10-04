import shap
import transformers
import datasets
import torch
import numpy as np
import scipy as sp
from pathlib import Path
import pandas as pd
from datasets import Dataset, load_dataset, load_from_disk
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

root = Path(__file__).parent.parent.parent

# load a BERT sentiment analysis model
tokenizer = transformers.DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
# NOTE: we load the `DistilBertModel` instead of the `DistilBertForSequenceClassification`
model = transformers.DistilBertModel.from_pretrained(
    "results/model"
).cuda()

# define a prediction function
def f(x):
    tv = torch.tensor([tokenizer.encode(v, padding='max_length', max_length=500, truncation=True) for v in x]).cuda()
    outputs = model(tv)[0].detach().cpu().numpy()
    return outputs[0]
    
dataset = load_from_disk(root / "results/dataset")
embeddings = np.array([f([x]) for x in dataset['test']['text']])
labels = np.array([x for x in dataset['test']['labels']])

print(embeddings.shape)

# Assume `embeddings` is a 2D numpy array or list of BERT embeddings
sample_idx = 2
tsne_model = TSNE(perplexity=3, n_components=3, init='pca', n_iter=2500, random_state=23)
tsne_squashed_values = [
    tsne_model.fit_transform(embeddings[i])
    for i in range(embeddings.shape[0])
]
tsne_values_label_high = np.array([
    i[0] for i in
    filter(lambda x: x[1] == 1, zip(tsne_squashed_values, labels))
])
tsne_values_label_low = np.array([
    i[0] for i in
    filter(lambda x: x[1] == 0, zip(tsne_squashed_values, labels))
])


# Assuming `tsne_squashed_values` contains your TSNE-transformed embeddings
x_high = tsne_values_label_high[:, 0]
y_high = tsne_values_label_high[:, 1]
z_high = tsne_values_label_high[:, 2]
x_low = tsne_values_label_low[:, 0]
y_low = tsne_values_label_low[:, 1]
z_low = tsne_values_label_low[:, 2]

# Create a 3D scatterplot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatterplot points
ax.scatter(x_high, y_high, z_high, c='r', marker='o')
ax.scatter(x_low, y_low, z_low, c='g', marker='o')

# Set labels for the axes
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# Show the plot
plt.savefig("results/tsne_plot.png")

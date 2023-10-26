# %%
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path

root = Path(__file__).parent.parent.parent

# Load the dataset
df = pd.read_csv(root / "train_clf.csv", header=0)

# Split the dataset into training and testing sets
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Define the features and target variable
features = [f"P{i}" for i in range(1, 38)]
target = "high_revenue"

# Convert the dataset to DMatrix format
dtrain = xgb.DMatrix(train_data[features], label=train_data[target])
dtest = xgb.DMatrix(test_data[features], label=test_data[target])


# Define the XGBoost parameters
params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 6,
    "eta": 0.3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": 42,
}

# Train the XGBoost model
num_rounds = 100
model = xgb.train(params, dtrain, num_rounds)

# Explain
explainer = shap.TreeExplainer(model)

SAMPLE_IDX = 2
shap_values = explainer(test_data[features])

# plot = shap.plots.waterfall(shap_values[SAMPLE_IDX], show=False)
# plot.savefig(f"results/shap_plot_xgb_waterfall.png")

# %%
shap.plots.initjs()
shap.plots.force(explainer.expected_value, shap_values.values)
# %%

plot = shap.plots.force(explainer.expected_value,
                        shap_values.values[SAMPLE_IDX],
                        matplotlib=True,
                        show=False)
plot.savefig(root / f"results/shap_plot_xgb_force.png")
# %%

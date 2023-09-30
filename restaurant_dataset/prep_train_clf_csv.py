# %% Prepare the training data for classification
# This is specifically meant for the restaurant dataset
# %%
import pandas as pd
from pathlib import Path

# %%

root = Path(__file__).parent.parent

# %%

df = pd.read_csv(root / "train.csv", header=0)

# %%

df.hist(column="revenue", bins=100)

# %%

df.head()

# %%

df.columns


# %%

df["revenue"].describe()

# %%

df["revenue"].quantile(0.6)

# %%
# Take the 60% of the data with the highest revenue and classify it as "high revenue"
df.loc[df["revenue"] > df["revenue"].quantile(0.6), "high_revenue"] = True

# %%
# See if it worked
df.head()
# %%
# Take all missing values in the "high_revenue" column and classify them as "low revenue"
df.loc[df["high_revenue"].isna(), "high_revenue"] = False

# %%

# See if it worked
df.head()

# %%
# Export the dataframe to a CSV file

df.to_csv("train_clf.csv", index=False)

# %%

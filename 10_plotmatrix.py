#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
#%%
df = pd.read_csv("./data/Hamilton.Data.txt", sep="\t")
df.columns = df.columns.str.strip()

print(df.head())
print(df.columns)
#%%
sns.pairplot(
    df,
    diag_kind="hist",
    plot_kws={
        "s": 40,
        "alpha": 0.7,
        "edgecolor": "black",
    }
)

plt.tight_layout()
plt.savefig(
    "./fig/Hamilton_scatter_matrix.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
#%%
model_1 = smf.ols(
    "Y ~ X1",
    data=df
).fit()

model_2 = smf.ols(
    "Y ~ X2",
    data=df
).fit()

model_12 = smf.ols(
    "Y ~ X1 + X2",
    data=df
).fit()

print(model_1.summary())
print(model_2.summary())
print(model_12.summary())
#%%
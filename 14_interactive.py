#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
#%%
df = pd.read_csv("./data/Salary.Survey.txt", sep="\t")
df.columns = df.columns.str.strip()
print(df.head())
print(df.columns)
print(df.dtypes)

print("E의 범주:", sorted(df["E"].unique()))
print("M의 범주:", sorted(df["M"].unique()))
#%%
#%%
styles = {
    (1, 0): {"color": "tab:blue",   "marker": "o"},
    (1, 1): {"color": "tab:orange", "marker": "o"},
    (2, 0): {"color": "tab:green",  "marker": "s"},
    (2, 1): {"color": "tab:red",    "marker": "s"},
    (3, 0): {"color": "tab:purple", "marker": "^"},
    (3, 1): {"color": "tab:brown",  "marker": "^"},
}

plt.figure(figsize=(8, 5))

for (e, m), style in styles.items():
    group = df[(df["E"] == e) & (df["M"] == m)]

    plt.scatter(
        group["X"],
        group["S"],
        color=style["color"],
        marker=style["marker"],
        s=65,
        edgecolor="black",
        alpha=0.8,
        label=f"E={e}, M={m}"
    )

plt.xlabel("Experience (X)", fontsize=12)
plt.ylabel("Salary (S)", fontsize=12)
plt.grid(alpha=0.3)

plt.legend(
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    fontsize=15
)

plt.tight_layout()

plt.savefig(
    "./fig/Salary_Survey_scatter.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
#%%
#%%
interaction_model = smf.ols(
    formula=(
        "S ~ X + C(E, Treatment(reference=3)) * M"
    ),
    data=df
).fit()

print(interaction_model.summary())
#%%
#%%
new_data = pd.DataFrame({
    "X": [0, 0, 0, 0, 0, 0],
    "E": [1, 1, 2, 2, 3, 3],
    "M": [0, 1, 0, 1, 0, 1]
})

prediction = interaction_model.predict(new_data)

print(prediction)
#%%
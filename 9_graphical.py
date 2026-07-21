#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%%
df = pd.read_csv("./data/Anscombe.txt", sep="\t")
print(df.head())
#%%
fig, axes = plt.subplots(
    nrows=2,
    ncols=2,
    figsize=(10, 8),
    sharex=True,
    sharey=True
)

for i, ax in enumerate(axes.flat, start=1):
    x_col = f"X{i}"
    y_col = f"Y{i}"

    corr = df[x_col].corr(df[y_col])

    slope, intercept = np.polyfit(
        df[x_col], df[y_col],
        deg=1
    )

    x_line = np.linspace(
        3, 21, 100
    )
    y_line = intercept + slope * x_line


    ax.scatter(
        df[x_col],
        df[y_col],
        edgecolor="black",
        s=55,
        label="Data"
    )

    ax.plot(
        x_line,
        y_line,
        color="red",
        linewidth=2,
        label="Regression"
    )

    ax.set_title(
        f"Dataset {i} (Corr = {corr:.2f})",
        fontsize=14
    )
    ax.set_xlabel(x_col, fontsize=14)
    ax.set_ylabel(y_col, fontsize=14)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("./fig/Anscombe_reg.png")
plt.show()
#%%
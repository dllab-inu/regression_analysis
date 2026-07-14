#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%%
x = np.linspace(-7, 7, num=15)
y = 50 - x**2

corr = np.corrcoef(x, y)[0, 1]
print(f"Corr: {corr:.4f}")
#%%
plt.figure(figsize=(5, 3))
plt.scatter(x, y, s=50)
plt.xlabel("x")
plt.ylabel("y")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("./fig/corr.png")
plt.show()
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

    ax.scatter(
        df[x_col],
        df[y_col],
        edgecolor="black",
        s=55
    )

    ax.set_title(
        f"Dataset {i} (Corr = {corr:.2f})",
        fontsize=14
    )
    ax.set_xlabel(x_col, fontsize=14)
    ax.set_ylabel(y_col, fontsize=14)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("./fig/Anscombe_corr.png")
plt.show()
#%%
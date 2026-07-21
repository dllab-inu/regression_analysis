#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
#%%
df = pd.read_csv("./data/Computer.Repair.Expanded.txt", sep="\t")
print(df)
print(df.columns)

x = df["Units"]
y = df["Minutes"]
#%%
X = sm.add_constant(x)
model = sm.OLS(y, X).fit()

print(model.summary())
#%%
fitted_values = model.fittedvalues
standardized_residuals = model.get_influence().resid_studentized_internal
observation_index = np.arange(1, len(df) + 1)
#%%
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].scatter(
    x,
    y,
    s=55,
    edgecolor="black",
    label="Observed"
)
sort_index = np.argsort(x)
x_sorted = x.iloc[sort_index]
y_fitted_sorted = fitted_values.iloc[sort_index]
axes[0].plot(
    x_sorted,
    y_fitted_sorted,
    color="red",
    linewidth=2,
    label="Fitted line"
)
axes[0].set_xlabel("Units", fontsize=13)
axes[0].set_ylabel("Minutes", fontsize=13)
axes[0].set_title("Minutes vs Units")
axes[0].grid(alpha=0.3)
axes[0].legend()

axes[1].scatter(
    fitted_values,
    standardized_residuals,
    s=55,
    edgecolor="black"
)
axes[1].axhline(
    y=0,
    color="black",
    linestyle="--",
    linewidth=1
)
axes[1].set_xlabel("Fitted values", fontsize=13)
axes[1].set_ylabel("Internally studentized residuals", fontsize=13)
axes[1].set_title("Residuals vs Fitted Values")
axes[1].grid(alpha=0.3)

axes[2].scatter(
    observation_index,
    standardized_residuals,
    s=55,
    edgecolor="black"
)
axes[2].axhline(
    y=0,
    color="black",
    linestyle="--",
    linewidth=1
)
axes[2].set_xlabel("Observation index", fontsize=13)
axes[2].set_ylabel("Internally studentized residuals", fontsize=13)
axes[2].set_title("Residuals vs Observation Index")
axes[2].set_xticks(observation_index)
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(
    "./fig/Computer_residual_plots.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
#%%
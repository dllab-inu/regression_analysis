#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
#%%
df = pd.read_csv("./data/NYRivers.txt", sep="\t")
print(df)
print(df.columns)

x = df[['Agr', 'Forest', 'Rsdntial', 'ComIndl']]
y = df["Nitrogen"]
#%%
X = sm.add_constant(x)
model = sm.OLS(y, X).fit()

print(model.summary())
#%%
model = sm.OLS(
    y.loc[df['River'] != 'Hackensack'], 
    sm.add_constant(
        x.loc[df['River'] != 'Hackensack']
    )
).fit()

print(model.summary())
#%%
model = smf.ols(
    "Nitrogen ~ ComIndl",
    data=df
).fit()
print(model.summary())

fitted_values = model.fittedvalues
standardized_residuals = model.get_influence().resid_studentized_internal
leverage = model.get_influence().hat_matrix_diag
observation_index = np.arange(1, len(df) + 1)
num_parameters = int(model.df_model) + 1
leverage_cutoff = 2 * num_parameters / len(df)
print(f"지레값 기준: {leverage_cutoff:.4f}")
#%%
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

x = df['ComIndl']
y = df['Nitrogen']
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
    observation_index,
    standardized_residuals,
    s=55,
    edgecolor="black"
)
axes[1].axhline(
    0,
    color="black",
    linestyle="--",
    linewidth=1
)
axes[1].set_xlabel("Observation index", fontsize=13)
axes[1].set_ylabel("Internally studentized residuals", fontsize=13)
axes[1].set_title("Residuals vs Observation Index", fontsize=13)
axes[1].grid(alpha=0.3)

axes[2].vlines(
    observation_index,
    0,
    leverage,
    color="steelblue",
    alpha=0.7
)
axes[2].scatter(
    observation_index,
    leverage,
    s=55,
    edgecolor="black"
)
axes[2].axhline(
    leverage_cutoff,
    color="red",
    linestyle="--",
    linewidth=1.5,
    label=fr"threshold$={leverage_cutoff:.3f}$"
)
axes[2].set_xlabel("Observation index", fontsize=13)
axes[2].set_ylabel("Leverage", fontsize=13)
axes[2].set_title("Leverage vs Observation Index", fontsize=13)
axes[2].set_ylim(bottom=0)
axes[2].grid(alpha=0.3)
axes[2].legend()

plt.tight_layout()
plt.savefig(
    "./fig/NYRivers_residual_leverage.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
#%%
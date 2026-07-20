#%%
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
#%%
df = pd.read_csv("./data/Supervisor.txt", sep="\t")
df.columns = df.columns.str.strip()

y_col = "Y"
x_cols = [col for col in df.columns if col != y_col]

formula = f"{y_col} ~ {' + '.join(x_cols)}"
model = smf.ols(formula, data=df).fit()

print(model.summary())
#%%
anova_table = pd.DataFrame(
    {
        "제곱합": [
            model.ess, # explained sum of squares
            model.ssr # sum of squares residual
        ],
        "자유도": [
            model.df_model,
            model.df_resid
        ],
        "평균제곱": [
            model.ess / model.df_model,
            model.ssr / model.df_resid
        ],
        "F": [
            model.fvalue,
            np.nan
        ],
        "p-value": [
            model.f_pvalue,
            np.nan
        ]
    },
    index=["회귀", "잔차"]
)

print(anova_table)
# %%
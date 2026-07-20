#%%
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import f
#%%
df = pd.read_csv("./data/Supervisor.txt", sep="\t")
df.columns = df.columns.str.strip()
df['W'] = df['X1'] + df['X3']

# 전체 모형
full_model = smf.ols(
    "Y ~ X1 + X3",
    data=df
).fit()

# 축소 모형
reduced_model = smf.ols(
    "Y ~ W",
    data=df
).fit()

print(reduced_model.summary())
#%%
sse_reduced = reduced_model.ssr
sse_full = full_model.ssr

df_reduced = reduced_model.df_model
df_full = full_model.df_model

F = (
    (sse_reduced - sse_full) / (df_full - df_reduced)
) / (
    sse_full / (len(df) - df_full - 1)
)

p_value = f.sf(
    F,
    df_full - df_reduced,
    len(df) - df_full - 1
)

print(f"축소 모형 SSE: {sse_reduced:.4f}")
print(f"전체 모형 SSE: {sse_full:.4f}")
print(f"축소 모형 자유도: {df_reduced}")
print(f"전체 모형 자유도: {df_full}")
print(f"F-통계량: {F:.4f}")
print(f"p-value: {p_value:.4f}")
# %%
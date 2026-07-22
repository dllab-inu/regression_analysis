#%%
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import f
#%%
df = pd.read_csv("./data/Preemployment.txt", sep="\t")
df.columns = df.columns.str.strip()
print(df.head())
print(df.columns)
print(df.dtypes)
#%%
y_col = "JPERF"
x_col = "TEST"
z_col = "RACE"

print(df[z_col].value_counts().sort_index())
#%%
# 축소모형
reduced_model = smf.ols(
    formula=f"{y_col} ~ {x_col}",
    data=df
).fit()

# 완전모형
full_model = smf.ols(
    formula=f"{y_col} ~ {x_col} + {x_col} : {z_col}",
    data=df
).fit()

print(reduced_model.summary())

print(full_model.summary())
#%%
sse_reduced = reduced_model.ssr
sse_full = full_model.ssr

f_statistic = (
    (sse_reduced - sse_full) / (full_model.df_model - reduced_model.df_model)
) / (
    sse_full / (len(df) - full_model.df_model - 1)
)

p_value = f.sf(
    f_statistic,
    full_model.df_model - reduced_model.df_model,
    len(df) - full_model.df_model - 1
)

print(f"SSE(RM): {sse_reduced:.4f}")
print(f"SSE(FM): {sse_full:.4f}")
print(f"F-통계량: {f_statistic:.4f}")
print(f"p-value: {p_value:.4f}")
#%%
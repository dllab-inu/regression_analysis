#%%
import pandas as pd
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
model = smf.ols(
    formula=(
        "S ~ X + C(E, Treatment(reference=3)) + M"
    ),
    data=df
).fit()

print(model.summary())
#%%
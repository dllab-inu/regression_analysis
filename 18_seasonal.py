#%%
import pandas as pd
import statsmodels.formula.api as smf
#%%
df = pd.read_csv("./data/Ski.Sales.txt", sep="\t")
df.columns = df.columns.str.strip()

print(df.head())
print(df.columns)
print(df.dtypes)
#%%
df["Quarter"] = df["Date"].apply(lambda x: x.split("/")[0])
print(df.head(10))
#%%
seasonal_model = smf.ols(
    formula=(
        "Sales ~ PDI + C(Quarter, Treatment(reference='Q4'))"
    ),
    data=df
).fit()

print(seasonal_model.summary())
#%%
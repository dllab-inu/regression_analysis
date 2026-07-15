#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#%%
data = pd.read_csv("./data/Student_Performance.csv")

data["Extracurricular Activities"] = data["Extracurricular Activities"].map({"Yes": 1, "No": 0})

X = data.drop(columns="Performance Index")
y = data["Performance Index"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
#%%
scaler = StandardScaler()

X_train = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)

X_test = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)
#%%
from sklearn.linear_model import Ridge, Lasso

ridge_model = Ridge(
    alpha=10000,
    fit_intercept=True
)
ridge_model.fit(X_train, y_train)

lasso_model = Lasso(
    alpha=3,
    fit_intercept=True,
    max_iter=10000
)
lasso_model.fit(X_train, y_train)

print("Ridge coefficients:")
for feature, coef in zip(X_train.columns, ridge_model.coef_):
    print(f"    {feature}: {coef:.3f}")

print("\nLASSO coefficients:")
for feature, coef in zip(X_train.columns, lasso_model.coef_):
    print(f"    {feature}: {coef:.3f}")
#%%
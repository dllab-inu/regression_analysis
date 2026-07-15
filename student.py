#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
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

model = LinearRegression(fit_intercept=True)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"Intercept: {model.intercept_:.1f}")
print("Coefficients:")

for feature, coef in zip(
    X_train.columns,
    model.coef_
):
    print(f"    {feature}: {coef:.1f}")
#%%
# from sklearn.metrics import mean_squared_error
# mse = mean_squared_error(y_test, y_pred)
mse = ((y_test - y_pred) ** 2).mean().item()
print(f"Mean Squared Error on Test Set: {mse:.1f}")
#%%
mape = ((y_test - y_pred) / y_test).abs().mean().item() * 100
print(f"Mean Absolute Percentage Error on Test Set: {mape:.1f}%")
#%%
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

alphas = np.logspace(-2, 10, 100)

ridge_coefficients = []
lasso_coefficients = []

for alpha in alphas:
    ridge_model = Ridge(
        alpha=alpha,
        fit_intercept=True
    )
    ridge_model.fit(X_train, y_train)
    ridge_coefficients.append(ridge_model.coef_)

    lasso_model = Lasso(
        alpha=alpha,
        fit_intercept=True,
        max_iter=10000
    )
    lasso_model.fit(X_train, y_train)
    lasso_coefficients.append(lasso_model.coef_)

ridge_coefficients = np.asarray(ridge_coefficients)
lasso_coefficients = np.asarray(lasso_coefficients)
#%%
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

for j, feature in enumerate(X_train.columns):
    axes[0].plot(
        alphas,
        ridge_coefficients[:, j],
        label=feature
    )
    axes[1].plot(
        alphas,
        lasso_coefficients[:, j],
        label=feature
    )

for ax in axes:
    ax.set_xscale("log")
    ax.axhline(
        y=0,
        color="black",
        linestyle="--",
        linewidth=1
    )
    ax.set_xlabel("Regularization hyperparameter")
    ax.grid(alpha=0.3, linestyle='--')

axes[0].set_title("Ridge coefficient path")
axes[0].set_ylabel("Coefficient")
axes[1].set_title("LASSO coefficient path")
axes[1].legend(
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

plt.tight_layout()
plt.savefig("./fig/regularization_path.png")
plt.show()
#%%
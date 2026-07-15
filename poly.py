#%%
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
#%%
np.random.seed(42)

n = 100
x = np.linspace(-3, 3, n)
epsilon = np.random.normal(loc=0, scale=1, size=n)
y = 3 + 2 * x + 1.5 * x**2 + epsilon
#%%
def fit_polynomial(x, y, degree):
    X = np.column_stack([
        x ** j for j in range(degree + 1)
    ])
    model = sm.OLS(y, X).fit()
    return model

linear_model = fit_polynomial(x, y, degree=1)
polynomial_model = fit_polynomial(x, y, degree=2)

# 적합값
linear_fitted = linear_model.fittedvalues
polynomial_fitted = polynomial_model.fittedvalues

# 잔차
linear_residuals = linear_model.resid
polynomial_residuals = polynomial_model.resid
#%%
print(linear_model.summary())

print(polynomial_model.summary())
#%%
# 잔차분석
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].scatter(
    x, y,
    alpha=0.7,
    color="gray",
    label="Observed data"
)
axes[0].plot(
    x, linear_fitted,
    color="red",
    linewidth=2,
    label="Linear regression"
)
axes[0].plot(
    x, polynomial_fitted,
    color="blue",
    linewidth=2,
    label="Polynomial regression"
)
axes[0].set_title("Observed data and fitted models")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
axes[0].legend()

axes[1].scatter(
    linear_fitted, y,
    alpha=0.7,
    color="red"
)
axes[1].axline(
    (0, 0),
    slope=1,
    color="black",
    linewidth=2,
    linestyle="--"
)
axes[1].set_title("linear regression")
axes[1].set_xlabel("Fitted values")
axes[1].set_ylabel("y")

axes[2].scatter(
    polynomial_fitted, y,
    alpha=0.7,
    color="blue"
)
axes[2].axline(
    (0, 0),
    slope=1,
    color="black",
    linewidth=2,
    linestyle="--"
)
axes[2].set_title("polynomial regression")
axes[2].set_xlabel("Fitted values")
axes[2].set_ylabel("y")

plt.tight_layout()
plt.savefig("./fig/residual_poly.png")
plt.show()
#%%
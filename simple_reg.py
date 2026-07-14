#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
#%%
df = pd.read_csv("./data/Computer.Repair.txt", sep="\t")
print(df)
print(df.columns)

x = df['Units']
y = df['Minutes']
#%%
X = sm.add_constant(x) # 상수항 추가
model = sm.OLS(y, X).fit()

print(model.summary())
#%%
intercept = model.params["const"]
slope = model.params['Units']

print(f"절편: {intercept:.4f}")
print(f"기울기: {slope:.4f}")
#%%
x_grid = np.linspace(x.min(), x.max(), 100)
y_pred = model.predict(sm.add_constant(x_grid))

plt.figure(figsize=(5, 4))
plt.scatter(
    x, y,
    edgecolor="black",
    s=60,
    label="data"
)

plt.plot(
    x_grid,
    y_pred,
    color='green',
    linewidth=3,
    label="regression line"
)

plt.xlabel('Units')
plt.ylabel('Minutes')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("./fig/computer_reg.png")
plt.show()
#%%
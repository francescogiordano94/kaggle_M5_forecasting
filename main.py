# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')

# %%
df_sales = pd.read_csv('sales_train_validation.csv')
df_calendar = pd.read_csv('calendar.csv')
df_sell_prices = pd.read_csv('sell_prices.csv')


# %%
df_sales.head()


# %%
df_sales.head()

# %%
df_sell_prices.head()


# %%
df_sell_prices.shape


# %%
plt.figure()
plt.plot(list(range(1913)),df_sales.iloc[11][6:])
plt.show()
# %%
# plt.figure()
# plt.plot(df_sales.iloc[0][6:])
# plt.show()


# %%
df_calendar.iloc[-(28*2)+1:-27]

# %%
df_sales.head()

# %%
items = df_sales.iloc[0:200]
print(item)
# %%
predict_next = 10

# CONTINUE HERE with the split
series = df_sales.iloc[0][6:]
n_steps = 365
predict_next = 10
n_batches = len(series) - n_steps - predict_next + 1
## X:split in n_batches series of 365 where each series is one step ahead of the previous
## y: the next <predict_next> elements after the corresponding series in X

X = np.empty((n_batches, n_steps, 1))
for step in range(n_steps):
    X[:, step, 0] = series[step : n_batches + step]

Y = np.empty((n_batches, predict_next))

for pred in range(predict_next):
    Y[:, pred] = series[pred + n_steps : pred + n_steps + n_batches]
# %% 

from sklearn.model_selection import train_test_split

X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.2, shuffle=False)

print(X_train.shape)
print(Y_train.shape)
print(X_valid.shape)
print(Y_valid.shape)


# %%
from tensorflow import keras
model = keras.models.Sequential([
    keras.layers.GRU(10, input_shape=(None, 1), return_sequences=False),
    keras.layers.Dense(10),
])

# %%
model.summary()


# %%
model.compile(loss="mean_squared_error", optimizer="adam")

model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid),
            epochs=100, callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])


# %%

y_pred = model.predict(series[np.newaxis, ...])
keras.losses.mse(Y_valid[-1],np.around(y_pred[-1]))
# %%
print(Y_valid[-1], np.around(y_pred[-1]))

# %%
series_test = series.astype(np.float32)
series_test = np.repeat(series_test,5)[np.newaxis, :, np.newaxis]
print(series_test.shape)
model.predict(series_test)

# %%

plt.figure()
plt.plot(range(1913), series)
plt.show()

# %%

for n in range(1200,1210):
    plt.figure()
    plt.plot(range(365),X[n])
    plt.show()

# %%

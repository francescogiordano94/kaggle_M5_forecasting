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
item0 = df_sales.iloc[0]
print(item0)

series0 = item0[6:]


# %%
model = keras.models.Sequential([
    keras.layers.SimpleRNN(20, input_shape=[None, 1], return_sequences=False),
    keras.layers.Dense(10),
])

# %%
model.summary()


# %%
model.compile(loss="mean_squared_error", optimizer="adam")


# %%
def create_dataset(dataset, look_back=20, predict_next=10):
    dataX, dataY = [], []
    for i in range(len(dataset)-predict_next-look_back+1):
        dataX.append(dataset[i:(i+look_back)])
        dataY.append(dataset[(i + look_back):(i + look_back+predict_next)])
    return np.array(dataX)[..., np.newaxis].astype(np.float32), np.atleast_3d(dataY).astype(np.float32)

# %%
Xtrain, Ytrain = create_dataset(series0[:1900])

print(Xtrain.shape)
print(Ytrain.shape)
# %%
model.fit(Xtrain, Ytrain, epochs = 100)

# %%
#model.predict(series0[1900:1910])

# %%
model = keras.models.Sequential([
    keras.layers.SimpleRNN(1, input_shape=[None, 1], return_sequences=True),
    keras.layers.Dense(1)])
model.summary()
model.compile(loss="mean_squared_error", optimizer="adam")
# %%
model.fit(X_train, y_train, epochs=5)

# %%

look_forward=10
X = series0[np.newaxis, :, np.newaxis]
print(X.shape)
Y = np.empty((1, len(series0) - look_forward, look_forward))
print(Y.shape)

for step_ahed in range(1, look_forward + 1):
    Y[:, :, step_ahed-1] = X[:, step_ahed:step_ahed+len(series0)-look_forward, 0]



# %%

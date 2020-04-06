# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import pandas as pd
import matplotlib.pyplot as plt


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
plt.figure()
plt.plot(df_sales.iloc[0][6:])
plt.show()


# %%
df_calendar.iloc[-(28*2)+1:-27]

# %%
df_sales.head()

# %%
item0 = df_sales.iloc[0:50]
print(item0)

series0 = item0[6:]

# %%
from tensorflow import keras

keras.__version__

# %%
model = keras.models.Sequential([
    keras.layers.SimpleRNN(20, input_shape=[None, 1]),
    keras.layers.Dense(10),
])

# %%
model.summary()


# %%
model.compile(loss="mean_squared_error", optimizer="adam")

# %%
X_train, y_train = series0[:1900], series0[1900:-1]
X_train = series0[:1900]

# %%
def create_dataset(dataset, look_back=20, predict_next=10):
    dataX, dataY = [], []
    for i in range(len(dataset)-predict_next-look_back+1):
        dataX.append(dataset[i:(i+look_back)])
        dataY.append(dataset[(i + look_back):(i + look_back+predict_next)])
    return np.atleast_3d(dataX).astype(np.float32), np.atleast_3d(dataY).astype(np.float32)

# %%
Xtrain, Ytrain = create_dataset(series0[:1900])
# %%
model.fit(Xtrain, Ytrain, epochs = 100)
# %%
#model.predict(series0[1900:1910])


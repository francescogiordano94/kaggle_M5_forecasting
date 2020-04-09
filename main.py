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
def create_dataset(dataset, look_back=20, predict_next=10):
    dataX, dataY = [], []
    for i in range(len(dataset)-predict_next-look_back+1):
        dataX.append(dataset[i:(i+look_back)])
        dataY.append(dataset[(i + look_back):(i + look_back+predict_next)])
    return np.array(dataX)[..., np.newaxis].astype(np.float32), np.atleast_3d(dataY).astype(np.float32)

# %%
Xtrain, Ytrain = create_dataset(series0[:1900])

# %%
model.fit(Xtrain, Ytrain, epochs = 100)

# %%
#model.predict(series0[1900:1910])

# %%
X_train = np.array([[[0], [1], [2], [1]]])
print(X_train.shape)
y_train = np.array([[[5]]])
print(y_train.shape)
model = keras.models.Sequential([
    keras.layers.SimpleRNN(1, input_shape=[None, 1], return_sequences=True),
    keras.layers.Dense(1)])
model.summary()
model.compile(loss="mean_squared_error", optimizer="adam")
# %%
model.fit(X_train, y_train, epochs=5)

# %%



def generate_series(batch_size, n_steps):
    time = np.linspace(0, 1, n_steps)
    series = []
    noise= np.random.rand(batch_size) * 2 - 1
    for i in range(batch_size):
        series.append(np.sin(time) + 1.0*noise[i])
    return np.array(series)[..., np.newaxis]

batch_size = 3000
n_steps = 41
series = generate_series(batch_size, n_steps)
print(series.shape)

X = series[:, :n_steps-1, :]
y = series[:, -1, :]

# X_train = series[:2000, :n_steps-1, :]
# y_train = series[:2000, -1, :]

# X_valid = series[2000:, :n_steps-1, :]
# y_valid = series[2000:, -1, :]

from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)

model = keras.models.Sequential([
    #reshape missing
    keras.layers.Flatten(input_shape=[n_steps-1, 1]),
    keras.layers.Dense(1)
])

model.compile(loss="mse", optimizer='adam')
model.fit(X_train, y_train,
          epochs = 1000,
          validation_data=(X_valid, y_valid),
          callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)])

# %%

model.evaluate(X_valid, y_valid)
#5.54e-15
# %%
rnn_model = keras.models.Sequential([
    keras.layers.SimpleRNN(4, input_shape=[n_steps-1, 1], return_sequences=True),
    keras.layers.SimpleRNN(1)
])

rnn_model.summary()

rnn_model.compile(loss="mse", optimizer='adam')
rnn_model.fit(X_train, y_train,
          epochs = 1000,
          validation_data=(X_valid, y_valid),
          callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)])

# %%
rnn_model.evaluate(X_valid, y_valid)



# %%
a = np.array([[[0],[1]],[[2],[3]],[[4],[5]]])[..., np.newaxis]
print((keras.layers.Flatten()(a).shape))
a.shape
# %%

model = keras.models.Sequential([
    keras.layers.SimpleRNN(1, input_shape=[None, 1])
])

optimizer = keras.optimizers.Adam()
model.compile(loss="mse", optimizer=optimizer)

# %%

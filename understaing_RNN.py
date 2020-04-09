# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import tensorflow as tf
from tensorflow import keras


# %%
def generate_time_series(batch_size, n_steps):
    freq1, freq2, offsets1, offsets2 = np.random.rand(4, batch_size, 1)
    time = np.linspace(0, 1, n_steps)
    series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10))  #   wave 1
    series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20)) # + wave 2
    series += 0.1 * (np.random.rand(batch_size, n_steps) - 0.5)   # + noise
    return series[..., np.newaxis].astype(np.float32)


# %%
batch_size = 3000
n_steps =50

series = generate_time_series(batch_size, n_steps)


# %%
series.shape


# %%
X, y = series[:, :n_steps-1, :], series[:, -1, :]
print(X.shape)
print(y.shape)


# %%
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
print(X_train.shape)
print(y_valid.shape)


# %%
model = keras.models.Sequential([
    keras.layers.SimpleRNN(1, input_shape=[None, 1], return_sequences=False)
])

model.compile(loss='mse', optimizer='adam')
history = model.fit(X_train, y_train, epochs=1, validation_data=(X_valid, y_valid))


# %%
y_pred = model.predict(X_valid)
print(y_pred.shape)
print(y_valid.shape)


# %%
## As expected the computation of the mse fails if y_pred is obtained with return_sequences=True due to shape mismatch, why id does not fails during trainig? which data is it using to compute the mse during trainig?
mse = keras.losses.mse(y_valid, y_pred)
mse


# %%

mse = ((y_pred - y_valid)**2).mean(axis=0)
mse #agree with val_loss


# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import tensorflow as tf
from tensorflow import keras

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

X_train = series[:2000, :n_steps-1, :]
y_train = series[:2000, -1, :]

X_valid = series[2000:, :n_steps-1, :]
y_valid = series[2000:, -1, :]




model = keras.models.Sequential([
    #reshape missing
    keras.layers.Flatten(input_shape=[n_steps-1, 1]),
    keras.layers.Dense(1)
])

model.compile(loss="mse", optimizer='adam')
model.fit(X_train, y_train,
          epochs = 10,
          validation_data=(X_valid, y_valid),
          callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)])

# %%

model.evaluate(X_valid, y_valid)
#5.54e-15
# %%
rnn_model = keras.models.Sequential([
    keras.layers.SimpleRNN(4, input_shape=[None, 1], return_sequences=False),
    # keras.layers.SimpleRNN(1, return_sequences=False)
])

rnn_model.summary()
# %%
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
batch_size = 1000
n_steps = 50 + 10
series = generate_series(batch_size, n_steps)

X, y = series[:, :n_steps -10, [0,0]], series[:, n_steps-10:, [0, 0]]

display(X.shape)
display(y.shape)

# %%
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)

model = keras.models.Sequential([
    keras.layers.SimpleRNN(20, input_shape=(50,2), return_sequences=False),
    # keras.layers.Dense(20)
])
model.compile(optimizer='adam', loss='mse')
model.summary()

model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=10)
# %%
y_pred = model.predict(X_valid)
display(y_pred)
display(y_pred.shape)
y_valid.shape
# %%
batch_size = 2
n_steps = 5
look_ahead=2

#series = generate_series(batch_size, n_steps + look_ahead)
series = np.array(range(batch_size*(n_steps+look_ahead))).reshape(2,7)[...,np.newaxis]
print(series)
print(series.shape)
Y = np.empty((batch_size, n_steps, look_ahead))

for step_ahead in range(1, look_ahead+1):
    Y[:, :, step_ahead -1] = series[:, step_ahead: step_ahead + n_steps, 0]

# %%


## NN's implementation of the split
# %%
from sklearn.model_selection import train_test_split

series0 = item0[6:]

n_steps = 365
predict_next = 10
n_batches = len(series0) - n_steps - predict_next + 1
X = np.empty((n_batches, n_steps, 1))
y = np.empty((n_batches, predict_next))


# %%

## X:split in n_batches series of 365 where each series is one step ahead of the previous
## y: the next <predict_next> elements after the corresponding series in X
for step in range(n_steps):
    X[:, step, 0] = series0[step:(n_batches + step)]
for step in range(predict_next):
    y[:, step] = series0[(step+n_steps):(step+n_steps+n_batches)]
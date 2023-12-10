import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# Suppress TensorFlow warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


nvda = pd.read_csv("https://raw.githubusercontent.com/tiagopint0/Trabalho2PCD/main/SourceFile/NVDA.csv")


# Selecting only the Close values and defining it as a floating
nvdadata = nvda[["Close"]].values.astype(float)

# Delete rows with NaN
nvdadata = nvdadata[~np.isnan(nvdadata).any(axis=1)]

# Normalizing the data
norm = MinMaxScaler(feature_range=(0, 1))
nvdadata_scaled = norm.fit_transform(nvdadata)

# Creating the time sequence
timesq = int(input("Time Series in days: "))
print("\n")

X, y = [], []
for i in range(len(nvdadata_scaled) - timesq):
    X.append(nvdadata_scaled[i:i + timesq])
    y.append(nvdadata_scaled[i + timesq])

X, y = np.array(X), np.array(y)

# Selecting the index for the Data split
split_index = int(len(X) * (1 - 0.3))

# Splitting the data
X_train, y_train = X[:split_index], y[:split_index]
X_test, y_test = X[split_index:], y[split_index:]


# Applying the model
model = Sequential()
model.add(LSTM(units=50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(units=1))
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss="mean_squared_error")

# Settingt he number fo epochs
epochs = int(input("Number of loops through the DataSet: "))
print("\n")

# Training the model
model.fit(X_train, y_train, epochs=epochs, batch_size=32)

# Evaluate the model on the test set
losstst = model.evaluate(X_test, y_test)


# Evaluate the model on the test set
losstr = model.evaluate(X_train, y_train)


# Make predictions using the trained model on both training and test sets
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Inverse transform the predictions to the original scale
train_predictions = norm.inverse_transform(train_predictions)
test_predictions = norm.inverse_transform(test_predictions)
y_train_original = norm.inverse_transform(y_train)
y_test_original = norm.inverse_transform(y_test)



# Plot the actual vs. predicted values for training set and test set
plt.figure(figsize=(12, 6))


# Actual values for test set
plt.plot(np.concatenate([np.full_like(y_train_original, np.nan), y_test_original]), label="Real", color="blue", linestyle="dashed")


# Predicted values for training and test sets
plt.plot(np.concatenate([train_predictions, test_predictions]), label="Predictions (Test)", color="red")

# Actual values for training set (same color for both training and test sets)
plt.plot(train_predictions, label="Predictions (Training)", color="green")

plt.legend()
plt.title("LSTM Model: Real vs. Predicted")
plt.show()

print(f"Test Loss: {losstst}")
print(f"Train Loss: {losstr}")

#%%

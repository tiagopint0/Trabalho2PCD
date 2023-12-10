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

# Load the data
nvda = pd.read_csv("https://raw.githubusercontent.com/tiagopint0/Trabalho2PCD/main/SourceFile/NVDA.csv")


# Extract the 'Close' prices as the target variable
nvdadata = nvda[['Close']].values.astype(float)

# Drop rows with NaN values
nvdadata = nvdadata[~np.isnan(nvdadata).any(axis=1)]

# Normalize the data using Min-Max scaling
norm = MinMaxScaler(feature_range=(0, 1))
nvdadata_scaled = norm.fit_transform(nvdadata)

# Create sequences of data for LSTM
timesq = int(input("Time Series in days: "))  # You can adjust this value based on your needs
print("\n")

X, y = [], []
for i in range(len(nvdadata_scaled) - timesq):
    X.append(nvdadata_scaled[i:i + timesq])
    y.append(nvdadata_scaled[i + timesq])

X, y = np.array(X), np.array(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and compile the model
model = Sequential()
model.add(LSTM(units=50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(units=1))
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Set the number of epochs
epochs = int(input("Number of loops through the DataSet: "))
print("\n")

# Train the model
model.fit(X_train, y_train, epochs=epochs, batch_size=32)

# Evaluate the model on the test set
loss = model.evaluate(X_test, y_test)


# Evaluate the model on the test set
loss = model.evaluate(X_train, y_train)


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

# Actual values (same color for both training and test sets)
plt.plot(y_train_original, label='Actual', color='blue', linestyle='dashed')
plt.plot(y_test_original, color='blue', linestyle='dashed')

# Predicted values for training set
plt.plot(train_predictions, label='Predicted (Training)', color='orange')

# Predicted values for test set
plt.plot(test_predictions, label='Predicted (Test)', color='red')

plt.legend()
plt.title('LSTM Model: Actual vs. Predicted')
plt.show()

print(f"Test Loss: {loss}")
print(f"Train Loss: {loss}")

#%%

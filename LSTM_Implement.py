import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, r2_score

def lstm_model(nvda):
    # Suppress TensorFlow warnings
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


    nvda = pd.read_csv("https://raw.githubusercontent.com/tiagopint0/Trabalho2PCD/main/SourceFile/NVDA.csv")


    # Selecting only the Close values and defining it as a floating
    nvda_data = nvda[["Close"]].values.astype(float)

    # Delete rows with NaN
    nvda_data = nvda_data[~np.isnan(nvda_data).any(axis=1)]

    # Normalizing the data
    norm = MinMaxScaler(feature_range=(0, 1))
    nvda_data_scaled = norm.fit_transform(nvda_data)

    # Creating the time sequence
    timesq = (int(input("Time Series in days: ")))
    print("\n")

    X, y = [], []
    for i in range(len(nvda_data_scaled) - timesq):
        X.append(nvda_data_scaled[i:i + timesq])
        y.append(nvda_data_scaled[i + timesq])

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

    # Setting he number fo epochs
    epochs = int(input("Number of loops through the DataSet: "))
    print("\n")

    # Training the model
    model.fit(X_train, y_train, epochs=epochs)

    # Making predictions on the trained model with both training and test sets
    test_predictions = model.predict(X_test)

    # Inverting the normalization
    y_train_original = norm.inverse_transform(y_train)
    y_test_original = norm.inverse_transform(y_test)
    test_predictions = norm.inverse_transform(test_predictions)

    # Calculating the RMSE and R2
    rmse = (np.sqrt(mean_squared_error(y_test_original, test_predictions)))
    r2 = r2_score(y_test_original, test_predictions)

    # Plotting the real vs. predicted values
    plt.figure(figsize=(10, 5))

    # Plotting the predicted values of the training set
    plt.plot(y_test_original, label="Real", color="blue", linestyle="dashed")

    # Plotting the predicted values of the training set
    plt.plot(test_predictions, label="Predictions", color="red")

    plt.legend()
    plt.title("LSTM Model: Real vs. Predicted")
    plt.show()
    print("\n")

    print("The model performance for testing set")
    print("--------------------------------------")
    print("RMSE is : ", format(rmse))
    print("R2 score is ", format(r2))

    print("\n")

    if rmse < 1:
        print(f"The RMSE score of {rmse} is relatiively low, this indicates the model predictions are close to the actual prices")
    elif 1 < rmse < 3:
        print(f"The RMSE score of {rmse} is not very low, this indicates the model predictions are close but whit some distance to the actual prices")
    elif 3 < rmse < 6:
        print(f"The RMSE score of {rmse} is not low, this indicates the model predictions are relatively far but whit some distance to the actual prices")
    else:
        print(f"The RMSE score of {rmse} is high, this indicates the model predictions are far to the actual prices")

    print("\n")

    if round(r2,2) >0.99: # type: ignore
        print(f"The R2 score of {r2} is very close to 1, this indicates the model explains almost all the variability in the prices")
    elif 0.9 < round(r2,2) < 0.99: # type: ignore
        print(f"The R2 score of {r2} is close to 1, this indicates the model explains a good part the variability in the prices")
    elif 0.75 < round(r2,2) < 0.9: # type: ignore
        print(f"The R2 score of {r2} has some distance to 1, this indicates the model explains a some part the variability in the prices")
    else:
        print(f"The R2 score of {r2} is not close to 1, this indicates the model doesn't explain the variability in the prices well")

    # Imputing predictions
    prediction_days = int(input("Enter the number of days for prediction: "))

    # Creating an array with the last time series
    last_data = nvda_data_scaled[-timesq:]

    # Creating a list to store the predicted values
    predictions = []

    # Implementing the  prediction time series
    for i in range(prediction_days):
        # Reshaping the array to match the model input shape
        last_data_reshaped = last_data.reshape(1, timesq, 1)

        # Predicting the Close
        predicted_value = model.predict(last_data_reshaped)
        predicted_value = norm.inverse_transform(predicted_value)
        predictions.append(predicted_value[0, 0])
        last_data = np.append(last_data[1:], predicted_value)

    print("\n")
    print("\n")

    # Print the predicted values
    print("Predicted values for the next", prediction_days, "days:")
    print(predictions)


if __name__ == '__main__':
    lstm_model(nvda=pd.DataFrame)


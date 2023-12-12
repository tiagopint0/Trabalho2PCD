import pandas as pd
import numpy as np
from sklearn import datasets
import ta
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

nvda = pd.read_csv("https://raw.githubusercontent.com/tiagopint0/Trabalho2PCD/main/SourceFile/NVDA.csv")

nvda_reord = ["Open", "High", "Low", "Volume", "Close"]

nvda_data = nvda[nvda_reord]

# Dropping NaNs
nvda_data = nvda_data.dropna()

# Defining the Features Data
nvda_feat = nvda_data.iloc[:, 0:nvda_data.shape[1]-1]

# Defining the Target Data
nvda_test = nvda_data["Close"]

# Creating the Model
X_train, X_test, Y_train, Y_test = train_test_split(nvda_feat.values, nvda_test.values, test_size=0.3)
lr = LinearRegression()
lr.fit(X_train, Y_train)

# Calculating the RMSE and R2
y_test_predict = lr.predict(X_test)
rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
r2 = r2_score(Y_test, y_test_predict)

"""
# Not good to illustrate the model performance

# Plotting the real vs. predicted values
plt.figure(figsize=(10, 5))

# Plotting the predicted values of the training set
plt.plot(Y_test, label="Real", color="blue", linestyle="dashed")

# Plotting the predicted values of the training set
plt.plot(X_test, label="Predictions", color="red")

plt.legend()
plt.title("LR Model: Real vs. Predicted")
plt.show()
"""

# Printing the Model Evaluation
print("The model performance for testing set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))

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

if r2 >0.99:
    print(f"The R2 score of {r2} is very close to 1, this indicates the model explains almost all the variability in the prices")
elif 0.9 < rmse < 0.99:
    print(f"The R2 score of {r2} is close to 1, this indicates the model explains a good part the variability in the prices")
elif 0.75 < rmse < 0.9:
    print(f"The R2 score of {r2} has some distance to 1, this indicates the model explains a some part the variability in the prices")
else:
    print(f"The R2 score of {r2} is not close to 1, this indicates the model doesn't explain the variability in the prices well")



#%%

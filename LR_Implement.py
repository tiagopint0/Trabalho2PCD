import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

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

# Calculating the Accuracy
print(X_test, Y_test)

# Calculating the RMSE and R2
y_test_predict = lr.predict(X_test)
rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
r2 = r2_score(Y_test, y_test_predict)


# Printing the Model Evaluation
print("The model performance for testing set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))

#%%

# percentagePrediction_LR

This code is an implementation of a linear regression model to predict the percentage score based on the number of hours studied. It uses the pandas library to read and manipulate data from a CSV file, matplotlib library to visualize the data, and scikit-learn library to perform machine learning tasks.

The first part of the code loads the data from a CSV file using the read_csv method from pandas and then plots the data using matplotlib. The x-axis represents the hours studied, and the y-axis represents the percentage score.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the data
url = "http://bit.ly/w-data"
data = pd.read_csv(url)

# Plot the data
data.plot(x='Hours', y='Scores', style='o')
plt.title('Hours vs Percentage')
plt.xlabel('Hours studied')
plt.ylabel('Percentage Score')
plt.show()

The second part of the code prepares the data by splitting it into training and testing sets using train_test_split from scikit-learn. It also creates X and y arrays containing the input and output variables, respectively.
# Prepare the data
X = data.iloc[:, :-1].values
y = data.iloc[:, 1].values

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

The third part of the code trains the linear regression model using the fit method from scikit-learn. It then predicts the test set results using the predict method.
# Train the model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

The final part of the code predicts the percentage score for a given number of hours studied (9.25 hours) using the trained model and the predict method. The result is displayed on the console using the print function.

# Predict the test set results
y_pred = regressor.predict(X_test)

# Predict the percentage score for 9.25 hours of study per day
hours = 9.25
score = regressor.predict([[hours]])
print("Predicted score for {} hours of study per day: {:.2f}%".format(hours, score[0]))
CODE IMPLEMENTATIONM - https://youtu.be/s5msenMCftQ

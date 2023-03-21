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

# Prepare the data
X = data.iloc[:, :-1].values
y = data.iloc[:, 1].values

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict the test set results
y_pred = regressor.predict(X_test)

# Predict the percentage score for 9.25 hours of study per day
hours = 9.25
score = regressor.predict([[hours]])
print("Predicted score for {} hours of study per day: {:.2f}%".format(hours, score[0]))

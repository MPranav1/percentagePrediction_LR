# percentagePrediction_LR

This code is an implementation of a linear regression model to predict the percentage score based on the number of hours studied. It uses the pandas library to read and manipulate data from a CSV file, matplotlib library to visualize the data, and scikit-learn library to perform machine learning tasks.

The first part of the code loads the data from a CSV file using the read_csv method from pandas and then plots the data using matplotlib. The x-axis represents the hours studied, and the y-axis represents the percentage score.

The second part of the code prepares the data by splitting it into training and testing sets using train_test_split from scikit-learn. It also creates X and y arrays containing the input and output variables, respectively.

The third part of the code trains the linear regression model using the fit method from scikit-learn. It then predicts the test set results using the predict method.

The final part of the code predicts the percentage score for a given number of hours studied (9.25 hours) using the trained model and the predict method. The result is displayed on the console using the print function.

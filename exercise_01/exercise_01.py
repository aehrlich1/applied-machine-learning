"""
This module does blah blah.

"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression

# Define OS independent paths
working_dir_path = os.path.dirname(__file__)
car_price_path = os.path.join(working_dir_path, "ressources\\car_price.csv")


# Create the data frames
df = pd.read_csv(car_price_path, index_col="car_ID")
data = df[['curbweight', 'enginesize', 'highwaympg',
           'horsepower', 'citympg', 'peakrpm', 'price']]


# Perform preliminary analysis
scatter_matrix = pd.plotting.scatter_matrix(data)
corr_matrix = pd.DataFrame.corr(data)


# Split data into training and test data
X = data[['curbweight', 'enginesize', 'highwaympg',
         'horsepower', 'citympg', 'peakrpm']]
y = data.price
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


# 2a. Linear Regression
# Set up the object and train the model
linear_regression = LinearRegression()
linear_regression.fit(X_train, y_train)

# Validate the model
cvs = cross_val_score(linear_regression, X_test, y_test)
print(f'Cross validation score: {cvs}')

# Calculate the mean scpre
mean = np.mean(cvs)
print(f'Mean cross validation score: {mean:.3f}')


# 2b. Polynomial Regression
#

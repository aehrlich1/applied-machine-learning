"""
This script solves the exercises from exercise_01
of the class Applied Machine Learning (AML)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline

# Define OS independent paths
working_dir_path = os.path.dirname(__file__)
car_price_path = os.path.join(working_dir_path, "ressources\\car_price.csv")

# Create the data frames
data = pd.read_csv(car_price_path, index_col="car_ID")
data = data[['curbweight', 'enginesize', 'highwaympg',
             'horsepower', 'citympg', 'peakrpm', 'price']]

# Split data into training and test data
X = data[['curbweight', 'enginesize', 'highwaympg',
         'horsepower', 'citympg', 'peakrpm']]
y = data.price
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


def preliminary_analysis():
    """"
    Documentation of function
    """
    # Perform preliminary analysis
    scatter_matrix = pd.plotting.scatter_matrix(data, figsize=(10, 10))
    plt.savefig(fname=working_dir_path + '/scatter_matrix.pdf')
    corr_matrix = pd.DataFrame.corr(data)
    print('Correlation Matrix:')
    print(corr_matrix, end='\n\n')


# 2a. Linear Regression
def linear_regression():
    """"
    Documentation of function
    """
    print('========== 2a. Linear Regression ==========')

    # Set up the object and train the model
    lin_regression = LinearRegression().fit(X_train, y_train)

    # Validate the model
    cvs = cross_val_score(lin_regression, X_train, y_train)
    print(f'Cross validation score: {cvs}')

    # Calculate the mean score
    mean = np.mean(cvs)
    print(f'Mean cross validation score: {mean:.3f}', end='\n\n')


# 2b. Polynomial Regression
def polynomial_regression():
    """"
    Documentation of function
    """
    print('========== 2b. Polynomial Regression ==========')

    # create the pipeline
    pipeline = Pipeline([
        ("poly_features", PolynomialFeatures()),
        ("lin_reg", LinearRegression())
    ])

    # perform cross validation
    grid_search = GridSearchCV(
        pipeline, {"poly_features__degree": [1, 2, 3, 4, 5]})
    grid_search.fit(X_train, y_train)

    # print best hyperparamter alpha for polynomial regrssion
    print(f'Best paramters found: {grid_search.best_params_}')

    # validate the model
    cvs = cross_val_score(grid_search, X_train, y_train)
    print(f'Cross validation score: {cvs}')

    # Calculate the mean score
    mean = np.mean(cvs)
    print(f'Mean cross validation score: {mean:.3f}', end='\n\n')


# 2c. Ridge Regression
def ridge_regression():
    """"
    Documentation of function
    """
    print('========== 2c. Ridge Regression ==========')

    # create the pipeline
    pipeline = Pipeline([
        ("poly_features", PolynomialFeatures()),
        ("ridge_reg", Ridge())
    ])

    grid_search = GridSearchCV(
        pipeline, {"poly_features__degree": [1],
                   "ridge_reg__alpha": [0, 1, 2, 5, 10, 100, 1e3, 1e4, 1e5]})
    grid_search.fit(X_train, y_train)

    # print best hyperparamter alpha for ridge regression
    print(f'Best paramters found: {grid_search.best_params_}')

    # perform cross validation
    cvs = cross_val_score(grid_search, X_train, y_train)
    print(f'Cross validation score: {cvs}')

    # Calculate the mean score
    mean = np.mean(cvs)
    print(f'Mean cross validation score: {mean:.3f}', end='\n\n')


# Run Tasks
linear_regression()
polynomial_regression()
ridge_regression()

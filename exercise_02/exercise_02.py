"""
This script solves the exercises from exercise_02
of the class Applied Machine Learning (AML)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_absolute_error, make_scorer


# define OS independent paths
working_dir_path = os.path.dirname(__file__)
wine_quality_path = os.path.join(
    working_dir_path, "ressources\\winequality_red.csv")

# create the data frames and split the data
data = pd.read_csv(wine_quality_path)
data_train, data_test = train_test_split(
    data.copy(), test_size=0.2, random_state=42)
# print(data_train.head())

# output minimum quality of the wines
print(f'Min Wine Quality: {data.quality.min()}')
print(f'Max Wine Quality: {data.quality.max()}')

# create a histogram for the wine quality
data.quality.hist()
plt.savefig(fname=working_dir_path + '/output/histogram.pdf')


# 1a. Kernel Methods
def kernel_methods():
    """"
    Comparison of different kernel methods
    """

    print('========== 1a. Kernel Methods ==========')

    X_train, y_train = data_train.drop(['quality'], axis=1), data_train.quality
    X_test, y_test = data_test.drop(['quality'], axis=1), data_test.quality

    # score = make_scorer(mean_absolute_error, greater_is_better=False)

    kernel_ridge_pipeline = Pipeline([
        ('scale', StandardScaler()),
        ('ridge', KernelRidge(kernel='rbf'))
    ])

    kernel_ridge_model = GridSearchCV(kernel_ridge_pipeline, [{
        'ridge__alpha': [0.001, 0.01, 0.1, 1],
        'ridge__gamma': [0.001, 0.01, 0.03, 0.05, 0.1],
    }],
        n_jobs=-1,
        scoring='neg_mean_absolute_error',
        cv=5)

    kernel_ridge_model.fit(X_train, y_train)

    kernel_ridge_score = kernel_ridge_model.score(X_test, y_test)
    kernel_ridge_list = ['kernel_ridge',
                         kernel_ridge_score, kernel_ridge_model.best_params_]

    # print(kernel_ridge_series)

    # SVR
    kernel_svr_pipeline = Pipeline([
        ('scale', StandardScaler()),
        ('svr', SVR(kernel='rbf'))
    ])

    kernel_svr_model = GridSearchCV(estimator=kernel_svr_pipeline, param_grid=[{
        'svr__C': [0.001, 0.01, 0.1, 1, 5, 10],
        'svr__epsilon': [0.001, 0.01, 0.1, 0.2, 0.5, 1],
        'svr__gamma': [0.001, 0.01, 0.03, 0.05, 0.1]
    }],
        n_jobs=-1,
        scoring='neg_mean_absolute_error',
        cv=5)

    kernel_svr_model.fit(X_train, y_train)
    kernel_svr_score = kernel_svr_model.score(X_test, y_test)
    kernel_svr_list = ['kernel_svr', kernel_svr_score,
                       kernel_svr_model.best_params_]

    # print(kernel_svr_series)

    # SVC
    kernel_svc_pipeline = Pipeline([
        ('scale', StandardScaler()),
        ('svc', SVC(kernel='rbf'))
    ])

    kernel_svc_model = GridSearchCV(kernel_svc_pipeline, [{
        'svc__C': [0.001, 0.01, 0.1, 1, 5, 10],
        'svc__gamma': [0.001, 0.01, 0.03, 0.05, 0.1],
    }],
        n_jobs=-1,
        scoring='neg_mean_absolute_error',
        cv=5)

    kernel_svc_model.fit(X_train, y_train)
    kernel_svc_score = kernel_svc_model.score(X_test, y_test)
    kernel_svc_list = ['kernel_svc', kernel_svc_score,
                       kernel_svc_model.best_params_]

    # print(kernel_svc_series)

    # combine model parameters and scoring into dataframe
    performance_df = pd.DataFrame(
        [kernel_ridge_list, kernel_svr_list, kernel_svc_list],
        columns=['Method', 'Test Score', 'Best Params']
    )

    print(performance_df)


# execute steps
kernel_methods()

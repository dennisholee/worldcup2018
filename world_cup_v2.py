# Team ranking during World Cup 2018 : https://www.bbc.com/sport/football/30730600

import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv("matches.txt")
X_orig = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 2].values

"""
Categorize data i.e. convert categories to numerical values
"""
ct_X = ColumnTransformer(
    transformers=[
        ("onehot", OneHotEncoder(sparse=False), [0])
    ],
    remainder="passthrough" )
X = ct_X.fit_transform(X_orig)

"""
Splitting the dataset into training and test set
"""
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

"""
Fitting simple linear regression to the training set
"""
regressor = LinearRegression()
regressor.fit(X=X_train, y=Y_train)

"""
Predicting the Test set result
"""
france_pred = ct_X.transform(np.array([["France", 20]])).astype(np.float)
croatia_pred = ct_X.transform(np.array([["Croatia", 7]])).astype(np.float)

predict_france = regressor.predict(france_pred)
predict_croatia = regressor.predict(croatia_pred)

print("%s: %s" % ("france", predict_france))
print("%s: %s" % ("croatia", predict_croatia))

# TODO: Adjust the versus team's ranking to put higher ranked with heavier weighting
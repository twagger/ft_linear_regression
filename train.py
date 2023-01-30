"""Module to train a linear regression model"""
# -----------------------------------------------------------------------------
# Module imports
# -----------------------------------------------------------------------------
# system
import os
import sys
# read csv files
import csv
# nd arrays
import numpy as np
# dataframes
import pandas as pd
# for plot
import matplotlib.pyplot as plt
# user modules
sys.path.insert(1, os.path.join(os.path.dirname(__file__), 'classes'))
sys.path.insert(1, os.path.join(os.path.dirname(__file__), 'utils'))
from MyLinearRegression import MyLinearRegression
from z_score import z_score


# -----------------------------------------------------------------------------
# Program : Train
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Argument management
    # -------------------------------------------------------------------------
    # argument management : no argument will be taken in account (display
    # usage if an argument is provided)
    if len(sys.argv) > 1:
        print("predict: invalid argument\n"
              "Usage: python train.py (with no argument)", file=sys.stderr)
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Linear regression
    # -------------------------------------------------------------------------
    # 0. parameters for training
    # alpha = 1e-10 # learning rate < Version with no standardization
    alpha = 1e-1 # learning rate
    max_iter = 100000 # max_iter

    # 1. open and load the training dataset
    try:
        df = pd.read_csv("./data.csv")
    except:
        print("Error when trying to read dataset", file=sys.stderr)
        sys.exit(1)

    # check that the expected columns are here and check their type
    if not set(['km', 'price']).issubset(df.columns):
        print("Missing columns in 'data.csv' file", file=sys.stderr)
        sys.exit(1)
    if not (df.km.dtype == float or df.km.dtype == int) \
            or not (df.price.dtype == float or df.price.dtype == int):
        print("Wrong column type in 'data.csv' file", file=sys.stderr)
        sys.exit(1)

    # set x and y
    x = np.array(df['km']).reshape((-1, 1))
    y = np.array(df['price']).reshape((-1, 1))

    # 2. create a model from MyLinearRegression class. The thetas are
    # initialized with specific values to ease the training
    # -------------------------------------------------------------------------
    # IF IN THE CORRECTION OF THIS PROJECT PASSING ANYTHING ELSE THAN THETAS 
    # IN MODELS.CSV, THEN I CAN USE THE FOLLOWING LINE AND DON'T USE STANDARDI-
    # ZATION
    # -------------------------------------------------------------------------
    # MyLR = MyLinearRegression(np.array([9000, -0.05]).reshape(-1, 1),
    #                           alpha=alpha, max_iter=max_iter)
    MyLR = MyLinearRegression(np.random.rand(2, 1).reshape(-1, 1), alpha=alpha,
                              max_iter=max_iter)

    # 3.standardize the data to ease gradient descent
    x_norm, mean, std = z_score(x)

    # 4. train the model (fit) with all the dataset)
    # MyLR.fit_(x, y) < Version with no standartization
    MyLR.fit_(x_norm, y)

    # 5. save the updated thetas in 'model.csv' file
    try:
        with open('models.csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerow(["thetas", "mean", "std"])

            thetas_str = ','.join([f'{theta[0]}' for theta in MyLR.thetas])
            writer.writerow([thetas_str, mean, std])
    except:
        print("Error when trying to read 'model.csv'", file=sys.stderr)
        sys.exit(1)

    # 6. display a message for the user
    print(f'The model as been trained!')

    # 7. plot the training data repartition and the prediction line
    # predicted values
    # y_hat = MyLR.predict_(x) < Version with no standardization
    y_hat = MyLR.predict_(x_norm)
    # plot
    plt.figure()
    plt.scatter(x, y, marker='o')
    plt.plot(x, y_hat, color='red')
    plt.show()

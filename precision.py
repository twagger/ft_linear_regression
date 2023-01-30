"""This module computes the loss for a model"""
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
# dataframe
import pandas as pd
# user modules
sys.path.insert(1, os.path.join(os.path.dirname(__file__), 'classes'))
from MyLinearRegression import MyLinearRegression


# -----------------------------------------------------------------------------
# Program : Computes and display the loss of the model
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Argument management
    # -------------------------------------------------------------------------
    # argument management : no argument will be taken in account (display
    # usage if an argument is provided)
    if len(sys.argv) > 1:
        print("predict: invalid argument\n"
              "Usage: python predict.py (with no argument)", file=sys.stderr)
        sys.exit()

    # -------------------------------------------------------------------------
    # Training dataset
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Linear regression
    # -------------------------------------------------------------------------
    # 1. Check if there is a 'model.csv' file in the current folder with a
    # theta configuration
    thetas = np.array([0.0, 0.0])
    try:
        with open('models.csv', 'r') as file:
            reader = csv.DictReader(file) # DictRader will skip the header row
            # I just have one row max in this project
            for row in reader:
                thetas = np.array([float(theta) for theta 
                                in row['thetas'].split(',')]).reshape(-1, 1)
                # get mean and standard deviation used in standardization
                mean = float(row['mean'])
                std = float(row['std'])
            # check that thetas are valid
            if np.isnan(thetas).any() is True:
                print('Something when wrong during the training, '
                      'the parameters are invalid.', file=sys.stderr)
                sys.exit(1)

    except FileNotFoundError:
        pass
    except SystemExit:
        sys.exit(1)
    except:
        print("Error when trying to read 'model.csv'", file=sys.stderr)
        sys.exit(1)

    # 2. create a model from MyLinearRegression class. The thetas are
    # initialized at 0.0 if no model.csv is found in the folder
    MyLR = MyLinearRegression(thetas.reshape(-1, 1))

    # 3. display the precision of the model for the training dataset
    x_norm = (x - mean) / std
    y_hat = MyLR.predict_(x_norm)

    # MSE
    print(f'{"":-<80}')
    print(f'MSE score : {MyLR.mse_(y, y_hat)}\n{MyLR.mse_.__doc__}')

    # RMSE
    print(f'{"":-<80}')
    print(f'RMSE score : {MyLR.rmse_(y, y_hat)}\n{MyLR.rmse_.__doc__}')

    # MAE
    print(f'{"":-<80}')
    print(f'MAE score : {MyLR.mae_(y, y_hat)}\n{MyLR.mae_.__doc__}')

    # R2SCORE
    print(f'{"":-<80}')
    print(f'R2 score : {MyLR.r2score_(y, y_hat)}\n{MyLR.r2score_.__doc__}')

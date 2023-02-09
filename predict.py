"""
Module to use a model to predict the price of a car according to its mileage
"""
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
from MyLinearRegression import MyLinearRegression


# -----------------------------------------------------------------------------
# Program : Predict
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
    # Prompt the user for mileage
    # -------------------------------------------------------------------------
    while 1:
        try:
            mileage = input("Enter a mileage\n>> ")
        except EOFError:
            print('\n')
            continue
        except KeyboardInterrupt:
            print('\nGoodbye !')
            sys.exit(0)

        # check if mileage can be converted to a proper numeric type
        try:
            mileage = float(mileage)
            if (mileage < 0):
                print('Wrong value, please enter a positive number\n')
                continue
            break
        except ValueError:
            print('Wrong value, please enter a number\n')

    # -------------------------------------------------------------------------
    # Linear regression
    # -------------------------------------------------------------------------
    # 1. Update model thetas, mean, std from file if it exists
    thetas = np.array([0.0, 0.0])
    mean = 0
    std = 1
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

    # 1. create a model from MyLinearRegression class. The thetas are
    # initialized at 0.0 if no model.csv is found in the folder
    MyLR = MyLinearRegression(thetas.reshape(-1, 1))

    # 2. predict one value (the prompted one) with standardized mileage
    mileage_norm = (mileage - mean) / std
    predicted_price = MyLR.predict_(np.array([[mileage_norm]]))

    # -------------------------------------------------------------------------
    # Display prediction
    # -------------------------------------------------------------------------
    # 3. display the predicted value to the user
    print(f'For a mileage of {mileage},'
          f' the predicted price is {predicted_price[0][0]}')

    # -------------------------------------------------------------------------
    # Plot prediction aside with the training dataset data
    # -------------------------------------------------------------------------
    # open and load the training dataset
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

    # normalize x data
    x_norm = (x - mean) / std
    y_hat = MyLR.predict_(x_norm)

    # plot
    plt.figure()
    plt.scatter(x, y, marker='o', label='training data')
    plt.scatter(mileage, predicted_price[0][0], marker='o',
                label='predicted data')
    plt.plot(x, y_hat, color='red', label='prediction function')
    plt.legend()
    plt.show()

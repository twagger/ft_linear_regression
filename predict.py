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
            break
        except ValueError:
            print('Wrong value, please enter a number\n')

    # -------------------------------------------------------------------------
    # Linear regression
    # -------------------------------------------------------------------------
    # 1. Check if there is a 'model.csv' file in the current folder with a
    # theta configuration
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

    # 3. display the predicted value to the user
    print(f'For a mileage of {mileage},'
          f' the predicted price is {predicted_price[0][0]}')

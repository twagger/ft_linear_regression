"""
My Linear Regression class
"""
# -----------------------------------------------------------------------------
# Module imports
# -----------------------------------------------------------------------------
# system
import os
import sys
from math import sqrt
# nd arrays
import numpy as np
# for decorators
import inspect
from functools import wraps
# for progress bar
from tqdm import tqdm
# for plot
import matplotlib.pyplot as plt
# user modules
sys.path.insert(1, os.path.join(os.path.dirname(__file__), '..', 'utils'))
from shape_validator import shape_validator
from type_validator import type_validator


# -----------------------------------------------------------------------------
# MyLinearRegression class
# -----------------------------------------------------------------------------
class MyLinearRegression():
    """
    Description:
    My personnal linear regression class to fit like a boss.
    """
    @type_validator
    @shape_validator({'thetas': ('n', 1)})
    def __init__(self, thetas: np.ndarray, alpha: float = 0.001,
                 max_iter: int = 1000):
        """Constructor"""
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = np.array(thetas).reshape((-1, 1))

    @type_validator
    @shape_validator({'x': ('m', 'n')})
    def predict_(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the vector of prediction y_hat from two non-empty numpy.array.
        """
        try:
            m, _ = x.shape
            x_prime = np.c_[np.ones((m, 1)), x]
            return x_prime.dot(self.thetas)
        except:
            return None

    @type_validator
    @shape_validator({'y': ('m', 1), 'y_hat': ('m', 1)})
    def loss_elem_(self, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        """
        Calculates all the elements (y_pred - y)^2 of the loss function.
        """
        try:
            return (y_hat - y) ** 2
        except:
            return None

    @type_validator
    @shape_validator({'y': ('m', 1), 'y_hat': ('m', 1)})
    def loss_(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        """
        Computes the half mean squared error of two non-empty numpy.array,
        without any for loop.
        The two arrays must have the same dimensions.
        """
        try:
            m, _ = y.shape
            loss_vector = self.loss_elem_(y, y_hat)
            return float((np.sum(loss_vector) / (2 * m)))
        except:
            return None

    @type_validator
    @shape_validator({'x': ('m', 'n'), 'y': ('m', 1)})
    def gradient_(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Computes the regularized linear gradient of three non-empty
        numpy.ndarray. The three arrays must have compatible shapes.
        """
        try:
            m, _ = x.shape
            x_prime = np.c_[np.ones((m, 1)), x]
            y_hat = x_prime.dot(self.thetas)
            return x_prime.T.dot(y_hat - y) / m
        except:
            return None

    @type_validator
    @shape_validator({'x': ('m', 'n'), 'y': ('m', 1)})
    def fit_(self, x: np.ndarray, y: np.ndarray,
             plot: bool = False) -> np.ndarray:
        """
        Description:
        Fits the model to the training dataset contained in x and y.
        """
        try:
            # calculation of the gradient vector
            # 1. X to X'
            m, _ = x.shape
            x_prime = np.c_[np.ones((m, 1)), x]

            if plot is True:
                # gradient update in loop with plot : learning curve and
                # current prediction function

                # init subplots
                fig, axs = plt.subplots(1, 2)
                # initialize the line plots
                learn_c, = axs[0].plot([], [], 'r-')
                current_pred, = axs[1].plot([], [], 'r-')

                # draw training data on prediction graph
                axs[1].set_xlabel = 'km'
                axs[1].set_ylabel = 'price'
                axs[1].scatter(x, y, label='training data')

                # 2. loop
                for it, _ in enumerate(tqdm(range(self.max_iter))):
                    # calculate the current predictions
                    y_hat = self.predict_(x)
                    # calculate current loss
                    loss = self.loss_(y, y_hat)
                    # Update the lines data
                    learn_c.set_data(np.append(learn_c.get_xdata(), it),
                                     np.append(learn_c.get_ydata(), loss))
                    current_pred.set_data(x, y_hat)

                    # Update the axis limits
                    for i in range(2):
                        axs[i].relim()
                        axs[i].autoscale_view()

                    # Redraw the figure
                    fig.canvas.draw()

                    # Pause to make animation visible
                    plt.pause(0.001)

                    # calculate the grandient for current thetas
                    gradient = self.gradient_(x, y)
                    # 4. calculate and assign the new thetas
                    self.thetas -= self.alpha * gradient

            else:
                # gradient update in loop with tqdm
                for _ in tqdm(range(self.max_iter)):
                    # calculate the grandient for current thetas
                    gradient = self.gradient_(x, y)
                    # calculate and assign the new thetas
                    self.thetas -= self.alpha * gradient
            return self.thetas
        except ValueError as exc:
            print(exc)
            return None

    @type_validator
    @shape_validator({'y': ('m', 1), 'y_hat': ('m', 1)})
    def mse_(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        """
        Mean Squared Error (MSE): It is the average of the squared differences
        between the predicted and actual values. The MSE is a measure of the
        quality of an estimator, it is always non-negative, and values closer
        to zero are better.
        """
        try:
            m, _ = y.shape
            loss_vector = self.loss_elem_(y, y_hat)
            return float((np.sum(loss_vector) / m))
        except:
            return None

    @type_validator
    @shape_validator({'y': ('m', 1), 'y_hat': ('m', 1)})
    def rmse_(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        """
        Root Mean Squared Error (RMSE): It is the square root of the mean
        squared error, which is more interpretable since it has the same units
        as the target variable. It is used to measure the difference between
        the predicted values and the actual values.
        """
        try:
            return sqrt(self.mse_(y, y_hat))
        except:
            return None

    @type_validator
    @shape_validator({'y': ('m', 1), 'y_hat': ('m', 1)})
    def mae_(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        """
        Mean Absolute Error (MAE): It is the average of the absolute
        differences between the predicted and actual values. It is always
        non-negative, and values closer to zero are better.
        """
        try:
            return float(np.sum(np.absolute(y_hat - y)) / len(y))
        except:
            return None

    @type_validator
    @shape_validator({'y': ('m', 1), 'y_hat': ('m', 1)})
    def r2score_(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        """
        R-squared score (R2 score): It is a statistical measure that represents
        the proportion of the variance in the dependent variable that is
        predictable from the independent variable.
        It ranges from 0 to 1, where 1 means that the model perfectly fits the
        data and 0 means that the model has no explanatory power.
        """
        try:
            mean_y = np.mean(y)
            sse = float((((y_hat - y).T.dot(y_hat - y)))[0][0])
            sst = float((((y - mean_y).T.dot(y - mean_y)))[0][0])
            return 1 - (sse / sst)
        except:
            return None

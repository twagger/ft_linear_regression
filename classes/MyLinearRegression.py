"""
My Linear Regression class
"""
# -----------------------------------------------------------------------------
# Module imports
# -----------------------------------------------------------------------------
# system
import os
import sys
# nd arrays
import numpy as np
# for decorators
import inspect
from functools import wraps
# for progress bar
from tqdm import tqdm
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
    @shape_validator({'y': ('m', 1), 'y_hat': ('m', 1)})
    def mse_(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        """
        Computes the mean squared error of two non-empty numpy.array,
        without any for loop.
        The two arrays must have the same dimensions.
        """
        try:
            m, _ = y.shape
            loss_vector = self.loss_elem_(y, y_hat)
            return float((np.sum(loss_vector) / m))
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
    def fit_(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Description:
        Fits the model to the training dataset contained in x and y.
        """
        try:
            # calculation of the gradient vector
            # 1. X to X'
            m, _ = x.shape
            x_prime = np.c_[np.ones((m, 1)), x]
            # 2. loop
            for _ in tqdm(range(self.max_iter)):
                # 3. calculate the grandient for current thetas
                gradient = self.gradient_(x, y)
                # 4. calculate and assign the new thetas
                self.thetas -= self.alpha * gradient
            return self.thetas
        except:
            return None

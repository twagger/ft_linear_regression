"""zscore normalization"""
# -----------------------------------------------------------------------------
# Module imports
# -----------------------------------------------------------------------------
# nd arrays
import numpy as np
# user modules
from shape_validator import shape_validator
from type_validator import type_validator


# -----------------------------------------------------------------------------
# zscore normalization
# -----------------------------------------------------------------------------
@type_validator
@shape_validator({'x': ('m', 1)})
def z_score(x: np.ndarray) -> tuple:
    """
    Computes the normalized version of a non-empty numpy.ndarray using the
        z-score standardization.
    It returns in that order :
        - the normalized vector
        - the mean used for standardization
        - the standard deviation used for standardization
    """
    try:
        mean_ = np.mean(x, axis=0)
        std_ =  np.std(x, axis=0)
        x_prime = (x - mean_) / std_
        return x_prime, mean_, std_
    except:
        return None

import numpy as np
from scipy.linalg import pinvh

import utils.constants as constants
from utils.error import Error


class IterativeTrimmedRegression:
    """Use iterative trimmed regression to solve robust linear regression.
    """
    def __init__(self, o=constants.O):
        self.error_name = constants.GENERALIZATION_ERROR_NAME
        self.o = o

    def get_signal_estimation(self, y, H, x, num_remaining_indices):
        """Run one iteration.
        
        @param y - the observations/response.
        @param H - the design matrix.
        @param x - estimation of signal, \hat{x}.
        @param num_remaining_indices - number of remaining coordinates.
        """
        # To throw 2*o largest coordinates.
        # set_remaining_indices - Set of remaining indices.
        #set_remaining_indices = np.argpartition(y - np.dot(H, x), num_remaining_indices)[:num_remaining_indices]
        #set_remaining_indices = np.sort(set_remaining_indices)
        temp = np.argsort(np.abs(y - np.dot(H, x)))
        set_remaining_indices = temp[:num_remaining_indices]
        H_remaining = np.array(H)[set_remaining_indices]
        y_remaining = np.array(y)[set_remaining_indices]
        x = np.dot(pinvh(np.dot(np.transpose(H_remaining), H_remaining)),
                   np.dot(np.transpose(H_remaining), y_remaining))
        return x

    def get_errors(self,
                   x_original,
                   y,
                   H,
                   SIGMA_half,
                   num_iter,
                   errors_needed=False):
        """Run several iterations.
        
        @param x_original - in order to compute error,  we need the original/true signal.
        @param y - the observations/response.
        @param H - the design matrix.
        @param SIGMA_half - half of the covariance of design matrix.
        @param num_iter - the number of iterations.
        @param errors_needed - If True, return generalization errors of each iteration.
        @return x - estimation \hat{x};
                gener_errors - generalization errors of estimation in each iteration.
        """
        x = np.zeros((constants.P))  # initial value of estimation
        # record generalization errors of estimation in each iteration
        errors = [0] * num_iter
        for i in range(num_iter):
            x = self.get_signal_estimation(y, H, x, constants.N - 3 * self.o)
            if errors_needed:
                errors[i] = Error().get_error(x_original, x, self.error_name,
                                              SIGMA_half, y, H)
        if errors_needed:
            return (x, errors)
        return x

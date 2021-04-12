import numpy as np

import utils.constants as constants


class GenerateData:
    """Generate the model.
    """
    def __init__(self, kappa=1., x=constants.X):
        """Initialize the model.
        
        @param kappa - condition number of design matrix.
        @param x - true signal.
        """
        self.mu = constants.MU
        self.n, self.p, self.o = constants.N, constants.P, constants.O
        self.x = x
        # half of design covariance
        temp = np.ones((constants.P))
        temp[constants.P // 2:] = kappa
        temp = np.random.permutation(temp)
        self.SIGMA_half = np.diag(temp)
        self.sigma = constants.SIGMA_NUMBER # variance of noise

    def generate_data(self):
        """Generate response and design matrix.
        
        @return response and design matrix.
        """
        H = np.dot(np.random.randn(self.n, self.p), self.SIGMA_half)
        # y = H * x + noise
        y = np.dot(H, self.x) + np.random.normal(
            self.mu, self.sigma, size=(self.n))
        y[:self.o] = 0
        y = np.random.permutation(y)
        return (y, H, self.SIGMA_half)

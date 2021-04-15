import numpy as np

import utils.constants as constants


class GenerateData:
    """Generate the model.
    """
    def __init__(self, kappa=1., o=constants.O):
        """Initialize the model.
        
        @param kappa - condition number of design matrix.
        @param x - true signal.
        """
        self.mu = constants.MU
        self.n, self.p, self.o = constants.N, constants.P, o
        self.x = constants.X
        # half of design covariance
        temp = np.ones((constants.P))
        temp[constants.P // 2:] = kappa
        temp = np.random.permutation(temp)
        self.SIGMA_half = np.diag(temp)
        self.sigma = constants.SIGMA_NUMBER  # variance of noise

    def generate_data(self):
        """Generate response and design matrix.
        
        @return response and design matrix.
        """
        H = np.dot(np.random.randn(self.n, self.p), self.SIGMA_half)
        # y = H * x + noise
        y = np.dot(H, self.x) + np.random.normal(
            self.mu, self.sigma, size=(self.n))
        y[:self.o] = 0
        indices = np.arange(len(H))
        np.random.shuffle(indices)
        H = np.asarray(H)[indices]
        y = np.asarray(y)[indices]
        return (y, H, self.SIGMA_half)

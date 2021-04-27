import numpy as np

import utils.constants as constants


class GenerateData:
    """Generate the model.
    """
    def __init__(self,
                 is_design_corruption=True,
                 kappa=constants.KAPPA,
                 o=constants.O,
                 p=constants.P):
        """Initialize the model.
        
        @param kappa - condition number of design matrix.
        @param o.
        @param p - number of features.
        """
        self.is_design_corruption = is_design_corruption
        self.mu = constants.MU
        self.sigma = constants.SIGMA_NUMBER  # variance of noise
        self.n, self.p, self.o = constants.N, p, o
        self.x = constants.X_VALUE * np.ones((self.p))
        # half of design covariance
        temp = np.ones((self.p))
        temp[self.p // 2:] = kappa
        temp = np.random.permutation(temp)
        self.SIGMA_half = np.diag(temp)

    def generate_data(self):
        """Generate response and design matrix.
        
        @return response and design matrix.
        """
        H = np.dot(np.random.randn(self.n, self.p), self.SIGMA_half)
        # y = H * x + noise
        y = np.dot(H, self.x) + np.random.normal(
            self.mu, self.sigma, size=(self.n))
        y[:self.o] = 0 # corruption
        if self.is_design_corruption:
            # corruption on the design
            vect = self.x / np.linalg.norm(self.x, 2) * np.sqrt(self.p)
            H[:self.o] = vect.T
        indices = np.arange(len(H))
        np.random.shuffle(indices)
        H = np.asarray(H)[indices]
        y = np.asarray(y)[indices]
        return (self.x, y, H, self.SIGMA_half)

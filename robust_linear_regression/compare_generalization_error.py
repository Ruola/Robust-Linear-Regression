import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10})
import numpy as np
from scipy.linalg import pinvh

from algorithms.iterative_trimmed_regression import IterativeTrimmedRegression
import utils.constants as constants
from utils.error import Error
from utils.generate_data import GenerateData


class CompareGeneralizationError:
    """Simulations on iterative trimmed regression.
    """
    def __init__(self, kappa=1.):
        """Initialize.

        @param kappa - condition number of design matrix.
        """
        self.kappa = kappa
        self.steps = constants.STEPS
        self.num_iter = constants.N_ITERATION
        self.x_original = constants.X

    def compare_convergence_rate(self):
        """Get the change of generalization error with respect to #iterations.
        """
        errors_matrices = np.zeros((constants.STEPS, constants.N_ITERATION))
        for i in range(constants.STEPS):  # do several experiments
            y, H, SIGMA_half = GenerateData(self.kappa,
                                            self.x_original).generate_data()
            x, error = IterativeTrimmedRegression().get_errors(
                self.x_original, y, H, SIGMA_half, self.num_iter, True)
            errors_matrices[i] = error
        plt.plot(np.mean(errors_matrices, axis=0),
                 label="IterativeTrimmedRegression")
        baseline_estimate = np.dot(pinvh(np.dot(np.transpose(H), H)),
                                   np.dot(np.transpose(H), y))
        baseline_error = Error().get_error(self.x_original, baseline_estimate,
                                           constants.GENERALIZATION_ERROR_NAME,
                                           SIGMA_half)
        plt.plot([baseline_error] * self.num_iter, label="Baseline")
        plt.xlabel("#iterations")
        plt.ylabel("generalization error")
        plt.title("Convergence rate as kappa " + str(int(self.kappa)))
        plt.legend()
        plt.savefig(
            os.path.dirname(os.path.abspath(__file__)) +
            "/figures/convergence rate as kappa " + str(int(self.kappa)) + ".pdf")
        plt.clf()


if __name__ == "__main__":
    kappa = 1.
    CompareGeneralizationError(kappa).compare_convergence_rate()

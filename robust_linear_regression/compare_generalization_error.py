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
    def __init__(self, is_design_corruption=True, kappa=1.):
        """Initialize.

        @param kappa - condition number of design matrix.
        """
        self.is_design_corruption = is_design_corruption
        self.kappa = kappa
        self.steps = constants.STEPS
        self.num_iter = constants.N_ITERATION
        if self.is_design_corruption:
            self.folder_name = "/figures/design matrix corruption"
        else:
            self.folder_name = "/figures/no design matrix corruption"

    def compare_convergence_rate(self):
        """Get the change of generalization error w.r.t. #iterations.
        """
        errors_matrices = np.zeros((constants.STEPS, constants.N_ITERATION))
        for i in range(constants.STEPS):  # do several experiments
            x_original, y, H, SIGMA_half = GenerateData(
                self.is_design_corruption, self.kappa).generate_data()
            x, error = IterativeTrimmedRegression().get_errors(
                x_original, y, H, SIGMA_half, self.num_iter, True)
            errors_matrices[i] = error
        plt.plot(np.mean(errors_matrices, axis=0),
                 label="IterativeTrimmedRegression")
        baseline_estimate = np.dot(pinvh(np.dot(np.transpose(H), H)),
                                   np.dot(np.transpose(H), y))
        baseline_error = Error().get_error(x_original, baseline_estimate,
                                           constants.GENERALIZATION_ERROR_NAME,
                                           SIGMA_half)
        plt.plot([baseline_error] * self.num_iter, label="Baseline")
        plt.xlabel("#iterations")
        plt.ylabel("generalization error")
        plt.title("Convergence rate as kappa=" + str(int(self.kappa)))
        plt.legend()
        plt.savefig(
            os.path.dirname(os.path.abspath(__file__)) + self.folder_name +
            "/convergence rate as kappa " + str(int(self.kappa)) + ".pdf")
        plt.clf()

    def change_fraction_of_outlier(self):
        """Change of generalization error w.r.t. fraction of outlier.
        """
        o_range = list(range(100, 301, 10))
        errors_matrices = np.zeros((constants.STEPS, len(o_range)))
        for i in range(constants.STEPS):  # do several experiments
            for j in range(len(o_range)):
                o = o_range[j]
                x_original, y, H, SIGMA_half = GenerateData(
                    self.is_design_corruption, self.kappa, o).generate_data()
                x, error = IterativeTrimmedRegression(o).get_errors(
                    x_original, y, H, SIGMA_half, self.num_iter, True)
                errors_matrices[i][j] += error[-1]
        fraction_of_outlier = np.array(o_range) / constants.N
        plt.plot(fraction_of_outlier,
                 np.mean(errors_matrices, axis=0),
                 label="IterativeTrimmedRegression")
        plt.xlabel("fraction of outlier")
        plt.ylabel("generalization error")
        plt.title("Change fraction of outlier as kappa=" +
                  str(int(self.kappa)))
        plt.legend()
        plt.savefig(
            os.path.dirname(os.path.abspath(__file__)) + self.folder_name +
            "/change fraction of outlier as kappa " + str(int(self.kappa)) +
            ".pdf")
        plt.clf()

    def change_condition_number(self):
        """Change of generalization error w.r.t. the condition number.
        """
        kappa_range = list(range(1, 50, 10))
        errors_matrices = np.zeros((constants.STEPS, len(kappa_range)))
        for i in range(constants.STEPS):  # do several experiments
            for j in range(len(kappa_range)):
                kappa = kappa_range[j]
                x_original, y, H, SIGMA_half = GenerateData(
                    self.is_design_corruption, kappa).generate_data()
                x, error = IterativeTrimmedRegression().get_errors(
                    x_original, y, H, SIGMA_half, self.num_iter, True)
                errors_matrices[i][j] += error[-1]
        fraction_of_outlier = np.array(kappa_range) / constants.N
        plt.plot(fraction_of_outlier,
                 np.mean(errors_matrices, axis=0),
                 label="IterativeTrimmedRegression")
        plt.xlabel("kappa")
        plt.ylabel("generalization error")
        plt.title("Change condition number")
        plt.legend()
        plt.savefig(
            os.path.dirname(os.path.abspath(__file__)) + self.folder_name +
            "/change condition number.pdf")
        plt.clf()

    def change_number_of_features(self):
        p_range = list(range(10, 200, 20))
        errors_matrices = np.zeros((constants.STEPS, len(p_range)))
        for i in range(constants.STEPS):  # do several experiments
            for j in range(len(p_range)):
                p = p_range[j]
                x_original, y, H, SIGMA_half = GenerateData(
                    self.is_design_corruption, self.kappa, constants.O,
                    p).generate_data()
                _, error = IterativeTrimmedRegression().get_errors(
                    x_original, y, H, SIGMA_half, self.num_iter, True)
                errors_matrices[i][j] += error[-1]
        plt.plot(np.mean(errors_matrices, axis=0),
                 label="IterativeTrimmedRegression")
        plt.xlabel("number of features")
        plt.ylabel("generalization error")
        plt.title("Change the number of features + kappa=" +
                  str(int(self.kappa)))
        plt.legend()
        plt.savefig(
            os.path.dirname(os.path.abspath(__file__)) + self.folder_name +
            "/change number of features kappa" + str(int(self.kappa)) + ".pdf")
        plt.clf()


if __name__ == "__main__":
    """Run simulations.
    """
    is_design_corruption = False
    kappa = 1.  # condition number

    CompareGeneralizationError(is_design_corruption,
                               kappa).compare_convergence_rate()
    CompareGeneralizationError(is_design_corruption,
                               kappa).change_fraction_of_outlier()
    CompareGeneralizationError(is_design_corruption).change_condition_number()
    CompareGeneralizationError(is_design_corruption,
                               kappa).change_number_of_features()

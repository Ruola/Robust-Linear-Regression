import numpy as np
"""Store some constant variable.

@param STEPS - number of experiments.
@param N_ITERATION - max number of iterations.
@param N - number of responses for obeservations.
@param P - number of features.
@param O - 
@param SIGMA_COVAR_MATRIX_HALF - half of covariance of design matrix.
@param MU - expectation of noise.
@param SIGMA_NUMBER - variance of noise.
@param X_VALUE - value of elements of true signal.
@param X - true signal.
@param KAPPA - condition number of design.
"""
STEPS = 200
N_ITERATION = 100
N, P, O = 1000, 10, 100
# noise
MU = 0
SIGMA_NUMBER = .1
# signal
X_VALUE = 10.
X = X_VALUE * np.ones((P))
KAPPA = 1.

PREDICTION_ERROR_NAME = "prediction error"
GENERALIZATION_ERROR_NAME = "generalization error"
EXACT_RECOVERY_NAME = "exact recovery"

## Robust linear regression

### How to use
- `git clone https://github.com/Ruola/Robust-Linear-Regression.git`
- `cd Robust-Linear-Regression`

### Algorithms

- Iterative Trimmed Regression.

### Simulations

- Get the change of generalization error with respect to #iterations in Iterative Trimmed Regression.
  - `python3 robust_linear_regression/compare_generalization_error.py`.
  - Results are in `/figures/`.

### Unit tests

- Unit tests are in the directory `/tests/`.
- Run `python3 -m unittest filename.py`, for example `python3 -m unittest tests/algorithms/test_iterative_trimmed_regression.py`.


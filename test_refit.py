"""Check aggregate fitting/evaluation against a direct small-matrix calculation."""
import unittest
import numpy as np
from refit_operators import solve, squared_errors


class AggregateFitTests(unittest.TestCase):
    def test_ridge_and_heldout_error(self):
        rng = np.random.default_rng(12)
        x = np.c_[rng.normal(size=(40, 4)), np.ones(40)]
        y = x @ rng.normal(size=(5, 3)) + rng.normal(size=(40, 3))
        lam = 2.3
        # Independent augmented least squares; intercept stays unpenalized.
        penalty = np.sqrt(lam) * np.eye(5)[:4]
        direct = np.linalg.lstsq(np.r_[x, penalty], np.r_[y, np.zeros((4, 3))], rcond=None)[0]
        recovered = solve(x.T @ x, x.T @ y, lam)
        np.testing.assert_allclose(recovered, direct, atol=1e-12)
        tx = np.c_[rng.normal(size=(17, 4)), np.ones(17)]
        ty = rng.normal(size=(17, 3))
        stats = {'xx': tx.T @ tx, 'xy': tx.T @ ty, 'yy_diag': np.sum(ty**2, axis=0)}
        np.testing.assert_allclose(squared_errors(recovered, stats), np.sum((ty-tx@direct)**2, axis=0), atol=1e-11)


if __name__ == '__main__':
    unittest.main()

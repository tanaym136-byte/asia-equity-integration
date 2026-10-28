import numpy as np
import pytest

from asiaint import dcc


def test_filter_gives_valid_correlation_matrices():
    rng = np.random.default_rng(1)
    z = rng.standard_normal((500, 3))
    _, R = dcc.dcc_filter(z, 0.05, 0.9)
    assert np.allclose(np.diagonal(R, axis1=1, axis2=2), 1.0)
    assert np.allclose(R, np.transpose(R, (0, 2, 1)))
    assert np.all(np.linalg.eigvalsh(R) > 0)
    assert np.all(np.abs(R) <= 1 + 1e-12)


def test_first_correlation_uses_no_current_information():
    rng = np.random.default_rng(2)
    z = rng.standard_normal((50, 2))
    z2 = z.copy()
    z2[10] *= 50  # a shock on day 10 can only move R from day 11 onwards
    _, R1 = dcc.dcc_filter(z, 0.05, 0.9, qbar=np.eye(2))
    _, R2 = dcc.dcc_filter(z2, 0.05, 0.9, qbar=np.eye(2))
    assert np.allclose(R1[:11], R2[:11])
    assert not np.allclose(R1[11], R2[11])


def test_likelihood_rejects_nonstationary_parameters():
    z = np.random.default_rng(3).standard_normal((100, 2))
    qbar = np.cov(z, rowvar=False)
    assert dcc.dcc_negloglik([0.1, 0.95], z, qbar) == 1e10
    assert dcc.dcc_negloglik([-0.01, 0.9], z, qbar) == 1e10


def test_recovers_parameters_from_simulated_data():
    corr = np.array([[1.0, 0.5, 0.4], [0.5, 1.0, 0.6], [0.4, 0.6, 1.0]])
    z = dcc.simulate_dcc(4000, a=0.04, b=0.93, corr=corr, seed=7)
    fit = dcc.fit_dcc(z)
    assert fit.a == pytest.approx(0.04, abs=0.02)
    assert fit.b == pytest.approx(0.93, abs=0.04)
    assert fit.a + fit.b < 1


def test_constant_correlation_data_gives_small_a():
    corr = np.array([[1.0, 0.5], [0.5, 1.0]])
    z = np.random.default_rng(11).multivariate_normal([0, 0], corr, size=3000)
    fit = dcc.fit_dcc(z)
    assert fit.a < 0.02
    assert np.mean(fit.R[:, 0, 1]) == pytest.approx(0.5, abs=0.05)

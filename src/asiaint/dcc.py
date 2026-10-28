"""Two-step DCC-GARCH (Engle, 2002).

Step 1: fit a univariate GJR-GARCH(1,1) with Student-t errors to each return series and keep the
        standardised residuals z_t = e_t / sigma_t.
Step 2: fit DCC(1,1) to the standardised residuals:
            Q_t = (1 - a - b) * Qbar + a * z_{t-1} z_{t-1}' + b * Q_{t-1}
            R_t = diag(Q_t)^(-1/2) Q_t diag(Q_t)^(-1/2)
        with Qbar the sample covariance of z (correlation targeting), estimated by maximising the
        correlation part of the Gaussian quasi-likelihood.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from arch import arch_model
from scipy.optimize import minimize


@dataclass
class UnivariateFit:
    name: str
    params: dict
    persistence: float
    sigma: pd.Series
    z: pd.Series


def fit_garch(r: pd.Series) -> UnivariateFit:
    """GJR-GARCH(1,1) with Student-t errors on returns in percent."""
    res = arch_model(r, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="t").fit(disp="off")
    p = res.params
    persistence = float(p["alpha[1]"] + p["beta[1]"] + 0.5 * p["gamma[1]"])
    sigma = res.conditional_volatility
    z = res.resid / sigma
    return UnivariateFit(str(r.name), {k: float(v) for k, v in p.items()}, persistence, sigma, z)


def dcc_filter(z: np.ndarray, a: float, b: float, qbar: np.ndarray | None = None):
    """Run the DCC recursion. Returns Q (T,k,k) and R (T,k,k); R[t] uses information up to t-1."""
    T, k = z.shape
    if qbar is None:
        qbar = np.cov(z, rowvar=False, bias=True)
    Q = np.empty((T, k, k))
    R = np.empty((T, k, k))
    q = qbar.copy()
    for t in range(T):
        if t > 0:
            zz = np.outer(z[t - 1], z[t - 1])
            q = (1 - a - b) * qbar + a * zz + b * q
        Q[t] = q
        d = 1.0 / np.sqrt(np.diag(q))
        R[t] = q * np.outer(d, d)
    return Q, R


def dcc_negloglik(theta, z: np.ndarray, qbar: np.ndarray) -> float:
    a, b = theta
    if a < 0 or b < 0 or a + b >= 0.9999:
        return 1e10
    _, R = dcc_filter(z, a, b, qbar)
    sign, logdet = np.linalg.slogdet(R)
    if np.any(sign <= 0):
        return 1e10
    quad = np.einsum("ti,tij,tj->t", z, np.linalg.inv(R), z)
    return float(0.5 * np.sum(logdet + quad - np.sum(z * z, axis=1)))


@dataclass
class DCCFit:
    a: float
    b: float
    loglik: float
    qbar: np.ndarray
    R: np.ndarray
    converged: bool


def fit_dcc(z: np.ndarray) -> DCCFit:
    """Estimate a and b from several starting points and keep the best."""
    qbar = np.cov(z, rowvar=False, bias=True)
    best = None
    for a0, b0 in [(0.01, 0.97), (0.03, 0.95), (0.05, 0.90), (0.10, 0.80)]:
        res = minimize(dcc_negloglik, x0=[a0, b0], args=(z, qbar), method="SLSQP",
                       bounds=[(1e-6, 0.5), (1e-6, 0.9999)],
                       constraints=[{"type": "ineq", "fun": lambda th: 0.9998 - th[0] - th[1]}])
        if best is None or res.fun < best.fun:
            best = res
    a, b = map(float, best.x)
    _, R = dcc_filter(z, a, b, qbar)
    return DCCFit(a, b, -float(best.fun), qbar, R, bool(best.success))


def fit_dcc_garch(returns: pd.DataFrame):
    """Fit the marginals and the DCC. Returns (univariate fits, DCC fit, z as DataFrame)."""
    uni = [fit_garch(returns[c]) for c in returns.columns]
    z = pd.concat([u.z for u in uni], axis=1)
    z.columns = returns.columns
    dcc = fit_dcc(z.to_numpy())
    return uni, dcc, z


def pairwise_frame(R: np.ndarray, index: pd.Index, names: list[str]) -> pd.DataFrame:
    """Turn a (T,k,k) correlation array into one column per pair."""
    out = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            out[f"{names[i]}-{names[j]}"] = R[:, i, j]
    return pd.DataFrame(out, index=index)


def simulate_dcc(T: int, a: float, b: float, corr: np.ndarray, seed: int = 0) -> np.ndarray:
    """Simulate Gaussian standardised residuals from a DCC(1,1) process (used in tests)."""
    rng = np.random.default_rng(seed)
    k = corr.shape[0]
    q = corr.copy()
    z = np.empty((T, k))
    prev = np.zeros(k)
    for t in range(T):
        if t > 0:
            q = (1 - a - b) * corr + a * np.outer(prev, prev) + b * q
        d = 1.0 / np.sqrt(np.diag(q))
        r = q * np.outer(d, d)
        z[t] = np.linalg.cholesky(r) @ rng.standard_normal(k)
        prev = z[t]
    return z

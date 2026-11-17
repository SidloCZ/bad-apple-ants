from __future__ import annotations

import numpy as np

R_KM = 1737.4


def ortho_forward(lon, lat, lon0, lat0):
    """Degrees in, tangent-plane km out (x east, y north) for the orthographic projection."""
    lam, phi = np.radians(lon), np.radians(lat)
    lam0, phi0 = np.radians(lon0), np.radians(lat0)
    x = R_KM * np.cos(phi) * np.sin(lam - lam0)
    y = R_KM * (np.cos(phi0) * np.sin(phi) - np.sin(phi0) * np.cos(phi) * np.cos(lam - lam0))
    return x, y


def ortho_inverse(x, y, lon0, lat0):
    """Tangent-plane km in, degrees out."""
    x = np.asarray(x, dtype=np.float64) / R_KM
    y = np.asarray(y, dtype=np.float64) / R_KM
    phi0, lam0 = np.radians(lat0), np.radians(lon0)
    rho = np.hypot(x, y)
    c = np.arcsin(np.clip(rho, 0, 1))
    safe = np.where(rho > 0, rho, 1)
    phi = np.where(
        rho > 0, np.arcsin(np.clip(np.cos(c) * np.sin(phi0) + y * np.sin(c) * np.cos(phi0) / safe, -1, 1)), phi0
    )
    lam = lam0 + np.arctan2(x * np.sin(c), rho * np.cos(c) * np.cos(phi0) - y * np.sin(c) * np.sin(phi0))
    lon = (np.degrees(lam) + 180) % 360 - 180
    return lon, np.degrees(phi)


def quad_terms(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], axis=-1)


def fit_quadratic(x, y, u, v):
    """Least-squares quadratic map (x, y) -> (u, v); returns 2x6 coefficients and max residual."""
    a = quad_terms(x, y)
    cu, *_ = np.linalg.lstsq(a, u, rcond=None)
    cv, *_ = np.linalg.lstsq(a, v, rcond=None)
    residual = max(np.abs(a @ cu - u).max(), np.abs(a @ cv - v).max())
    return np.stack([cu, cv]), float(residual)


def apply_quadratic(coefficients, x, y):
    a = quad_terms(x, y)
    return a @ coefficients[0], a @ coefficients[1]

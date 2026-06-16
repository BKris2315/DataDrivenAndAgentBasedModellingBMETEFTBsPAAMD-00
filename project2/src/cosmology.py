"""Small cosmology helper functions used by the notebooks."""
from __future__ import annotations

import numpy as np

C_KM_S = 299_792.458


def distance_modulus_to_mpc(mu):
    """Convert distance modulus mu to luminosity distance in Mpc.

    mu = 5 log10(D_L / 10 pc), so D_L(Mpc) = 10 ** ((mu - 25) / 5).
    """
    return 10 ** ((np.asarray(mu, dtype=float) - 25.0) / 5.0)


def redshift_to_velocity_lowz(z):
    """Low-redshift approximation v = c z in km/s.

    This is intentionally used only for z << 1. For high-z cosmology one should fit a
    proper luminosity-distance model, not a straight line.
    """
    return C_KM_S * np.asarray(z, dtype=float)


def wavelength_redshift(observed_angstrom, rest_angstrom):
    """Compute z from observed and rest wavelengths."""
    return np.asarray(observed_angstrom, dtype=float) / np.asarray(rest_angstrom, dtype=float) - 1.0

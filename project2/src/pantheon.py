"""Loading, cleaning, and fitting utilities for Pantheon+SH0ES."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.linear_model import HuberRegressor, LinearRegression

from .cosmology import distance_modulus_to_mpc, redshift_to_velocity_lowz

PANTHEON_FILE = Path("data/raw/Pantheon+SH0ES.dat")
PROCESSED_FILE = Path("data/processed/pantheon_clean.csv")

REQUIRED_COLUMNS = [
    "CID", "IDSURVEY", "zHD", "zHDERR", "zCMB", "zCMBERR", "zHEL", "zHELERR",
    "MU_SH0ES", "MU_SH0ES_ERR_DIAG", "CEPH_DIST", "IS_CALIBRATOR", "USED_IN_SH0ES_HF",
    "RA", "DEC", "VPEC", "VPECERR", "HOST_LOGMASS",
]


@dataclass
class CleaningSummary:
    raw_rows: int
    rows_after_missing: int
    rows_after_calibrator_filter: int
    rows_after_redshift_cut: int
    z_min: float
    z_max: float
    mu_min: float
    mu_max: float

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([self.__dict__])


def load_pantheon(path: str | Path = PANTHEON_FILE) -> pd.DataFrame:
    """Load the official Pantheon+SH0ES whitespace-separated table."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run `python download_official_data.py` or manually place "
            "Pantheon+SH0ES.dat in data/raw/."
        )
    df = pd.read_csv(path, delim_whitespace=True, comment="#")
    return df


def clean_pantheon(
    df: pd.DataFrame,
    z_min: float = 0.01,
    z_max: float = 0.15,
    exclude_calibrators: bool = True,
) -> tuple[pd.DataFrame, CleaningSummary]:
    """Clean Pantheon+ for a low-z Hubble-law project.

    The cleaning is deliberately conservative and easy to explain in a class report:
    - keep columns needed for the analysis;
    - convert common placeholder values such as -9 and -999 into NaN;
    - remove rows without redshift or distance modulus;
    - optionally remove cepheid calibrator rows to avoid mixing calibration objects with Hubble-flow objects;
    - keep only low-redshift objects where v approx c z is defensible.
    """
    raw_rows = len(df)
    cols = [c for c in REQUIRED_COLUMNS if c in df.columns]
    cleaned = df[cols].copy()

    numeric_cols = [c for c in cleaned.columns if c != "CID"]
    for c in numeric_cols:
        cleaned[c] = pd.to_numeric(cleaned[c], errors="coerce")

    cleaned = cleaned.replace({-9: np.nan, -99: np.nan, -999: np.nan, -9999: np.nan})
    cleaned = cleaned.dropna(subset=["zHD", "MU_SH0ES", "MU_SH0ES_ERR_DIAG"])
    rows_after_missing = len(cleaned)

    if exclude_calibrators and "IS_CALIBRATOR" in cleaned.columns:
        cleaned = cleaned[cleaned["IS_CALIBRATOR"].fillna(0).astype(int) == 0].copy()
    rows_after_calibrator_filter = len(cleaned)

    cleaned = cleaned[(cleaned["zHD"] >= z_min) & (cleaned["zHD"] <= z_max)].copy()
    rows_after_redshift_cut = len(cleaned)

    cleaned["distance_mpc"] = distance_modulus_to_mpc(cleaned["MU_SH0ES"])
    cleaned["velocity_km_s"] = redshift_to_velocity_lowz(cleaned["zHD"])
    cleaned["distance_err_mpc"] = cleaned["distance_mpc"] * np.log(10) / 5 * cleaned["MU_SH0ES_ERR_DIAG"]
    # Pantheon+ gives luminosity distance.  For a simple local Hubble-law
    # velocity-distance plot, D_L/(1+z) is a better low-redshift distance proxy
    # than D_L itself, especially near the upper edge of z <= 0.15.
    cleaned["hubble_distance_mpc"] = cleaned["distance_mpc"] / (1.0 + cleaned["zHD"])
    cleaned["hubble_distance_err_mpc"] = cleaned["distance_err_mpc"] / (1.0 + cleaned["zHD"])
    cleaned["weight_mu"] = 1.0 / np.square(cleaned["MU_SH0ES_ERR_DIAG"])

    summary = CleaningSummary(
        raw_rows=raw_rows,
        rows_after_missing=rows_after_missing,
        rows_after_calibrator_filter=rows_after_calibrator_filter,
        rows_after_redshift_cut=rows_after_redshift_cut,
        z_min=float(cleaned["zHD"].min()) if len(cleaned) else np.nan,
        z_max=float(cleaned["zHD"].max()) if len(cleaned) else np.nan,
        mu_min=float(cleaned["MU_SH0ES"].min()) if len(cleaned) else np.nan,
        mu_max=float(cleaned["MU_SH0ES"].max()) if len(cleaned) else np.nan,
    )
    return cleaned.reset_index(drop=True), summary


def save_cleaned(df: pd.DataFrame, path: str | Path = PROCESSED_FILE) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_cleaned(path: str | Path = PROCESSED_FILE) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError("Run notebook 01 or run_official_pantheon_analysis.py first.")
    df = pd.read_csv(path)
    # Older processed files did not include the corrected local-distance proxy.
    # Reconstruct it on load so downstream notebooks remain robust.
    if "hubble_distance_mpc" not in df.columns and {"distance_mpc", "zHD"}.issubset(df.columns):
        df["hubble_distance_mpc"] = df["distance_mpc"] / (1.0 + df["zHD"])
    if "hubble_distance_err_mpc" not in df.columns and {"distance_err_mpc", "zHD"}.issubset(df.columns):
        df["hubble_distance_err_mpc"] = df["distance_err_mpc"] / (1.0 + df["zHD"])
    return df


DISTANCE_LABELS = {
    "distance_mpc": "luminosity distance D_L",
    "hubble_distance_mpc": "local Hubble-distance proxy D_L/(1+z)",
}


def default_hubble_distance_col(df: pd.DataFrame) -> str:
    """Return the preferred distance column for the simple local Hubble fit."""
    return "hubble_distance_mpc" if "hubble_distance_mpc" in df.columns else "distance_mpc"


def distance_label(distance_col: str) -> str:
    """Human-readable label for a distance column."""
    return DISTANCE_LABELS.get(distance_col, distance_col)


def _fit_arrays(df: pd.DataFrame, distance_col: str | None = None) -> tuple[np.ndarray, np.ndarray, str]:
    """Extract finite distance and velocity arrays for Hubble-law fitting."""
    distance_col = distance_col or default_hubble_distance_col(df)
    if distance_col not in df.columns:
        raise KeyError(f"Missing distance column {distance_col!r}.")
    d = df[distance_col].to_numpy(float)
    v = df["velocity_km_s"].to_numpy(float)
    ok = np.isfinite(d) & np.isfinite(v)
    return d[ok], v[ok], distance_col


def _fit_row(method: str, h0: float, intercept: float, residual: np.ndarray, n: int, distance_col: str) -> dict:
    """Build a consistent row for fit-comparison tables."""
    return {
        "method": method,
        "distance_col": distance_col,
        "distance_definition": distance_label(distance_col),
        "H0": float(h0),
        "intercept": float(intercept),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "n": int(n),
    }


def _mu_weights(df: pd.DataFrame) -> np.ndarray:
    """Return safe relative weights from distance-modulus uncertainty."""
    if "weight_mu" in df.columns:
        w = df["weight_mu"].to_numpy(float)
    elif "MU_SH0ES_ERR_DIAG" in df.columns:
        sigma = df["MU_SH0ES_ERR_DIAG"].to_numpy(float)
        w = 1.0 / np.square(sigma)
    else:
        w = np.ones(len(df), dtype=float)
    finite_positive = np.isfinite(w) & (w > 0)
    if finite_positive.any():
        fill = float(np.median(w[finite_positive]))
    else:
        fill = 1.0
    return np.where(finite_positive, w, fill)


def fit_h0_origin(df: pd.DataFrame, distance_col: str | None = None) -> dict:
    """Fit v = H0 d with zero intercept using ordinary least squares."""
    d, v, distance_col = _fit_arrays(df, distance_col)
    h0 = np.sum(d * v) / np.sum(d * d)
    residual = v - h0 * d
    return _fit_row("ordinary least squares, forced through origin", h0, 0.0, residual, len(d), distance_col)


def fit_h0_free_intercept(df: pd.DataFrame, distance_col: str | None = None) -> dict:
    """Fit v = H0 d + b using ordinary least squares."""
    d, y, distance_col = _fit_arrays(df, distance_col)
    X = d.reshape(-1, 1)
    model = LinearRegression().fit(X, y)
    pred = model.predict(X)
    return _fit_row("ordinary least squares, free intercept", model.coef_[0], model.intercept_, y - pred, len(d), distance_col)


def fit_h0_huber(df: pd.DataFrame, distance_col: str | None = None) -> dict:
    """Robust fit v = H0 d + b using Huber regression."""
    d, y, distance_col = _fit_arrays(df, distance_col)
    X = d.reshape(-1, 1)
    model = HuberRegressor().fit(X, y)
    pred = model.predict(X)
    return _fit_row("Huber robust regression", model.coef_[0], model.intercept_, y - pred, len(d), distance_col)


def fit_h0_weighted_origin(df: pd.DataFrame, distance_col: str | None = None) -> dict:
    """Fit v = H0 d through the origin with distance-modulus precision weights.

    This is still a simplified educational fit: the weights come from the
    diagonal distance-modulus errors, while a full cosmology fit would use the
    full covariance matrix and redshift/peculiar-velocity uncertainties.
    """
    distance_col = distance_col or default_hubble_distance_col(df)
    d_all = df[distance_col].to_numpy(float)
    v_all = df["velocity_km_s"].to_numpy(float)
    w_all = _mu_weights(df)
    ok = np.isfinite(d_all) & np.isfinite(v_all) & np.isfinite(w_all) & (w_all > 0)
    d, v, w = d_all[ok], v_all[ok], w_all[ok]
    h0 = np.sum(w * d * v) / np.sum(w * d * d)
    residual = v - h0 * d
    return _fit_row("weighted least squares, forced through origin", h0, 0.0, residual, len(d), distance_col)


def fit_h0_weighted_free_intercept(df: pd.DataFrame, distance_col: str | None = None) -> dict:
    """Fit v = H0 d + b with distance-modulus precision weights."""
    distance_col = distance_col or default_hubble_distance_col(df)
    d_all = df[distance_col].to_numpy(float)
    v_all = df["velocity_km_s"].to_numpy(float)
    w_all = _mu_weights(df)
    ok = np.isfinite(d_all) & np.isfinite(v_all) & np.isfinite(w_all) & (w_all > 0)
    d, v, w = d_all[ok], v_all[ok], w_all[ok]
    X = np.column_stack([d, np.ones_like(d)])
    sw = np.sqrt(w)
    coef, intercept = np.linalg.lstsq(X * sw[:, None], v * sw, rcond=None)[0]
    pred = coef * d + intercept
    return _fit_row("weighted least squares, free intercept", coef, intercept, v - pred, len(d), distance_col)


FIT_METHODS = {
    "origin": fit_h0_origin,
    "free_intercept": fit_h0_free_intercept,
    "huber": fit_h0_huber,
    "weighted_origin": fit_h0_weighted_origin,
    "weighted_free_intercept": fit_h0_weighted_free_intercept,
}


def _fit_by_name(df: pd.DataFrame, method: str, distance_col: str | None = None) -> dict:
    if method not in FIT_METHODS:
        raise KeyError(f"Unknown fit method {method!r}. Expected one of {sorted(FIT_METHODS)}.")
    row = FIT_METHODS[method](df, distance_col=distance_col)
    row["method_key"] = method
    return row


def fit_results_table(
    df: pd.DataFrame,
    distance_cols: Iterable[str] | None = None,
    methods: Iterable[str] = ("origin", "free_intercept", "huber", "weighted_origin", "weighted_free_intercept"),
) -> pd.DataFrame:
    """Compare Hubble-law fits across distance definitions and fit methods."""
    if distance_cols is None:
        distance_cols = [default_hubble_distance_col(df)]
    rows = []
    for distance_col in distance_cols:
        for method in methods:
            rows.append(_fit_by_name(df, method, distance_col=distance_col))
    return pd.DataFrame(rows)


def h0_for_redshift_cuts(
    df: pd.DataFrame,
    cuts: Iterable[float] = (0.03, 0.05, 0.08, 0.10, 0.15),
    distance_col: str | None = None,
    methods: Iterable[str] = ("origin",),
) -> pd.DataFrame:
    rows = []
    for zmax in cuts:
        sub = df[df["zHD"] <= zmax].copy()
        if len(sub) < 5:
            continue
        for method in methods:
            res = _fit_by_name(sub, method, distance_col=distance_col)
            res["z_min"] = float(sub["zHD"].min())
            res["z_max_cut"] = float(zmax)
            res["sample"] = f"z <= {zmax:g}"
            rows.append(res)
    return pd.DataFrame(rows)


def h0_for_redshift_bins(
    df: pd.DataFrame,
    bins=(0.01, 0.03, 0.05, 0.08, 0.10, 0.15),
    distance_col: str | None = None,
    methods: Iterable[str] = ("origin",),
) -> pd.DataFrame:
    rows = []
    cats = pd.cut(df["zHD"], bins=bins, include_lowest=True)
    for interval, sub in df.groupby(cats, observed=True):
        if len(sub) < 5:
            continue
        for method in methods:
            res = _fit_by_name(sub, method, distance_col=distance_col)
            res["bin"] = str(interval)
            res["z_min"] = float(sub["zHD"].min())
            res["z_max"] = float(sub["zHD"].max())
            res["sample"] = str(interval)
            rows.append(res)
    return pd.DataFrame(rows)


def h0_for_lower_redshift_cuts(
    df: pd.DataFrame,
    zmins: Iterable[float] = (0.005, 0.01, 0.015, 0.02, 0.03),
    zmax: float = 0.15,
    distance_col: str | None = None,
    methods: Iterable[str] = ("origin",),
) -> pd.DataFrame:
    """Fit after removing progressively more nearby supernovae."""
    rows = []
    for zmin in zmins:
        sub = df[(df["zHD"] >= zmin) & (df["zHD"] <= zmax)].copy()
        if len(sub) < 5:
            continue
        for method in methods:
            res = _fit_by_name(sub, method, distance_col=distance_col)
            res["z_min_cut"] = float(zmin)
            res["z_max_cut"] = float(zmax)
            res["sample"] = f"{zmin:g} <= z <= {zmax:g}"
            rows.append(res)
    return pd.DataFrame(rows)


def bootstrap_h0(
    df: pd.DataFrame,
    method: str = "origin",
    distance_col: str | None = None,
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> dict:
    """Bootstrap the Hubble-law slope for a simple sampling-uncertainty check."""
    rng = np.random.default_rng(random_state)
    n = len(df)
    point = _fit_by_name(df, method, distance_col=distance_col)
    boot = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        sample = df.iloc[rng.integers(0, n, size=n)]
        boot[i] = _fit_by_name(sample, method, distance_col=point["distance_col"])["H0"]
    qs = np.percentile(boot, [2.5, 16, 50, 84, 97.5])
    return {
        "method": point["method"],
        "method_key": method,
        "distance_col": point["distance_col"],
        "distance_definition": point["distance_definition"],
        "H0": point["H0"],
        "H0_bootstrap_median": float(qs[2]),
        "H0_ci16": float(qs[1]),
        "H0_ci84": float(qs[3]),
        "H0_ci95_low": float(qs[0]),
        "H0_ci95_high": float(qs[4]),
        "n_bootstrap": int(n_bootstrap),
        "n": int(n),
    }

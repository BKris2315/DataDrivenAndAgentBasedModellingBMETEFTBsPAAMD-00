"""SDSS spectrum loading and redshift measurement utilities."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from .cosmology import wavelength_redshift

SDSS_SPECTRUM_FILE = Path("data/raw/spec-0532-51993-0497.fits")

# Strong rest-frame emission lines expected in many galaxy/QSO spectra, Angstrom.
REST_LINES = {
    "[O II] 3727": 3727.09,
    "[Ne III] 3869": 3869.86,
    "H-delta 4102": 4101.74,
    "H-gamma 4341": 4340.47,
    "H-beta 4861": 4861.33,
    "[O III] 4959": 4958.91,
    "[O III] 5007": 5006.84,
    "H-alpha 6563": 6562.80,
    "[N II] 6584": 6583.45,
}


def load_sdss_spectrum(path: str | Path = SDSS_SPECTRUM_FILE) -> tuple[pd.DataFrame, dict]:
    """Load an SDSS spPlate/spec FITS spectrum.

    Returns a dataframe with wavelength, flux, ivar, and model when present,
    plus metadata such as catalog redshift when available.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run `python download_official_data.py` or manually download the FITS spectrum."
        )
    with fits.open(path) as hdul:
        data = hdul[1].data
        names = set(data.names)
        loglam = np.array(data["loglam"], dtype=float)
        wave = 10 ** loglam
        df = pd.DataFrame({
            "wavelength": wave,
            "flux": np.array(data["flux"], dtype=float),
            "ivar": np.array(data["ivar"], dtype=float) if "ivar" in names else np.nan,
        })
        if "model" in names:
            df["model"] = np.array(data["model"], dtype=float)
        meta = {}
        # Common SDSS spec file: HDU 2 has one-row specObj table containing Z, CLASS, PLATE, MJD, FIBERID.
        if len(hdul) > 2 and getattr(hdul[2].data, "names", None):
            row = hdul[2].data[0]
            for key in ["Z", "Z_ERR", "CLASS", "PLATE", "MJD", "FIBERID", "SPECOBJID"]:
                if key in hdul[2].data.names:
                    val = row[key]
                    if isinstance(val, bytes):
                        val = val.decode().strip()
                    meta[key] = val.item() if hasattr(val, "item") else val
    return df, meta


def continuum_subtract(flux: np.ndarray, kernel_size: int = 151) -> np.ndarray:
    """Remove slowly varying continuum with a median filter."""
    if kernel_size % 2 == 0:
        kernel_size += 1
    continuum = median_filter(flux, size=kernel_size, mode="nearest")
    return flux - continuum


def gaussian(x, amp, mu, sigma, offset):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + offset


def find_candidate_emission_peaks(df: pd.DataFrame, min_prominence_sigma: float = 5.0) -> pd.DataFrame:
    """Find prominent emission peaks in a continuum-subtracted spectrum."""
    clean = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["wavelength", "flux"]).copy()
    residual = continuum_subtract(clean["flux"].to_numpy(float))
    noise = np.nanmedian(np.abs(residual - np.nanmedian(residual))) * 1.4826
    if not np.isfinite(noise) or noise <= 0:
        noise = np.nanstd(residual)
    peaks, props = find_peaks(residual, prominence=min_prominence_sigma * noise, distance=8)
    out = pd.DataFrame({
        "observed_wavelength": clean["wavelength"].to_numpy()[peaks],
        "line_signal": residual[peaks],
        "prominence": props.get("prominences", np.full(len(peaks), np.nan)),
    }).sort_values("prominence", ascending=False).reset_index(drop=True)
    return out


def match_peaks_to_lines(peaks: pd.DataFrame, z_min: float = 0.0, z_max: float = 1.0, tolerance: float = 0.004) -> pd.DataFrame:
    """Match observed peaks to rest wavelengths by looking for a common redshift.

    The method considers every pair of observed peak and known rest line, computes z,
    keeps plausible values, and clusters them by redshift. The dominant cluster gives
    a rough redshift without reading the catalog z.
    """
    candidates = []
    for _, p in peaks.iterrows():
        obs = float(p["observed_wavelength"])
        for name, rest in REST_LINES.items():
            z = obs / rest - 1.0
            if z_min <= z <= z_max:
                candidates.append({"observed_wavelength": obs, "rest_line": name, "rest_wavelength": rest, "z_candidate": z, "prominence": p["prominence"]})
    cand = pd.DataFrame(candidates)
    if cand.empty:
        return cand
    cand = cand.sort_values("z_candidate").reset_index(drop=True)
    # simple clustering by sorted z proximity
    labels = []
    current_label = 0
    last_z = None
    for z in cand["z_candidate"]:
        if last_z is not None and abs(z - last_z) > tolerance:
            current_label += 1
        labels.append(current_label)
        last_z = z
    cand["cluster"] = labels
    cluster_score = cand.groupby("cluster").agg(n=("z_candidate", "size"), z_med=("z_candidate", "median"), prominence_sum=("prominence", "sum")).reset_index()
    cluster_score = cluster_score.sort_values(["n", "prominence_sum"], ascending=False)
    best = int(cluster_score.iloc[0]["cluster"])
    best_matches = cand[cand["cluster"] == best].copy()
    return best_matches.sort_values("rest_wavelength").reset_index(drop=True)


def refine_line_centers(df: pd.DataFrame, z_initial: float, window_angstrom: float = 18.0) -> pd.DataFrame:
    """Fit Gaussian centers around expected observed wavelengths for several lines."""
    rows = []
    wave = df["wavelength"].to_numpy(float)
    flux_resid = continuum_subtract(df["flux"].to_numpy(float))
    for name, rest in REST_LINES.items():
        expected = rest * (1.0 + z_initial)
        mask = (wave > expected - window_angstrom) & (wave < expected + window_angstrom)
        if mask.sum() < 8:
            continue
        x = wave[mask]
        y = flux_resid[mask]
        if np.nanmax(y) <= 0:
            continue
        p0 = [np.nanmax(y), x[np.nanargmax(y)], 3.0, np.nanmedian(y)]
        try:
            popt, pcov = curve_fit(gaussian, x, y, p0=p0, maxfev=5000)
            amp, mu, sigma, offset = popt
            if amp <= 0 or not (expected - window_angstrom < mu < expected + window_angstrom):
                continue
            z = wavelength_redshift(mu, rest)
            rows.append({
                "rest_line": name,
                "rest_wavelength": rest,
                "expected_observed_wavelength": expected,
                "measured_observed_wavelength": mu,
                "z_measured": float(z),
                "amplitude": float(amp),
                "sigma_angstrom": abs(float(sigma)),
            })
        except Exception:
            continue
    return pd.DataFrame(rows).sort_values("rest_wavelength").reset_index(drop=True)


def weighted_mean_redshift(lines: pd.DataFrame) -> tuple[float, float]:
    """Compute a weighted mean redshift using fitted line amplitudes as simple weights."""
    if lines.empty:
        return np.nan, np.nan
    z = lines["z_measured"].to_numpy(float)
    weights = np.clip(lines["amplitude"].to_numpy(float), 0, None)
    if weights.sum() <= 0:
        weights = np.ones_like(z)
    z_mean = np.average(z, weights=weights)
    scatter = np.sqrt(np.average((z - z_mean) ** 2, weights=weights))
    return float(z_mean), float(scatter)

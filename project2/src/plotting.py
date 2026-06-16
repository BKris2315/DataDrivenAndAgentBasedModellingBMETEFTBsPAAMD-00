"""Plotting utilities for the project."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .pantheon import default_hubble_distance_col, distance_label, fit_h0_origin
from .sdss_spectrum import continuum_subtract, REST_LINES

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)


def savefig(path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_hubble_diagram(
    df: pd.DataFrame,
    path: str | Path = FIG_DIR / "pantheon_hubble_diagram.png",
    distance_col: str | None = None,
) -> None:
    distance_col = distance_col or default_hubble_distance_col(df)
    fit = fit_h0_origin(df, distance_col=distance_col)
    h0 = fit["H0"]
    x = df[distance_col].to_numpy(float)
    y = df["velocity_km_s"].to_numpy(float)
    grid = np.linspace(0, x.max() * 1.03, 200)
    err_col = "hubble_distance_err_mpc" if distance_col == "hubble_distance_mpc" else "distance_err_mpc"
    plt.figure(figsize=(8, 5))
    plt.errorbar(x, y, xerr=df.get(err_col, None), fmt=".", alpha=0.55, markersize=4, linewidth=0.6)
    plt.plot(grid, h0 * grid, label=f"v = H0 d, H0 = {h0:.2f} km/s/Mpc")
    plt.xlabel(f"{distance_label(distance_col)} (Mpc)")
    plt.ylabel("Recession velocity approximation czHD (km/s)")
    plt.title("Low-redshift Pantheon+SH0ES Hubble diagram")
    plt.legend()
    savefig(path)


def plot_mu_vs_redshift(df: pd.DataFrame, path: str | Path = FIG_DIR / "pantheon_mu_vs_redshift.png") -> None:
    plt.figure(figsize=(8, 5))
    plt.errorbar(df["zHD"], df["MU_SH0ES"], yerr=df["MU_SH0ES_ERR_DIAG"], fmt=".", alpha=0.6, markersize=4, linewidth=0.6)
    plt.xlabel("Hubble diagram redshift zHD")
    plt.ylabel("Distance modulus MU_SH0ES")
    plt.title("Pantheon+SH0ES distance modulus vs redshift")
    savefig(path)


def plot_residuals(
    df: pd.DataFrame,
    path: str | Path = FIG_DIR / "pantheon_residuals.png",
    distance_col: str | None = None,
) -> None:
    distance_col = distance_col or default_hubble_distance_col(df)
    fit = fit_h0_origin(df, distance_col=distance_col)
    h0 = fit["H0"]
    residuals = df["velocity_km_s"] - h0 * df[distance_col]
    plt.figure(figsize=(8, 5))
    plt.scatter(df[distance_col], residuals, s=16, alpha=0.6)
    plt.axhline(0, linewidth=1)
    plt.xlabel(f"{distance_label(distance_col)} (Mpc)")
    plt.ylabel("Velocity residual v - H0 d (km/s)")
    plt.title("Residuals around the linear Hubble-law fit")
    savefig(path)


def plot_h0_by_redshift_cut(cuts_df: pd.DataFrame, path: str | Path = FIG_DIR / "pantheon_h0_by_redshift_cut.png") -> None:
    plt.figure(figsize=(8, 5))
    if "method" in cuts_df.columns and cuts_df["method"].nunique() > 1:
        for method, sub in cuts_df.groupby("method", sort=False):
            plt.plot(sub["z_max_cut"], sub["H0"], marker="o", label=method)
        plt.legend(fontsize=8)
    else:
        plt.plot(cuts_df["z_max_cut"], cuts_df["H0"], marker="o")
    plt.xlabel("Maximum redshift included")
    plt.ylabel("Estimated H0 (km/s/Mpc)")
    plt.title("Sensitivity of H0 estimate to redshift cut")
    savefig(path)


def plot_sdss_spectrum_with_lines(df: pd.DataFrame, line_results: pd.DataFrame, z: float, path: str | Path = FIG_DIR / "sdss_spectrum_redshift.png") -> None:
    wave = df["wavelength"].to_numpy(float)
    flux = df["flux"].to_numpy(float)
    plt.figure(figsize=(10, 5))
    plt.plot(wave, flux, linewidth=0.8)
    for _, row in line_results.iterrows():
        obs = row["measured_observed_wavelength"]
        plt.axvline(obs, linestyle="--", linewidth=1)
        plt.text(obs, np.nanpercentile(flux, 93), row["rest_line"], rotation=90, va="top", fontsize=8)
    plt.xlabel("Observed wavelength (Angstrom)")
    plt.ylabel("SDSS flux")
    plt.title(f"SDSS DR15 spectrum with fitted emission lines, measured z = {z:.5f}")
    savefig(path)


def plot_sdss_restframe(df: pd.DataFrame, z: float, path: str | Path = FIG_DIR / "sdss_restframe_spectrum.png") -> None:
    wave_rest = df["wavelength"].to_numpy(float) / (1 + z)
    residual = continuum_subtract(df["flux"].to_numpy(float))
    plt.figure(figsize=(10, 5))
    plt.plot(wave_rest, residual, linewidth=0.8)
    for name, rest in REST_LINES.items():
        if wave_rest.min() < rest < wave_rest.max():
            plt.axvline(rest, linestyle="--", linewidth=0.8)
            plt.text(rest, np.nanpercentile(residual, 94), name, rotation=90, va="top", fontsize=8)
    plt.xlabel("Rest-frame wavelength after dividing by (1+z) (Angstrom)")
    plt.ylabel("Continuum-subtracted flux")
    plt.title("SDSS spectrum shifted to rest frame using measured redshift")
    savefig(path)

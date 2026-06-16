"""Photometric redshift utilities for SDSS ugriz galaxy samples.

This module supports the notebook
`05_sdss_photometric_redshift_prediction.ipynb`.

The goal is not to beat professional photo-z catalogues. The goal is to make a
transparent class-project comparison between:

1. classical empirical models, for example linear/ridge regression, k-nearest
   neighbours, decision trees, random forests;
2. a simple neural-network style model, scikit-learn's MLPRegressor;
3. the spectroscopic redshift, used as the supervised target.

The methods here deliberately keep the feature engineering visible so the
notebook can explain what is happening.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import requests

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

C_KM_S = 299_792.458


DEFAULT_SQL = """
SELECT TOP 10000
    p.objID,
    p.ra,
    p.dec,
    p.modelMag_u AS u,
    p.modelMag_g AS g,
    p.modelMag_r AS r,
    p.modelMag_i AS i,
    p.modelMag_z AS zmag,
    p.modelMagErr_u AS err_u,
    p.modelMagErr_g AS err_g,
    p.modelMagErr_r AS err_r,
    p.modelMagErr_i AS err_i,
    p.modelMagErr_z AS err_z,
    s.z AS spec_z,
    s.zWarning
FROM PhotoObj AS p
JOIN SpecObj AS s ON s.bestObjID = p.objID
WHERE
    s.class = 'GALAXY'
    AND s.z BETWEEN 0.01 AND 0.6
    AND s.zWarning = 0
    AND p.clean = 1
    AND p.modelMag_u BETWEEN 10 AND 28
    AND p.modelMag_g BETWEEN 10 AND 28
    AND p.modelMag_r BETWEEN 10 AND 28
    AND p.modelMag_i BETWEEN 10 AND 28
    AND p.modelMag_z BETWEEN 10 AND 28
    AND p.modelMagErr_u BETWEEN 0 AND 1
    AND p.modelMagErr_g BETWEEN 0 AND 1
    AND p.modelMagErr_r BETWEEN 0 AND 1
    AND p.modelMagErr_i BETWEEN 0 AND 1
    AND p.modelMagErr_z BETWEEN 0 AND 1
""".strip()


def sdss_sql_url(sql: str = DEFAULT_SQL, data_release: str = "dr17") -> str:
    """Return a SkyServer SQL CSV URL for an SDSS query.

    The SkyServer SQL endpoint is convenient for a class project because it
    produces a simple CSV from an official SDSS database query.
    """
    base = f"https://skyserver.sdss.org/{data_release}/SkyServerWS/SearchTools/SqlSearch"
    return base + "?" + urlencode({"cmd": sql, "format": "csv"})


def download_sdss_photoz_sample(
    output_path: str | Path,
    sql: str = DEFAULT_SQL,
    data_release: str = "dr17",
    timeout: int = 120,
) -> Path:
    """Download an SDSS spectroscopic-training sample for photo-z modelling.

    Parameters
    ----------
    output_path:
        CSV path where the official SDSS query result will be saved.
    sql:
        SQL query. The default selects clean galaxies with ugriz model
        magnitudes and spectroscopic redshifts.
    data_release:
        SDSS data release string, e.g. "dr17" or "dr15".
    timeout:
        HTTP timeout in seconds.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    url = sdss_sql_url(sql=sql, data_release=data_release)
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    text = response.text
    if "Error" in text[:1000] or "SQL" in text[:1000] and "Exception" in text[:1000]:
        raise RuntimeError(
            "SkyServer returned a possible SQL error. Open the URL printed below in a browser:\n"
            + url
        )
    output_path.write_text(text, encoding="utf-8")
    return output_path


def _first_present(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    """Return the first candidate column that exists, ignoring case/spacing."""
    normalized = {str(c).strip().lower(): c for c in columns}
    for candidate in candidates:
        key = candidate.strip().lower()
        if key in normalized:
            return normalized[key]
    return None


def _looks_like_magnitude(values: pd.Series) -> bool:
    """Heuristic for distinguishing an SDSS z-band magnitude from a photo-z value."""
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return False
    median = float(numeric.median())
    p95 = float(numeric.quantile(0.95))
    return 8.0 < median < 30.0 and p95 > 10.0


def _looks_like_redshift(values: pd.Series) -> bool:
    """Heuristic for catalogue photo-z columns."""
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return False
    median = float(numeric.median())
    p99 = float(numeric.quantile(0.99))
    return -0.1 <= median < 2.0 and p99 < 5.0


def load_photoz_csv(path: str | Path) -> pd.DataFrame:
    """Load an SDSS photo-z sample and normalize column names.

    The project downloader writes columns such as ``spec_z`` and ``zmag``.
    Some public SDSS/photo-z CSV files instead use names such as ``redshift``
    for the spectroscopic target and ``photometric_z`` for the SDSS z-band
    magnitude.  This loader accepts both formats and creates the canonical
    columns used by the rest of the project.
    """
    df = pd.read_csv(path, comment="#")
    df.columns = [c.strip() for c in df.columns]

    # Normalize common aliases without deleting the original columns.
    # Target spectroscopic redshift.
    if "spec_z" not in df.columns:
        spec_col = _first_present(
            df.columns,
            ["spec_z", "spectroscopic_z", "specz", "z_spec", "redshift", "z"],
        )
        if spec_col is None:
            raise KeyError(
                "Could not find a spectroscopic redshift column. Expected one of: "
                "spec_z, spectroscopic_z, specz, z_spec, redshift, z."
            )
        df["spec_z"] = df[spec_col]

    # SDSS z-band magnitude.  In some datasets this is unfortunately named
    # photometric_z.  It is a magnitude if values are typically ~10-30, not a
    # redshift.
    if "zmag" not in df.columns:
        zmag_col = None
        for candidate in ["zmag", "modelMag_z", "z_mag", "z_band", "z_magnitude", "z", "photometric_z"]:
            col = _first_present(df.columns, [candidate])
            if col is None:
                continue
            if candidate.lower() == "photometric_z" and not _looks_like_magnitude(df[col]):
                continue
            zmag_col = col
            break
        if zmag_col is not None:
            df["zmag"] = df[zmag_col]

    # Optional already-estimated photometric redshift from a catalogue.  Keep it
    # separately so it is not confused with the z-band magnitude.
    if "catalog_photoz" not in df.columns:
        photoz_col = _first_present(
            df.columns,
            ["photoz", "photo_z", "photo_z_mean", "zphot", "z_phot", "catalog_photoz"],
        )
        if photoz_col is None:
            possible = _first_present(df.columns, ["photometric_z"])
            if possible is not None and _looks_like_redshift(df[possible]):
                photoz_col = possible
        if photoz_col is not None:
            df["catalog_photoz"] = df[photoz_col]

    df["spec_z"] = pd.to_numeric(df["spec_z"], errors="coerce")
    if "zmag" in df.columns:
        df["zmag"] = pd.to_numeric(df["zmag"], errors="coerce")
    return df


def clean_photoz_sample(
    df: pd.DataFrame,
    z_min: float = 0.01,
    z_max: float = 0.6,
    max_redshift_error: float | None = 0.01,
    color_bounds: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Clean a raw SDSS ugriz + spectroscopic-redshift sample.

    The cleaning steps are intentionally explicit for teaching:
    - coerce numeric columns;
    - remove missing/non-finite values;
    - keep physically plausible magnitudes, redshifts, and colours;
    - remove rows with bad SDSS spectroscopic warnings;
    - derive colour features from adjacent magnitude bands.
    """
    if color_bounds is None:
        color_bounds = {
            "u_g": (-1.0, 6.0),
            "g_r": (-1.0, 3.0),
            "r_i": (-1.0, 2.0),
            "i_z": (-1.5, 2.0),
        }

    df = df.copy()
    numeric_cols = [
        "u", "g", "r", "i", "zmag",
        "err_u", "err_g", "err_r", "err_i", "err_z",
        "spec_z", "redshift_error", "zWarning",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    required = ["u", "g", "r", "i", "zmag", "spec_z"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(
            "Missing required photo-z columns after alias normalization: "
            + ", ".join(missing)
            + ". Need ugriz magnitudes plus a spectroscopic redshift target. "
            + "Common accepted aliases include redshift -> spec_z and photometric_z/z -> zmag."
        )
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=required)

    if "zWarning" in df.columns:
        df = df[df["zWarning"] == 0]
    if "class" in df.columns:
        df = df[df["class"].astype(str).str.upper().eq("GALAXY")]

    for band in ["u", "g", "r", "i", "zmag"]:
        df = df[(df[band] > 10) & (df[band] < 30)]
    df = df[(df["spec_z"] >= z_min) & (df["spec_z"] <= z_max)]
    if max_redshift_error is not None and "redshift_error" in df.columns:
        df = df[(df["redshift_error"] >= 0) & (df["redshift_error"] <= max_redshift_error)]

    # Colour indices are often more informative than raw magnitudes because a
    # redshifted spectrum changes relative brightness between filters.
    df["u_g"] = df["u"] - df["g"]
    df["g_r"] = df["g"] - df["r"]
    df["r_i"] = df["r"] - df["i"]
    df["i_z"] = df["i"] - df["zmag"]
    for col, (lo, hi) in color_bounds.items():
        if col in df.columns:
            df = df[df[col].between(lo, hi)]

    # A rough apparent-brightness feature. It is not a distance by itself, but
    # it can help empirical models when the training and test distributions are
    # similar.
    df["r_mag"] = df["r"]
    return df.reset_index(drop=True)


PHOTOZ_FEATURES = ["u", "g", "r", "i", "zmag", "u_g", "g_r", "r_i", "i_z", "r_mag"]


def stratified_photoz_sample(
    df: pd.DataFrame,
    max_rows: int | None = 60000,
    target: str = "spec_z",
    n_bins: int = 12,
    random_state: int = 42,
) -> pd.DataFrame:
    """Return a reproducible redshift-stratified modelling sample.

    Very large public SDSS CSVs can make a teaching notebook slow and produce
    enormous output files.  A stratified subsample keeps the redshift
    distribution represented while making repeated model comparisons practical.
    """
    if max_rows is None or len(df) <= max_rows:
        return df.reset_index(drop=True)

    sampled = []
    bins = pd.qcut(df[target], q=n_bins, duplicates="drop")
    per_bin = int(np.ceil(max_rows / bins.nunique()))
    for _, sub in df.groupby(bins, observed=True):
        take = min(len(sub), per_bin)
        sampled.append(sub.sample(n=take, random_state=random_state))
    out = pd.concat(sampled).sample(frac=1.0, random_state=random_state)
    if len(out) > max_rows:
        out = out.sample(n=max_rows, random_state=random_state)
    return out.sort_values(target).reset_index(drop=True)


def photoz_sample_summary(df: pd.DataFrame, label: str) -> dict:
    """Compact row for documenting sample size and redshift coverage."""
    return {
        "sample": label,
        "n": int(len(df)),
        "z_min": float(df["spec_z"].min()) if len(df) else np.nan,
        "z_median": float(df["spec_z"].median()) if len(df) else np.nan,
        "z_max": float(df["spec_z"].max()) if len(df) else np.nan,
    }


def train_test_photoz(
    df: pd.DataFrame,
    features: Iterable[str] = PHOTOZ_FEATURES,
    target: str = "spec_z",
    test_size: float = 0.25,
    random_state: int = 42,
    stratify_bins: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a reproducible train/test split for photo-z modelling."""
    features = list(features)
    X = df[features]
    y = df[target]
    stratify = None
    if stratify_bins and len(df) >= stratify_bins * 2:
        try:
            stratify = pd.qcut(y, q=stratify_bins, duplicates="drop")
            if pd.Series(stratify).nunique(dropna=True) < 2:
                stratify = None
        except ValueError:
            stratify = None
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)


def make_models(random_state: int = 42) -> Dict[str, Pipeline]:
    """Return comparable classical and ML-style photometric-redshift models."""
    def scaled(model):
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ])

    def tree(model):
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ])

    return {
        "median_redshift_baseline": scaled(DummyRegressor(strategy="median")),
        "ridge_linear_baseline": scaled(Ridge(alpha=1.0)),
        "knn_classical": scaled(KNeighborsRegressor(n_neighbors=25, weights="distance")),
        "decision_tree_classical": tree(DecisionTreeRegressor(max_depth=12, min_samples_leaf=20, random_state=random_state)),
        "random_forest_classical": tree(RandomForestRegressor(n_estimators=250, min_samples_leaf=5, random_state=random_state, n_jobs=-1)),
        "gradient_boosting_classical": tree(GradientBoostingRegressor(random_state=random_state)),
        "hist_gradient_boosting_classical": tree(HistGradientBoostingRegressor(max_iter=220, learning_rate=0.06, random_state=random_state)),
        "mlp_neural_network": scaled(MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            alpha=1e-4,
            max_iter=500,
            early_stopping=True,
            random_state=random_state,
        )),
    }


def photoz_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute standard photometric-redshift evaluation metrics."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    dz = y_pred - y_true
    dz_norm = dz / (1.0 + y_true)
    med = np.median(dz_norm)
    nmad = 1.4826 * np.median(np.abs(dz_norm - med))
    outlier_rate = np.mean(np.abs(dz_norm) > 0.15)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    return {
        "MAE_z": mean_absolute_error(y_true, y_pred),
        "RMSE_z": rmse,
        "bias_mean_dz_norm": float(np.mean(dz_norm)),
        "NMAD_dz_norm": float(nmad),
        "outlier_rate_abs_dz_norm_gt_0p15": float(outlier_rate),
        "R2": r2_score(y_true, y_pred),
    }


def fit_and_evaluate_models(X_train, X_test, y_train, y_test, random_state: int = 42):
    """Fit all models and return predictions plus a metrics table.

    All models are trained on exactly the same train/test split, so the
    resulting metrics form a fair side-by-side comparison.  This is the central
    table for the ML/photo-z part of the project.
    """
    rows = []
    predictions = pd.DataFrame({"spec_z": y_test.to_numpy()}, index=y_test.index)
    fitted = {}
    for name, model in make_models(random_state=random_state).items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        predictions[name] = y_pred
        metric_row = {"model": name}
        metric_row.update(photoz_metrics(y_test, y_pred))
        rows.append(metric_row)
        fitted[name] = model
    metrics = pd.DataFrame(rows).sort_values("NMAD_dz_norm").reset_index(drop=True)
    return fitted, predictions, metrics


def add_catalog_photoz_metrics(
    df_clean: pd.DataFrame,
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    target_col: str = "spec_z",
    catalog_col: str = "catalog_photoz",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Add an already-existing catalogue photo-z estimate to the comparison.

    Some datasets contain both a spectroscopic redshift and a catalogue
    photometric-redshift estimate.  When present, this function evaluates that
    column on the same test-set indices as the trained models.
    """
    if catalog_col not in df_clean.columns:
        return predictions, metrics

    aligned = pd.to_numeric(df_clean.loc[predictions.index, catalog_col], errors="coerce")
    valid = aligned.notna() & predictions[target_col].notna()
    if valid.sum() < 3:
        return predictions, metrics

    predictions = predictions.copy()
    predictions["catalog_photoz_reference"] = aligned
    row = {"model": "catalog_photoz_reference"}
    row.update(photoz_metrics(predictions.loc[valid, target_col], predictions.loc[valid, "catalog_photoz_reference"]))
    metrics = pd.concat([metrics, pd.DataFrame([row])], ignore_index=True)
    metrics = metrics.sort_values("NMAD_dz_norm").reset_index(drop=True)
    return predictions, metrics


def model_prediction_columns(predictions: pd.DataFrame) -> list[str]:
    """Return prediction columns, excluding the spectroscopic ground truth."""
    return [
        c for c in predictions.columns
        if c != "spec_z" and not c.endswith("_sigma") and not c.endswith("_uncertainty")
    ]


def residual_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return per-object residuals for every model in long format."""
    rows = []
    y_true = predictions["spec_z"].astype(float)
    for model in model_prediction_columns(predictions):
        y_pred = pd.to_numeric(predictions[model], errors="coerce")
        dz = y_pred - y_true
        dz_norm = dz / (1.0 + y_true)
        tmp = pd.DataFrame({
            "model": model,
            "spec_z": y_true,
            "pred_z": y_pred,
            "dz": dz,
            "dz_norm": dz_norm,
            "abs_dz_norm": dz_norm.abs(),
        })
        rows.append(tmp)
    return pd.concat(rows, ignore_index=True)


def binned_residual_summary(
    predictions: pd.DataFrame,
    bins: Iterable[float] | None = None,
) -> pd.DataFrame:
    """Summarize normalized residuals in spectroscopic-redshift bins."""
    if bins is None:
        bins = np.linspace(0.0, max(0.6, float(predictions["spec_z"].max())), 7)
    long = residual_table(predictions).dropna(subset=["spec_z", "dz_norm"])
    long["z_bin"] = pd.cut(long["spec_z"], bins=bins, include_lowest=True)
    summary = (
        long.groupby(["model", "z_bin"], observed=True)
        .agg(
            n=("dz_norm", "size"),
            median_dz_norm=("dz_norm", "median"),
            mean_abs_dz_norm=("abs_dz_norm", "mean"),
            nmad_dz_norm=("dz_norm", lambda x: 1.4826 * np.median(np.abs(x - np.median(x)))),
        )
        .reset_index()
    )
    summary["z_bin"] = summary["z_bin"].astype(str)
    return summary


def random_forest_tree_uncertainty(model: Pipeline, X_test: pd.DataFrame) -> np.ndarray:
    """Estimate uncertainty from the scatter among trees in a random forest.

    This is not a full Bayesian uncertainty, but it is a useful project-level
    diagnostic: if the forest's individual trees disagree strongly, the object
    may lie in a difficult or sparsely sampled region of colour space.
    """
    rf = model.named_steps.get("model")
    imputer = model.named_steps.get("imputer")
    if rf is None or not hasattr(rf, "estimators_"):
        raise TypeError("Expected a fitted RandomForestRegressor pipeline.")
    X_imp = imputer.transform(X_test) if imputer is not None else X_test
    tree_preds = np.vstack([tree.predict(X_imp) for tree in rf.estimators_])
    return tree_preds.std(axis=0)


def fit_mlp_ensemble_uncertainty(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    n_members: int = 8,
    base_random_state: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """Train a small ensemble of MLPs and return mean prediction and std.

    This mirrors the idea of uncertainty-aware photo-z modelling without adding
    TensorFlow as a dependency.  Different random initializations produce a
    distribution of predictions; the standard deviation is used as an empirical
    model-uncertainty proxy.
    """
    preds = []
    for k in range(n_members):
        seed = base_random_state + k
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            alpha=1e-4,
            max_iter=500,
            early_stopping=True,
            random_state=seed,
        )),
        ])
        model.fit(X_train, y_train)
        preds.append(model.predict(X_test))
    arr = np.vstack(preds)
    return arr.mean(axis=0), arr.std(axis=0)


def uncertainty_diagnostics(y_true: np.ndarray, y_pred: np.ndarray, sigma: np.ndarray) -> Dict[str, float]:
    """Evaluate whether larger estimated uncertainty corresponds to larger error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    abs_err = np.abs(y_pred - y_true)
    valid = np.isfinite(abs_err) & np.isfinite(sigma) & (sigma >= 0)
    if valid.sum() < 3:
        return {"mean_sigma": np.nan, "median_sigma": np.nan, "corr_abs_error_sigma": np.nan}
    corr = np.corrcoef(abs_err[valid], sigma[valid])[0, 1]
    return {
        "mean_sigma": float(np.mean(sigma[valid])),
        "median_sigma": float(np.median(sigma[valid])),
        "corr_abs_error_sigma": float(corr),
    }


def save_predictions_and_metrics(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    output_dir: str | Path,
    binned_summary: pd.DataFrame | None = None,
    uncertainty_summary: pd.DataFrame | None = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "photoz_predictions.csv", index=True)
    metrics.to_csv(output_dir / "photoz_model_comparison.csv", index=False)
    residual_table(predictions).to_csv(output_dir / "photoz_residuals_long.csv", index=False)
    if binned_summary is not None:
        binned_summary.to_csv(output_dir / "photoz_binned_residual_summary.csv", index=False)
    if uncertainty_summary is not None:
        uncertainty_summary.to_csv(output_dir / "photoz_uncertainty_summary.csv", index=False)


def plot_model_metric_comparison(metrics: pd.DataFrame, output_dir: str | Path):
    """Save bar-chart comparisons for the main photo-z metrics."""
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_specs = [
        ("NMAD_dz_norm", "lower is better"),
        ("MAE_z", "lower is better"),
        ("RMSE_z", "lower is better"),
        ("outlier_rate_abs_dz_norm_gt_0p15", "lower is better"),
        ("R2", "higher is better"),
    ]
    for metric, subtitle in metric_specs:
        if metric not in metrics.columns:
            continue
        ordered = metrics.sort_values(metric, ascending=(metric != "R2"))
        plt.figure(figsize=(9, 4.8))
        plt.bar(ordered["model"], ordered[metric])
        plt.xticks(rotation=40, ha="right")
        plt.ylabel(metric)
        plt.title(f"Photo-z model comparison: {metric} ({subtitle})")
        plt.tight_layout()
        safe = metric.replace("/", "_").replace(">", "gt").replace("<", "lt")
        plt.savefig(output_dir / f"photoz_metric_{safe}.png", dpi=160)
        plt.close()


def plot_all_predicted_vs_true(predictions: pd.DataFrame, output_dir: str | Path):
    """Save one predicted-vs-true plot per model plus a combined overlay."""
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    y_true = predictions["spec_z"].astype(float)
    lim_max = float(np.nanmax([y_true.max()] + [pd.to_numeric(predictions[c], errors="coerce").max() for c in model_prediction_columns(predictions)]))
    lim = [0, lim_max]

    for model in model_prediction_columns(predictions):
        y_pred = pd.to_numeric(predictions[model], errors="coerce")
        plt.figure(figsize=(6, 6))
        plt.scatter(y_true, y_pred, s=8, alpha=0.35)
        plt.plot(lim, lim, linestyle="--")
        plt.xlabel("Spectroscopic redshift, z_spec")
        plt.ylabel("Predicted photometric redshift, z_pred")
        plt.title(f"Predicted vs true redshift: {model}")
        plt.tight_layout()
        plt.savefig(output_dir / f"photoz_pred_vs_true_{model}.png", dpi=160)
        plt.close()

    plt.figure(figsize=(7, 6))
    for model in model_prediction_columns(predictions):
        y_pred = pd.to_numeric(predictions[model], errors="coerce")
        plt.scatter(y_true, y_pred, s=7, alpha=0.22, label=model)
    plt.plot(lim, lim, linestyle="--")
    plt.xlabel("Spectroscopic redshift, z_spec")
    plt.ylabel("Predicted photometric redshift, z_pred")
    plt.title("All photo-z models: predicted vs true redshift")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(output_dir / "photoz_all_models_predicted_vs_true.png", dpi=160)
    plt.close()


def plot_residual_comparisons(predictions: pd.DataFrame, output_dir: str | Path):
    """Save normalized residual plots for every model."""
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    y_true = predictions["spec_z"].astype(float)

    for model in model_prediction_columns(predictions):
        y_pred = pd.to_numeric(predictions[model], errors="coerce")
        dz_norm = (y_pred - y_true) / (1.0 + y_true)
        plt.figure(figsize=(7, 4.8))
        plt.scatter(y_true, dz_norm, s=8, alpha=0.35)
        plt.axhline(0, linestyle="--")
        plt.axhline(0.15, linestyle=":")
        plt.axhline(-0.15, linestyle=":")
        plt.xlabel("Spectroscopic redshift, z_spec")
        plt.ylabel("(z_pred - z_spec) / (1 + z_spec)")
        plt.title(f"Normalized residuals: {model}")
        plt.tight_layout()
        plt.savefig(output_dir / f"photoz_residuals_{model}.png", dpi=160)
        plt.close()

    plt.figure(figsize=(8, 5))
    for model in model_prediction_columns(predictions):
        y_pred = pd.to_numeric(predictions[model], errors="coerce")
        dz_norm = (y_pred - y_true) / (1.0 + y_true)
        plt.hist(dz_norm.dropna(), bins=60, histtype="step", density=True, label=model)
    plt.xlabel("Normalized residual")
    plt.ylabel("Density")
    plt.title("Residual distribution comparison")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(output_dir / "photoz_residual_histogram_comparison.png", dpi=160)
    plt.close()


def plot_binned_residual_summary(summary: pd.DataFrame, output_dir: str | Path):
    """Plot NMAD in redshift bins for all models."""
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pivot = summary.pivot(index="z_bin", columns="model", values="nmad_dz_norm")
    plt.figure(figsize=(9, 5))
    for model in pivot.columns:
        plt.plot(range(len(pivot.index)), pivot[model], marker="o", label=model)
    plt.xticks(range(len(pivot.index)), pivot.index, rotation=35, ha="right")
    plt.ylabel("NMAD of normalized residuals")
    plt.xlabel("Spectroscopic-redshift bin")
    plt.title("Photo-z performance as a function of redshift")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(output_dir / "photoz_nmad_by_redshift_bin.png", dpi=160)
    plt.close()

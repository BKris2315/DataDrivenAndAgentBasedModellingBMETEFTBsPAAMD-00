"""Run the SDSS photometric-redshift model comparison from the command line."""

from pathlib import Path

import pandas as pd

from src.photoz import (
    PHOTOZ_FEATURES,
    add_catalog_photoz_metrics,
    binned_residual_summary,
    clean_photoz_sample,
    fit_and_evaluate_models,
    fit_mlp_ensemble_uncertainty,
    load_photoz_csv,
    photoz_metrics,
    plot_all_predicted_vs_true,
    plot_binned_residual_summary,
    plot_model_metric_comparison,
    plot_residual_comparisons,
    photoz_sample_summary,
    random_forest_tree_uncertainty,
    save_predictions_and_metrics,
    stratified_photoz_sample,
    train_test_photoz,
    uncertainty_diagnostics,
)

RAW_CANDIDATES = [
    Path("data/raw/sdss_photoz_training_sample.csv"),
    Path("data/raw/PhotoZ_SDSS.csv"),
]
PROCESSED = Path("data/processed/sdss_photoz_clean.csv")
MODELING_SAMPLE = Path("data/processed/sdss_photoz_modeling_sample.csv")
REPORT = Path("report")
FIGURES = Path("figures")
MAX_MODEL_ROWS = 50000


def main():
    raw_path = next((path for path in RAW_CANDIDATES if path.exists()), None)
    if raw_path is None:
        raise FileNotFoundError(
            "Missing SDSS photo-z CSV. Run python download_sdss_photoz_sample.py first, "
            "or manually save a CSV as data/raw/sdss_photoz_training_sample.csv or data/raw/PhotoZ_SDSS.csv."
        )
    FIGURES.mkdir(exist_ok=True)
    REPORT.mkdir(exist_ok=True)
    PROCESSED.parent.mkdir(parents=True, exist_ok=True)

    raw = load_photoz_csv(raw_path)
    clean = clean_photoz_sample(raw)
    clean.to_csv(PROCESSED, index=False)
    modeling = stratified_photoz_sample(clean, max_rows=MAX_MODEL_ROWS)
    modeling.to_csv(MODELING_SAMPLE, index=False)
    pd.DataFrame([
        photoz_sample_summary(raw.dropna(subset=["spec_z"]), "raw_loaded"),
        photoz_sample_summary(clean, "clean_after_quality_cuts"),
        photoz_sample_summary(modeling, f"modeling_stratified_max_{MAX_MODEL_ROWS}"),
    ]).to_csv(REPORT / "photoz_sample_summary.csv", index=False)

    X_train, X_test, y_train, y_test = train_test_photoz(modeling, PHOTOZ_FEATURES)
    fitted, predictions, metrics = fit_and_evaluate_models(X_train, X_test, y_train, y_test)
    predictions, metrics = add_catalog_photoz_metrics(modeling, predictions, metrics)

    binned = binned_residual_summary(predictions)

    uncertainty_rows = []
    if "random_forest_classical" in fitted:
        sigma_rf = random_forest_tree_uncertainty(fitted["random_forest_classical"], X_test)
        predictions["random_forest_tree_sigma"] = sigma_rf
        row = {"method": "random_forest_tree_scatter"}
        row.update(uncertainty_diagnostics(y_test, predictions["random_forest_classical"], sigma_rf))
        uncertainty_rows.append(row)

    # MLP ensemble is useful for an uncertainty-aware extension, but can be slow
    # on very large samples. Use it by default only for moderate data sizes.
    if len(X_train) <= 50000:
        mlp_mean, mlp_sigma = fit_mlp_ensemble_uncertainty(X_train, X_test, y_train, n_members=4)
        predictions["mlp_ensemble_mean"] = mlp_mean
        predictions["mlp_ensemble_sigma"] = mlp_sigma
        ensemble_row = {"model": "mlp_ensemble_mean"}
        ensemble_row.update(photoz_metrics(y_test, mlp_mean))
        metrics = pd.concat([metrics, pd.DataFrame([ensemble_row])], ignore_index=True)
        metrics = metrics.sort_values("NMAD_dz_norm").reset_index(drop=True)
        row = {"method": "mlp_ensemble_std"}
        row.update(uncertainty_diagnostics(y_test, mlp_mean, mlp_sigma))
        uncertainty_rows.append(row)

    uncertainty_summary = pd.DataFrame(uncertainty_rows)

    save_predictions_and_metrics(predictions, metrics, REPORT, binned, uncertainty_summary)
    plot_model_metric_comparison(metrics, FIGURES)
    plot_all_predicted_vs_true(predictions, FIGURES)
    plot_residual_comparisons(predictions, FIGURES)
    plot_binned_residual_summary(binned, FIGURES)

    print("\nModel comparison sorted by NMAD:")
    print(metrics.to_string(index=False))
    if not uncertainty_summary.empty:
        print("\nUncertainty diagnostics:")
        print(uncertainty_summary.to_string(index=False))
    print("\nSaved comparison tables to report/ and plots to figures/.")


if __name__ == "__main__":
    main()

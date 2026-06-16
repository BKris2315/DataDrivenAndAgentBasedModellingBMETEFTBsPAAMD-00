"""Run the full official-data analysis after downloading Pantheon+ and SDSS files."""
from pathlib import Path
import pandas as pd

from src.pantheon import (
    bootstrap_h0,
    clean_pantheon,
    fit_results_table,
    h0_for_lower_redshift_cuts,
    h0_for_redshift_bins,
    h0_for_redshift_cuts,
    load_pantheon,
    save_cleaned,
)
from src.plotting import plot_hubble_diagram, plot_mu_vs_redshift, plot_residuals, plot_h0_by_redshift_cut, plot_sdss_spectrum_with_lines, plot_sdss_restframe
from src.sdss_spectrum import load_sdss_spectrum, find_candidate_emission_peaks, match_peaks_to_lines, refine_line_centers, weighted_mean_redshift

Path("report").mkdir(exist_ok=True)
Path("figures").mkdir(exist_ok=True)

raw = load_pantheon()
clean, summary = clean_pantheon(raw)
save_cleaned(clean)
summary.to_frame().to_csv("report/pantheon_cleaning_summary.csv", index=False)
distance_cols = ["distance_mpc", "hubble_distance_mpc"]
sample_methods = ["origin", "free_intercept", "huber", "weighted_origin"]
fit_results_table(clean, distance_cols=distance_cols).to_csv("report/pantheon_fit_results.csv", index=False)
h0_for_redshift_cuts(clean, distance_col="hubble_distance_mpc", methods=sample_methods).to_csv("report/pantheon_h0_by_redshift_cut.csv", index=False)
h0_for_redshift_bins(clean, distance_col="hubble_distance_mpc", methods=["origin", "weighted_origin"]).to_csv("report/pantheon_h0_by_redshift_bin.csv", index=False)
h0_for_lower_redshift_cuts(clean, distance_col="hubble_distance_mpc", methods=sample_methods).to_csv("report/pantheon_h0_by_lower_redshift_cut.csv", index=False)
pd.concat([
    h0_for_redshift_cuts(clean, distance_col="distance_mpc", methods=["origin"]),
    h0_for_redshift_cuts(clean, distance_col="hubble_distance_mpc", methods=["origin"]),
], ignore_index=True).to_csv("report/pantheon_h0_distance_definition_by_cut.csv", index=False)
pd.DataFrame([
    bootstrap_h0(clean, method="origin", distance_col="hubble_distance_mpc", n_bootstrap=1000),
    bootstrap_h0(clean, method="weighted_origin", distance_col="hubble_distance_mpc", n_bootstrap=1000),
]).to_csv("report/pantheon_h0_bootstrap.csv", index=False)
calibrator_rows = []
for exclude_calibrators in [True, False]:
    variant, _ = clean_pantheon(raw, z_min=0.01, z_max=0.15, exclude_calibrators=exclude_calibrators)
    table = fit_results_table(variant, distance_cols=["hubble_distance_mpc"], methods=["origin", "weighted_origin"])
    table["calibrator_choice"] = "calibrators excluded" if exclude_calibrators else "calibrators included"
    table["n_clean"] = len(variant)
    calibrator_rows.append(table)
pd.concat(calibrator_rows, ignore_index=True).to_csv("report/pantheon_h0_calibrator_choice.csv", index=False)
plot_mu_vs_redshift(clean)
plot_hubble_diagram(clean, distance_col="hubble_distance_mpc")
plot_residuals(clean, distance_col="hubble_distance_mpc")
plot_h0_by_redshift_cut(pd.read_csv("report/pantheon_h0_by_redshift_cut.csv"))

spec, meta = load_sdss_spectrum()
peaks = find_candidate_emission_peaks(spec)
matches = match_peaks_to_lines(peaks, z_min=0.05, z_max=0.8)
z_initial = matches["z_candidate"].median()
lines = refine_line_centers(spec, z_initial)
z_mean, z_scatter = weighted_mean_redshift(lines)
peaks.to_csv("report/sdss_candidate_peaks.csv", index=False)
matches.to_csv("report/sdss_peak_line_matches.csv", index=False)
lines.to_csv("report/sdss_fitted_lines.csv", index=False)
pd.DataFrame([{**meta, "z_measured": z_mean, "z_line_scatter": z_scatter}]).to_csv("report/sdss_redshift_summary.csv", index=False)
plot_sdss_spectrum_with_lines(spec, lines, z_mean)
plot_sdss_restframe(spec, z_mean)

print("Analysis complete. See report/ and figures/.")

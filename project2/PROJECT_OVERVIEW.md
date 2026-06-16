# Project Overview: Measuring Cosmic Expansion with Pantheon+ and SDSS

## 1. Project aim

This project uses public astronomical data to study two connected questions:

1. Can official supernova distance-redshift data reproduce the local Hubble expansion relation?
2. How well can redshift be estimated from SDSS photometry when spectra are not available?

The project combines three related analyses:

- a low-redshift Hubble-law fit using the official Pantheon+SH0ES Type Ia supernova distance table;
- a sample-dependence study showing how the fitted expansion rate changes under different redshift cuts and fit choices;
- SDSS redshift analyses, including one direct spectrum-based redshift measurement and one machine-learning photometric-redshift comparison.

The main idea is that redshift can be measured directly from spectra, while distance is harder. Type Ia supernovae provide calibrated distances, so they are useful for testing the distance-redshift relation. Photometric redshift models then show what happens when only broadband colours are available instead of spectra.

## 2. Data sources

### Pantheon+SH0ES supernova data

The main cosmology analysis uses:

```text
data/raw/Pantheon+SH0ES.dat
```

Important columns include:

- `zHD`: Hubble-diagram redshift;
- `MU_SH0ES`: calibrated distance modulus;
- `MU_SH0ES_ERR_DIAG`: diagonal distance-modulus uncertainty;
- `IS_CALIBRATOR`: flag identifying Cepheid calibrator objects.

After preprocessing, the cleaned sample contains 705 low-redshift non-calibrator supernovae with:

- `0.01016 <= zHD <= 0.1494`;
- converted luminosity distance `distance_mpc`;
- local Hubble-distance proxy `hubble_distance_mpc = D_L/(1+z)`;
- approximate recession velocity `velocity_km_s = c zHD`.

### SDSS spectrum

The spectrum notebook uses the SDSS FITS spectrum:

```text
data/raw/spec-0532-51993-0497.fits
```

It demonstrates direct redshift measurement from spectral-line shifts. The FITS catalog redshift is approximately `z = 0.30949`. The current automated line matching gives a line-system estimate near `z = 0.18517`, so this part should be discussed as a useful method demonstration and a warning that automatic line identification can fail without careful physical checks.

### SDSS photometric-redshift sample

The photo-z analysis accepts either:

```text
data/raw/sdss_photoz_training_sample.csv
data/raw/PhotoZ_SDSS.csv
```

The included `PhotoZ_SDSS.csv` table contains 1,000,000 rows. After quality cuts, 983,821 rows remain. The modelling step uses a reproducible redshift-stratified sample of 50,000 rows so that notebooks remain fast and report files stay manageable.

## 3. Pantheon+ preprocessing

The preprocessing notebook performs these steps:

1. Load the official whitespace-separated Pantheon+SH0ES table.
2. Keep the columns needed for redshift, distance modulus, uncertainty, calibration flags, and sky position.
3. Convert placeholder missing values such as `-9`, `-99`, and `-999` to missing values.
4. Drop rows missing redshift or distance modulus.
5. Exclude Cepheid calibrators for the main Hubble-flow fit.
6. Keep the low-redshift range `0.01 <= zHD <= 0.15`.
7. Convert distance modulus to luminosity distance:

   ```text
   D_L(Mpc) = 10^((mu - 25)/5)
   ```

8. Convert redshift to approximate recession velocity:

   ```text
   v = c z
   ```

9. Also compute:

   ```text
   d_Hubble = D_L / (1 + z)
   ```

This final distance proxy is important. Pantheon+ reports luminosity distance, but the simple local linear Hubble-law fit is better represented by `D_L/(1+z)` than by raw `D_L`, especially near the upper edge of `z <= 0.15`.

## 4. Hubble-law model

The educational Hubble-law model is:

```text
v = H0 d
```

where:

- `v` is approximated as `c zHD`;
- `d` is either raw luminosity distance `D_L` or the local Hubble-distance proxy `D_L/(1+z)`;
- the slope `H0` is the fitted expansion rate in `km/s/Mpc`.

The project compares several fitting choices:

| Method | Purpose |
|---|---|
| Ordinary least squares through origin | Main simple Hubble-law fit, assuming zero distance implies zero velocity |
| Ordinary least squares with free intercept | Diagnostic for offsets, local flows, or model mismatch |
| Huber robust regression | Checks whether a few high-residual objects dominate the result |
| Weighted least squares through origin | Uses distance-modulus uncertainties as approximate relative weights |
| Weighted least squares with free intercept | Weighted diagnostic fit with intercept |
| Bootstrap resampling | Estimates sampling sensitivity by refitting resampled supernova lists |

## 5. Hubble-law results

The main fit table is saved as:

```text
report/pantheon_fit_results.csv
```

Important results:

| Distance definition | Fit method | H0 (km/s/Mpc) | Notes |
|---|---:|---:|---|
| `D_L` | forced-origin OLS | 67.02 | Raw luminosity distance gives a lower slope |
| `D_L` | Huber robust | 65.65 | Robust raw-distance fit |
| `D_L/(1+z)` | forced-origin OLS | 73.88 | Preferred simple local-distance fit |
| `D_L/(1+z)` | free-intercept OLS | 73.94 | Very similar slope, small intercept |
| `D_L/(1+z)` | Huber robust | 74.75 | Robust diagnostic |
| `D_L/(1+z)` | weighted forced-origin | 74.57 | Uses diagonal distance-modulus uncertainty weights |

The bootstrap table is saved as:

```text
report/pantheon_h0_bootstrap.csv
```

For the preferred forced-origin `D_L/(1+z)` fit:

- point estimate: `H0 = 73.88 km/s/Mpc`;
- bootstrap median: `73.90 km/s/Mpc`;
- 68 percent interval: `73.52` to `74.23 km/s/Mpc`;
- 95 percent interval: `73.18` to `74.56 km/s/Mpc`.

These intervals only represent the simplified bootstrap sampling test. They are not a full Pantheon+ cosmological uncertainty because the full covariance matrix and full cosmological model are not fitted.

## 6. Sample-dependence analysis

The sample-dependence notebook asks whether the fitted slope is stable under reasonable data-selection choices.

### Cumulative redshift cuts

The project fits all supernovae below several maximum redshifts:

```text
z <= 0.03, 0.05, 0.08, 0.10, 0.15
```

The comparison makes the distance-definition issue clear:

| Maximum z | H0 using `D_L` | H0 using `D_L/(1+z)` |
|---:|---:|---:|
| 0.03 | 71.60 | 73.27 |
| 0.05 | 70.58 | 72.91 |
| 0.08 | 69.86 | 73.24 |
| 0.10 | 69.48 | 73.21 |
| 0.15 | 67.02 | 73.88 |

Raw luminosity distance produces a visible downward drift as higher-redshift objects are included. The `D_L/(1+z)` proxy gives a much more stable local Hubble-law slope.

### Independent redshift bins

The project also fits independent redshift bins. This is a different question from cumulative cuts, because each bin contains a separate subset of objects. The bins are noisier, but they show whether one redshift range behaves unusually.

### Lower redshift cuts

The project tests the effect of removing the nearest objects. This matters because nearby galaxies can have peculiar velocities of hundreds of `km/s`, which are a large fraction of `cz` at very low redshift.

### Calibrator inclusion

The main analysis excludes Cepheid calibrator rows because calibrators serve a different role from Hubble-flow objects. A check with calibrators included gives nearly the same simplified `D_L/(1+z)` forced-origin result:

- calibrators excluded: `H0 = 73.88`;
- calibrators included: `H0 = 73.88`.

This supports the conclusion that the main low-redshift result is not being driven by that particular filtering choice.

## 7. SDSS spectrum redshift method

The spectrum notebook demonstrates direct redshift measurement:

1. Load wavelength and flux arrays from the SDSS FITS file.
2. Remove the smooth continuum using a median-filter style subtraction.
3. Detect candidate emission peaks.
4. Match observed peaks to known rest-frame lines.
5. Refine line centers with Gaussian fits.
6. Compute redshift from:

   ```text
   z = lambda_observed / lambda_rest - 1
   ```

This is the physically direct way to measure redshift. However, the current automated match finds a line system near `z = 0.18517`, while the FITS catalog redshift is `z = 0.30949`. This discrepancy should be treated as an important caveat: automatic peak matching can lock onto an incorrect line system, especially for complex spectra or objects classified as QSO.

## 8. Photometric-redshift problem

Spectroscopic redshift is precise because it uses shifted spectral lines. Photometric redshift is less precise because it uses only broadband magnitudes.

The photo-z notebook tries to learn:

```text
(u, g, r, i, zmag, u-g, g-r, r-i, i-z, r_mag) -> z_spec
```

The target `z_spec` is the SDSS spectroscopic redshift. The model prediction is a photometric redshift estimate.

Feature engineering:

- raw SDSS model magnitudes: `u`, `g`, `r`, `i`, `zmag`;
- colour indices: `u_g`, `g_r`, `r_i`, `i_z`;
- apparent-brightness proxy: `r_mag`.

Cleaning includes:

- galaxy-only rows when a `class` column exists;
- finite magnitudes;
- plausible magnitude range;
- `0.01 <= spec_z <= 0.6`;
- redshift-error filtering when available;
- removal of extreme colour outliers.

## 9. Photometric-redshift models

The project compares these models:

| Model | Type | Reason for inclusion |
|---|---|---|
| Median redshift baseline | Constant baseline | Tests whether models beat simply predicting a typical redshift |
| Ridge regression | Linear model | Simple interpretable baseline |
| k-nearest neighbours | Classical non-parametric method | Similar colours often imply similar redshift |
| Decision tree | Single nonlinear tree | Easy to understand but can overfit |
| Random forest | Tree ensemble | Reduces instability of one tree |
| Gradient boosting | Boosted tree ensemble | Strong tabular baseline |
| Histogram gradient boosting | Efficient boosted tree ensemble | Often performs well on large tabular data |
| MLPRegressor | Neural-network style model | Tests a simple learned nonlinear mapping |
| MLP ensemble mean | Ensemble uncertainty extension | Averages several MLPs and estimates scatter |

The models are evaluated on the same held-out test set from the same redshift-stratified modelling sample.

## 10. Photo-z evaluation metrics

The project reports:

| Metric | Meaning |
|---|---|
| `MAE_z` | Mean absolute redshift error |
| `RMSE_z` | Root mean squared redshift error |
| `bias_mean_dz_norm` | Mean normalized residual |
| `NMAD_dz_norm` | Robust scatter of normalized residuals |
| `outlier_rate_abs_dz_norm_gt_0p15` | Fraction with catastrophic normalized error above 0.15 |
| `R2` | Standard regression coefficient of determination |

The normalized residual is:

```text
dz_norm = (z_pred - z_spec) / (1 + z_spec)
```

In photo-z work, NMAD, bias, and outlier rate are usually more informative than R2 alone.

## 11. Photo-z results

The model comparison table is saved as:

```text
report/photoz_model_comparison.csv
```

Current ranking by NMAD:

| Model | MAE | RMSE | NMAD | Outlier rate | R2 |
|---|---:|---:|---:|---:|---:|
| Random forest | 0.01796 | 0.02654 | 0.01634 | 0.00168 | 0.87749 |
| MLP ensemble mean | 0.01797 | 0.02644 | 0.01656 | 0.00136 | 0.87846 |
| Histogram gradient boosting | 0.01832 | 0.02671 | 0.01703 | 0.00152 | 0.87593 |
| kNN | 0.01904 | 0.02775 | 0.01732 | 0.00144 | 0.86610 |
| MLP neural network | 0.01883 | 0.02827 | 0.01733 | 0.00200 | 0.86098 |
| Decision tree | 0.01995 | 0.02925 | 0.01814 | 0.00192 | 0.85117 |
| Gradient boosting | 0.01973 | 0.02829 | 0.01861 | 0.00160 | 0.86083 |
| Ridge regression | 0.02680 | 0.03691 | 0.02772 | 0.00248 | 0.76302 |
| Median baseline | 0.06144 | 0.07736 | 0.06756 | 0.00000 | -0.04069 |

The key interpretation is that the trained photo-z models clearly beat the median baseline, so the SDSS colours contain real redshift information. The best model by NMAD is the random forest. The MLP ensemble is very close and has slightly better RMSE/R2 in this run, but the random forest is simpler to explain and is the cleanest main result.

## 12. Uncertainty diagnostics

The project includes two simple uncertainty proxies:

1. random-forest tree scatter;
2. MLP ensemble scatter.

These are not calibrated Bayesian uncertainties. They are diagnostic checks: if the uncertainty proxy is useful, it should correlate with absolute prediction error.

Current results:

| Method | Mean sigma | Median sigma | Correlation with absolute error |
|---|---:|---:|---:|
| Random-forest tree scatter | 0.01726 | 0.01502 | 0.50169 |
| MLP ensemble standard deviation | 0.00491 | 0.00372 | 0.32954 |

The random-forest scatter is the more useful uncertainty proxy in this run.

## 13. Main conclusions

The Pantheon+ analysis shows the expected Hubble expansion trend: larger distances correspond to larger redshift-derived recession velocities. The best simple local fit uses `D_L/(1+z)` and gives `H0` near `74 km/s/Mpc`.

The sample-dependence study is central to the project. It shows that the fitted value is not just a property of a formula; it depends on distance definition, redshift range, and fitting method. Using raw luminosity distance makes the simple slope drift with redshift, while `D_L/(1+z)` gives a more stable educational local-Hubble result.

The SDSS spectrum analysis demonstrates the physically direct redshift method, but the current automated line-matching result does not agree with the catalog redshift. This should be discussed honestly as a limitation of automatic line identification.

The photometric-redshift analysis shows that SDSS broadband colours can predict approximate galaxy redshift, but photo-z estimates are not substitutes for spectroscopic redshifts. They have scatter, outliers, and model-dependent uncertainty.

## 14. Important limitations

- The Hubble-law fit is educational and simplified. It does not fit a full cosmological luminosity-distance model.
- The analysis uses only diagonal distance-modulus uncertainties, not the full Pantheon+ covariance matrix.
- The velocity approximation `v = cz` is only a low-redshift approximation.
- The bootstrap intervals are sampling diagnostics, not professional cosmological error bars.
- The SDSS spectrum line matching needs manual line-identification review because the automated measured redshift differs from the catalog value.
- The photo-z models are empirical and depend on the training sample. They may not generalize outside the redshift, magnitude, and colour range represented in the data.

## 15. Reproducible workflow

Run the Pantheon+ and SDSS spectrum analysis:

```bash
python run_official_analysis.py
```

Run the SDSS photo-z model comparison:

```bash
python run_photoz_analysis.py
```

Read the notebooks in this order:

```text
notebooks/01_data_preprocessing_pantheon.ipynb
notebooks/02_hubble_law_fit_pantheon.ipynb
notebooks/03_sample_dependence_pantheon.ipynb
notebooks/04_sdss_spectrum_redshift_analysis.ipynb
notebooks/05_sdss_photometric_redshift_prediction.ipynb
notebooks/06_sdss_photometric_redshift_comparison.ipynb
```

Key outputs are written to:

```text
report/
figures/
data/processed/
```


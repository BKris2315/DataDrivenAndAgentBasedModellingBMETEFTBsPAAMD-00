# Measuring cosmic expansion with official Pantheon+ and SDSS data

This project estimates the local expansion rate of the Universe from the official Pantheon+SH0ES Type Ia supernova dataset and includes two SDSS redshift analyses:

1. direct spectroscopic redshift measurement from an SDSS FITS spectrum;
2. photometric redshift prediction from SDSS ugriz magnitudes using classical and machine-learning models.

The project is designed as a class project with readable notebooks. There is only one notebook per analysis; no duplicate executed/non-executed versions are included.

## Project questions

1. How do Type Ia supernova redshift and distance measurements show Hubble expansion?
2. What preprocessing is required before fitting a low-redshift Hubble law?
3. How sensitive is the fitted expansion rate to redshift cuts and fitting choices?
4. Can we measure a redshift directly from an SDSS spectrum and compare it to the SDSS catalog value?
5. If only broadband SDSS photometry is available, how well can classical and ML models predict redshift?

## Official data sources

- **Pantheon+SH0ES**: Type Ia supernova distance table `Pantheon+SH0ES.dat` from the PantheonPlusSH0ES/DataRelease GitHub repository.
- **SDSS DR15 spectrum**: optical spectrum `spec-0532-51993-0497.fits` for the object in the user-provided SkyServer link. SkyServer identifies the object as `SDSS J140740.08+021748.1`, with `SpecObjID = 599115398197045248`, class `QSO`, catalog redshift around `z = 0.309`, plate `532`, MJD `51993`, fiber `497`.
- **SDSS DR17 photometric-redshift training sample**: a reproducible SkyServer SQL query joining `PhotoObj` and `SpecObj` to obtain clean galaxy ugriz model magnitudes and spectroscopic redshifts. This is used for the photometric-redshift model comparison.

## Setup

```bash
pip install -r requirements.txt
python download_official_data.py
python download_sdss_photoz_sample.py
```

If download is blocked, manually place these files in `data/raw/`:

```text
data/raw/Pantheon+SH0ES.dat
data/raw/spec-0532-51993-0497.fits
data/raw/sdss_photoz_training_sample.csv
```

The photo-z downloader prints the official SDSS SkyServer SQL URL, so you can open it in a browser and save the CSV manually if needed. Notebook 05 and `run_photoz_analysis.py` also accept the compatible filename `data/raw/PhotoZ_SDSS.csv`.

## Run the analyses

Pantheon+ and SDSS spectrum analysis:

```bash
python run_official_analysis.py
```

SDSS photometric-redshift model comparison:

```bash
python run_photoz_analysis.py
```

Outputs are written to:

```text
figures/
report/
data/processed/
```

## Notebook order

1. `notebooks/01_data_preprocessing_pantheon.ipynb`
   - loads the official Pantheon+SH0ES table;
   - explains the relevant columns;
   - handles placeholder missing values;
   - removes rows unsuitable for the simple low-redshift Hubble-law fit;
   - converts distance modulus to luminosity distance;
   - writes `data/processed/pantheon_clean.csv`.

2. `notebooks/02_hubble_law_fit_pantheon.ipynb`
   - makes the Hubble diagram;
   - fits `v = H0 d`;
   - compares luminosity distance with the local proxy `D_L/(1+z)`;
   - compares forced-origin, free-intercept, robust, and uncertainty-weighted regression;
   - adds a simple bootstrap sampling-uncertainty check;
   - explains what each result means and why the simple fit has limitations.

3. `notebooks/03_sample_dependence_pantheon.ipynb`
   - repeats the Hubble-law fit under several redshift cuts and bins;
   - compares upper cuts, lower cuts, independent bins, fit methods, and distance definitions;
   - checks whether including Cepheid calibrator rows changes the simplified Hubble-flow result;
   - discusses why nearby objects are noisy because of peculiar velocities;
   - turns the analysis into a real investigation rather than a single line fit.

4. `notebooks/04_sdss_spectrum_redshift_analysis.ipynb`
   - downloads/loads the actual SDSS DR15 FITS spectrum;
   - removes the continuum;
   - detects emission-line peaks;
   - matches them to known rest-frame spectral lines;
   - fits Gaussian centers;
   - computes redshift from wavelengths using `z = lambda_obs/lambda_rest - 1`;
   - compares the measured value with the SDSS catalog redshift.

5. `notebooks/05_sdss_photometric_redshift_prediction.ipynb`
   - downloads/loads an official SDSS galaxy sample with ugriz magnitudes and spectroscopic redshifts, or the compatible `PhotoZ_SDSS.csv` file;
   - explains photometric redshift versus spectroscopic redshift;
   - cleans the sample, removes near-zero/bad redshifts and extreme colour outliers, and creates colour features such as `u-g`, `g-r`, `r-i`, `i-z`;
   - uses a reproducible redshift-stratified modelling sample so the notebook is fast and the report files are not enormous;
   - compares a median baseline with ridge regression, kNN, decision trees, random forests, gradient boosting, histogram gradient boosting, and a simple neural-network style `MLPRegressor`;
   - evaluates models using MAE, RMSE, normalized bias, NMAD, outlier rate, and R².

6. `notebooks/06_sdss_photometric_redshift_comparison.ipynb`
   - audits the saved outputs from notebook 05;
   - ranks models by NMAD and R²;
   - checks redshift-bin dependence and uncertainty-proxy diagnostics;
   - replaces the older experimental notebook that contained stale deep-learning state.

## Important limitations to mention in the report

The Pantheon+ distance moduli are not raw telescope images. They are calibrated supernova distances produced by a scientific pipeline. This is appropriate for a class project because the goal is not to redo supernova light-curve fitting, but to use official distance-redshift data to test Hubble expansion.

The low-redshift approximation `v = c z` is only used for small redshift. For larger redshift, the relation between luminosity distance and redshift is not exactly linear, and a full cosmological model should be fitted.

Pantheon+ reports luminosity distance. The updated notebooks keep that value but use `D_L/(1+z)` as the preferred distance proxy for the simple local linear Hubble-law fit. This makes the redshift-cut sample dependence much clearer, but it is still an educational approximation rather than a full cosmology fit.

A simple slope fit does not reproduce the full modern SH0ES or Planck analyses. It is an educational measurement of the local Hubble law and of how sample selection affects the result.

The photometric-redshift notebook predicts redshift from broadband magnitudes. This is not equivalent to measuring redshift from spectral lines. It is a scalable approximation that should be evaluated against spectroscopic redshifts and discussed in terms of bias, scatter, outliers, and whether the model uncertainty proxy is actually correlated with absolute prediction error.

# Project report template

## Title

Measuring cosmic expansion using official Pantheon+ supernova data and SDSS spectral redshift measurements

## Research question

Can official astronomical data reproduce the Hubble expansion relation, and how sensitive is the measured local expansion rate to preprocessing and sample selection?

## Data

The main dataset is the official Pantheon+SH0ES Type Ia supernova distance table. The spectral demonstration uses one official SDSS DR15 FITS spectrum for the object SDSS J140740.08+021748.1.

## Methods summary

1. Pantheon+ preprocessing:
   - selected redshift and distance-modulus columns;
   - converted placeholder missing values to NaN;
   - dropped rows missing essential quantities;
   - excluded Cepheid calibrator objects for the Hubble-flow fit;
   - applied low-redshift cuts;
   - converted distance modulus to luminosity distance;
   - also computed `D_L/(1+z)` as a local Hubble-distance proxy;
   - converted redshift to approximate velocity with v = cz.

2. Hubble-law fitting:
   - fitted v = H0 d through the origin;
   - compared raw luminosity distance with `D_L/(1+z)`;
   - compared with free-intercept, Huber robust, and uncertainty-weighted fits;
   - added a bootstrap sampling-uncertainty check;
   - inspected residuals.

3. Sample-dependence analysis:
   - repeated the fit under different maximum redshift cuts;
   - repeated the fit in independent redshift bins;
   - tested the effect of removing very nearby objects.
   - compared Cepheid calibrator inclusion/exclusion for the simplified Hubble-flow fit.

4. SDSS redshift analysis:
   - loaded the SDSS FITS spectrum;
   - subtracted the continuum;
   - detected emission peaks;
   - matched peaks to known rest-frame spectral lines;
   - refined line centers with Gaussian fits;
   - computed z = lambda_obs / lambda_rest - 1;
   - compared with the SDSS catalog redshift.

## Expected conclusion structure

After running the notebooks, fill in the numerical values from `report/*.csv`:

- The cleaned Pantheon+ sample contains N = ___ objects after preprocessing.
- Using raw luminosity distance gives H0 = ___ km/s/Mpc, while using `D_L/(1+z)` gives H0 = ___ km/s/Mpc.
- The robust fit gives H0 = ___ km/s/Mpc.
- The bootstrap 68% interval is approximately ___ to ___ km/s/Mpc.
- Under the redshift-cut tests, raw luminosity distance changes from ___ to ___ km/s/Mpc, while `D_L/(1+z)` changes from ___ to ___ km/s/Mpc.
- The SDSS spectrum gives measured z = ___, compared with catalog z = ___; the difference is ___ percent.

## Scientific conclusion

The official data show the expected Hubble expansion trend: larger distances correspond to larger redshift-derived recession velocities. However, the fitted expansion rate is not a single number independent of choices. It depends on cleaning, redshift range, and fitting method. This demonstrates why real cosmological measurements require careful sample selection, uncertainty modeling, and calibration.

The SDSS spectrum analysis verifies the physical origin of redshift by measuring the displacement of spectral lines directly. This connects the single-object spectral measurement to the large-sample redshift-distance relation in the Pantheon+ analysis.


## Photometric redshift model comparison

Explain why photometric redshift is different from spectroscopic redshift. The SDSS spectrum notebook measures redshift directly from shifted spectral lines, while this section predicts redshift from broadband `u`, `g`, `r`, `i`, `z` magnitudes.

Describe the official SDSS SQL query, the cleaning rules, and the feature engineering. In particular, explain why colour features such as `u-g`, `g-r`, `r-i`, and `i-z` are useful.

Include the model-comparison table from `report/photoz_model_comparison.csv`. Discuss which model had the lowest NMAD and whether the residual plot shows bias or high-redshift failures.

Also include `report/photoz_sample_summary.csv` so the reader can see the raw, cleaned, and redshift-stratified modelling sample sizes. Compare the best model with `median_redshift_baseline`; this makes clear that the models are using colour information rather than only predicting a typical redshift.

Suggested conclusion sentence:

> The photometric-redshift models can recover an approximate redshift trend from SDSS colours, but their scatter and outliers show why photometric redshifts are not a substitute for spectroscopic measurements when precise redshifts are required.

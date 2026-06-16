# Official data notes

## Pantheon+SH0ES

The main Hubble-law part of this project uses the official Pantheon+SH0ES Type Ia supernova distance table, `Pantheon+SH0ES.dat`, from the PantheonPlusSH0ES public data release.

The project does not treat the Pantheon+ file as a mysterious CSV. The preprocessing notebook explains:

- which redshift column is used;
- which distance-modulus column is used;
- how missing-value placeholders are handled;
- why low-redshift cuts are applied;
- how distance modulus is converted to luminosity distance;
- why the local linear Hubble-law fit also compares `D_L/(1+z)` as a distance proxy;
- why the simple relation `v = cz` is only a low-redshift approximation.

## SDSS spectrum

The spectroscopic redshift notebook uses the user-supplied SDSS SkyServer object. The actual spectrum is the DR15 FITS file for plate 532, MJD 51993, fiber 497. The analysis reads the wavelength and flux arrays, subtracts a smooth continuum, detects line peaks, fits line centers, and computes redshift from shifted wavelengths.

This is the most physically direct redshift measurement in the project.

## SDSS photometric-redshift sample

The photometric-redshift notebook uses an official SDSS SkyServer SQL query. It joins photometric objects with spectroscopic objects so that each row contains:

- SDSS `u`, `g`, `r`, `i`, `z` model magnitudes;
- magnitude uncertainties;
- a trusted SDSS spectroscopic redshift used as the target label.

This analysis is inspired by small classical photo-z examples such as decision-tree redshift prediction and ML examples based on SDSS ugriz magnitudes. The project implements its own transparent comparison rather than copying those repositories.

The current notebook also accepts the compatible `PhotoZ_SDSS.csv` file present in this folder. In that file, the column named `photometric_z` behaves like the SDSS z-band magnitude, while `redshift` is normalized to the spectroscopic target `spec_z`.

For runtime and interpretability, notebook 05 saves the full cleaned table and then trains models on a reproducible redshift-stratified modelling sample. This keeps the comparison fair without creating enormous notebook outputs.

The key conceptual distinction is:

- **spectroscopic redshift**: measured from wavelength shifts in a spectrum;
- **photometric redshift**: predicted from broadband colours and calibrated using spectroscopic training data.

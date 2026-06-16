Official raw data files are downloaded here.

Run from the project root:

    python download_official_data.py

Expected files:

- Pantheon+SH0ES.dat
- spec-0532-51993-0497.fits
- sdss_photoz_training_sample.csv, or PhotoZ_SDSS.csv for the compatible public-style SDSS photo-z table

Manual downloads:

- Pantheon+SH0ES.dat: PantheonPlusSH0ES/DataRelease GitHub repository, Pantheon+_Data/4_DISTANCES_AND_COVAR/.
- SDSS spectrum: https://dr15.sdss.org/sas/dr15/sdss/spectro/redux/26/spectra/lite/0532/spec-0532-51993-0497.fits
- SDSS photo-z sample: run `python download_sdss_photoz_sample.py` to print the official SkyServer SQL URL and save the result.

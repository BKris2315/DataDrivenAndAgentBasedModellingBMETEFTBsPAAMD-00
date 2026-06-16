"""Download an official SDSS ugriz + spectroscopic-redshift sample.

This script queries the official SDSS SkyServer SQL endpoint and saves a CSV
that can be used in notebook 05 for photometric-redshift prediction.

If the automatic download fails, open the printed URL in a browser, download the
CSV result manually, and save it as:

    data/raw/sdss_photoz_training_sample.csv

The notebooks also accept the existing public-style filename:

    data/raw/PhotoZ_SDSS.csv
"""

from pathlib import Path

from src.photoz import DEFAULT_SQL, download_sdss_photoz_sample, sdss_sql_url

OUTPUT = Path("data/raw/sdss_photoz_training_sample.csv")

if __name__ == "__main__":
    print("Official SDSS SQL query URL:")
    print(sdss_sql_url(DEFAULT_SQL, data_release="dr17"))
    print()
    path = download_sdss_photoz_sample(OUTPUT, sql=DEFAULT_SQL, data_release="dr17")
    print(f"Saved SDSS sample to {path}")

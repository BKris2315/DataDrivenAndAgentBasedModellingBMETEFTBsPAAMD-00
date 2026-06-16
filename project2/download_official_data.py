"""Download official data files used in the project.

This script downloads:
1. Pantheon+SH0ES Type Ia supernova distance table from the official GitHub release.
2. One SDSS DR15 optical spectrum FITS file from the official SDSS Science Archive Server.

The project notebooks also work if you download these files manually and place them in data/raw/.
"""
from __future__ import annotations

from pathlib import Path
import requests

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "Pantheon+SH0ES.dat": "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES.dat",
    "spec-0532-51993-0497.fits": "https://dr15.sdss.org/sas/dr15/sdss/spectro/redux/26/spectra/lite/0532/spec-0532-51993-0497.fits",
}


def download(url: str, path: Path) -> None:
    print(f"Downloading {url}\n  -> {path}")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    path.write_bytes(r.content)
    print(f"Saved {path} ({path.stat().st_size / 1024:.1f} KiB)")


def main() -> None:
    for filename, url in FILES.items():
        out = RAW_DIR / filename
        if out.exists() and out.stat().st_size > 0:
            print(f"Already present: {out}")
            continue
        download(url, out)


if __name__ == "__main__":
    main()

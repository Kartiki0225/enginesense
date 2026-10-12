"""Download the NASA C-MAPSS turbofan dataset and keep the FD001 files in data/.

Run once:  python download_data.py
"""

import io
import urllib.request
import zipfile
from pathlib import Path

# Official link from NASA's Prognostics Center of Excellence data repository.
URL = (
    "https://phm-datasets.s3.amazonaws.com/NASA/"
    "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
)
DATA_DIR = Path(__file__).parent / "data"
WANTED = ["train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt"]


def extract_wanted(archive: zipfile.ZipFile) -> None:
    """Save the FD001 files. The NASA zip contains another zip, so look inside it too."""
    for name in archive.namelist():
        filename = Path(name).name
        if filename in WANTED:
            (DATA_DIR / filename).write_bytes(archive.read(name))
            print("saved", filename)
        elif filename.lower().endswith(".zip"):
            inner = zipfile.ZipFile(io.BytesIO(archive.read(name)))
            extract_wanted(inner)


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    if all((DATA_DIR / name).exists() for name in WANTED):
        print("Dataset already downloaded.")
        return

    print("Downloading about 12 MB from NASA...")
    with urllib.request.urlopen(URL) as response:
        outer = zipfile.ZipFile(io.BytesIO(response.read()))
    extract_wanted(outer)

    missing = [name for name in WANTED if not (DATA_DIR / name).exists()]
    if missing:
        raise SystemExit(f"Could not find these files in the download: {missing}")
    print("Done.")


if __name__ == "__main__":
    main()

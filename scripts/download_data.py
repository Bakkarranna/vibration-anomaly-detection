"""Download the Case Western Reserve University bearing data files.

The dataset is the standard reference for bearing fault diagnosis. Every file
is a 12 kHz drive-end accelerometer recording at one of four motor loads, with
one defect type and one defect size per group. Run this first; the analysis
needs about 134 MB of ``.mat`` files.

    python scripts/download_data.py
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vibdiag.catalogue import download_manifest, file_ids  # noqa: E402

BASE = "https://engineering.case.edu/sites/default/files"


def download(file_id: int, directory: Path, force: bool = False) -> Path:
    target = directory / f"{file_id}.mat"
    if target.exists() and not force:
        return target
    url = f"{BASE}/{file_id}.mat"
    with urllib.request.urlopen(url, timeout=120) as response:
        payload = response.read()
    if not payload.startswith(b"MATLAB"):
        raise RuntimeError(f"{url} did not return a MATLAB file")
    target.write_bytes(payload)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/raw")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--ids", type=int, nargs="*", default=None)
    args = parser.parse_args()

    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    descriptions = {file_id: description for file_id, _, description in download_manifest()}
    ids = args.ids if args.ids else file_ids()
    for file_id in ids:
        path = download(file_id, directory, force=args.force)
        size = path.stat().st_size / 1e6
        print(f"{file_id:>4}  {size:6.2f} MB  {descriptions[file_id]}")
    print(f"\n{len(ids)} files in {directory}")


if __name__ == "__main__":
    main()

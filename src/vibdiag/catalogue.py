"""Catalogue of the Case Western Reserve University bearing data files.

The data centre ships 40 files: a healthy reference and three defect types
(inner race, ball, outer race) at three defect sizes, each recorded at four
motor loads. Everything sums to 4 + 3 * 3 * 4 = 40 files. The catalogue is the
single source of truth for labels, sizes and loads; both the download script
and the analysis read from it.

Each ``.mat`` file stores the accelerometer channels as
``X<id>_DE_time`` / ``X<id>_FE_time`` (drive end, fan end) and, for the faulty
files, ``X<id>_BA_time`` (baseplate). The shaft speed is in ``X<id>RPM`` where
present. Two files are known quirks: 98 has no RPM channel and 99 stores the
98 arrays alongside its own, so the loader selects by exact file id.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat

SAMPLE_RATE_HZ = 12_000.0

# Motor load in hp, indexed by the file's position within its fault/size group.
LOADS_HP = (0.0, 1.0, 2.0, 3.0)

# Shaft speeds observed in the RPM channel for the two loads where a healthy
# file is missing its speed channel. 0 hp files run at 1796-1797 rpm, 1 hp at
# 1772 rpm, 2 hp at 1748 rpm and 3 hp at 1725 rpm.
NOMINAL_RPM = {0.0: 1797.0, 1.0: 1772.0, 2.0: 1748.0, 3.0: 1725.0}

DEFECTS = {
    "healthy": (),
    "inner_race": (0.007, 0.014, 0.021),
    "ball": (0.007, 0.014, 0.021),
    "outer_race": (0.007, 0.014, 0.021),
}


@dataclass(frozen=True)
class FileRecord:
    """One data file and everything known about its recording condition."""

    file_id: int
    label: str
    defect_size_in: float | None
    load_hp: float

    @property
    def key(self) -> str:
        return f"X{self.file_id:03d}"


def _build_catalogue() -> dict[int, FileRecord]:
    records: dict[int, FileRecord] = {97: FileRecord(97, "healthy", None, 0.0)}
    # The four healthy files cross the four loads directly.
    records.update(
        {97 + i: FileRecord(97 + i, "healthy", None, load) for i, load in enumerate(LOADS_HP)}
    )
    starts = {
        "inner_race": (105, 169, 209),
        "ball": (118, 185, 225),
        "outer_race": (130, 197, 234),
    }
    for label, size_starts in starts.items():
        for size, start in zip(DEFECTS[label], size_starts):
            for load_index, load in enumerate(LOADS_HP):
                file_id = start + load_index
                records[file_id] = FileRecord(file_id, label, size, load)
    return records


FILES: dict[int, FileRecord] = _build_catalogue()


def file_ids() -> list[int]:
    """Every catalogue file id, sorted."""
    return sorted(FILES)


def load_signal(file_id: int, data_dir: Path, channel: str = "DE") -> dict:
    """Load one file's accelerometer channel and its shaft speed.

    Returns a dict with the signal array, the sample rate, the file's shaft
    speed and whether that speed came from the file or from the nominal table.
    """
    record = FILES[file_id]
    mat = loadmat(data_dir / f"{file_id}.mat")
    keys = [k for k in mat if not k.startswith("__")]
    speed_key = f"{record.key}RPM"
    channel_key = f"{record.key}_{channel}_time"

    if channel_key not in mat:
        raise KeyError(f"{file_id}.mat has no {channel_key} (found {keys})")

    if speed_key in mat:
        rpm = float(np.ravel(mat[speed_key])[0])
        rpm_source = "file"
    else:
        rpm = NOMINAL_RPM[record.load_hp]
        rpm_source = "nominal"

    return {
        "file_id": file_id,
        "label": record.label,
        "defect_size_in": record.defect_size_in,
        "load_hp": record.load_hp,
        "signal": np.ravel(mat[channel_key]).astype(float),
        "sample_rate": SAMPLE_RATE_HZ,
        "rpm": rpm,
        "rpm_source": rpm_source,
    }


def iter_dataset(data_dir: Path) -> Iterator[dict]:
    """Yield the loaded signals for every catalogue file that exists locally."""
    for file_id in file_ids():
        path = data_dir / f"{file_id}.mat"
        if not path.exists():
            continue
        yield load_signal(file_id, data_dir)


def download_manifest() -> list[tuple[int, str, str]]:
    """(file id, label, description) for the download script."""
    rows = []
    for file_id in file_ids():
        record = FILES[file_id]
        size = "-" if record.defect_size_in is None else f"{record.defect_size_in:.3f}"
        rows.append((
            file_id,
            record.label,
            (
                f"{record.label.replace('_', ' ')}, "
                f"{record.load_hp:g} hp, {size} in"
            ),
        ))
    return rows

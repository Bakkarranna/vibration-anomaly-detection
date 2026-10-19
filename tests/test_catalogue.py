"""Tests for the CWRU file catalogue."""

from __future__ import annotations

import dataclasses

import pytest

from vibdiag.catalogue import (
    FILES,
    LOADS_HP,
    NOMINAL_RPM,
    FileRecord,
    download_manifest,
    file_ids,
)


def test_catalogue_covers_forty_files():
    assert len(FILES) == 40
    assert len(file_ids()) == 40
    assert len({r.file_id for r in FILES.values()}) == 40


def test_every_fault_class_has_three_sizes_and_four_loads():
    for label in ("inner_race", "ball", "outer_race"):
        records = [r for r in FILES.values() if r.label == label]
        assert len(records) == 12, label
        assert sorted({r.defect_size_in for r in records}) == [0.007, 0.014, 0.021], label
        for size in (0.007, 0.014, 0.021):
            sizes = sorted(r.load_hp for r in records if r.defect_size_in == size)
            assert sizes == sorted(LOADS_HP), (label, size)


def test_healthy_files_cover_all_four_loads():
    healthy = [r for r in FILES.values() if r.label == "healthy"]
    assert len(healthy) == 4
    assert all(r.defect_size_in is None for r in healthy)
    assert sorted(r.load_hp for r in healthy) == sorted(LOADS_HP)


def test_known_file_positions():
    assert FILES[97].label == "healthy" and FILES[97].load_hp == 0.0
    assert FILES[100].label == "healthy" and FILES[100].load_hp == 3.0
    assert FILES[105].label == "inner_race" and FILES[105].defect_size_in == 0.007
    assert FILES[169].label == "inner_race" and FILES[169].defect_size_in == 0.014
    assert FILES[209].label == "inner_race" and FILES[209].defect_size_in == 0.021
    assert FILES[118].label == "ball" and FILES[185].defect_size_in == 0.014
    assert FILES[225].label == "ball" and FILES[225].defect_size_in == 0.021
    assert FILES[130].label == "outer_race" and FILES[130].defect_size_in == 0.007
    assert FILES[197].label == "outer_race" and FILES[234].defect_size_in == 0.021


def test_nominal_speeds_fall_with_load():
    speeds = [NOMINAL_RPM[load] for load in LOADS_HP]
    assert speeds == sorted(speeds, reverse=True)


def test_download_manifest_describes_every_file():
    manifest = download_manifest()
    assert len(manifest) == 40
    assert all(isinstance(row, tuple) and len(row) == 3 for row in manifest)
    assert any("inner race" in row[2] for row in manifest)
    assert any("outer race" in row[2] for row in manifest)


def test_file_record_is_immutable():
    record = FileRecord(97, "healthy", None, 0.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.label = "inner_race"  # type: ignore[misc]

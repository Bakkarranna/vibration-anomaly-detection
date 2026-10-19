"""Tests for dataset assembly and the two evaluation protocols."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vibdiag.features import feature_names
from vibdiag.model import (
    build_feature_table,
    confusion,
    file_level_predictions,
    leave_one_file_out,
    train_on_smallest_defect,
)

FEATURE_COLUMNS = feature_names()
SIZES = (0.007, 0.014, 0.021)
LABELS = ("inner_race", "outer_race")


def _rows() -> list[dict]:
    """Synthetic segment table: each fault lights up its own band peak."""
    rows = []
    for label in LABELS + ("healthy",):
        sizes = (None,) if label == "healthy" else SIZES
        for size in sizes:
            size_index = 0 if size is None else list(SIZES).index(size) + 1
            for load in (0.0, 1.0):
                file_id = 1000 * size_index + int(load) + 1
                for index in range(6):
                    values = {
                        "rms": 0.1, "kurtosis": 3.0, "crest_factor": 3.0,
                        "envelope_rms": 0.05,
                        "bpfi_peak_ratio": 0.01, "bpfo_peak_ratio": 0.01,
                        "bsf_peak_ratio": 0.01, "ftf_peak_ratio": 0.01,
                    }
                    if label == "inner_race":
                        values["bpfi_peak_ratio"] = 6.0 - 0.2 * index
                    elif label == "outer_race":
                        values["bpfo_peak_ratio"] = 5.0 + 0.1 * index
                    rows.append(
                        {
                            "file_id": file_id, "label": label,
                            "defect_size_in": size, "load_hp": load,
                            "rpm": 1750.0, "segment_index": index, **values,
                        }
                    )
    return rows


@pytest.fixture
def table() -> pd.DataFrame:
    frame = pd.DataFrame(
        _rows(),
        columns=["file_id", "label", "defect_size_in", "load_hp", "rpm", "segment_index"] +
        FEATURE_COLUMNS,
    )
    return frame


def test_segment_table_has_fixed_columns(table: pd.DataFrame):
    assert set(FEATURE_COLUMNS).issubset(table.columns)


def test_train_on_smallest_defect_classifies_unseen_sizes(table: pd.DataFrame):
    predictions = train_on_smallest_defect(table)
    assert len(predictions) > 0
    assert (predictions["label"] == predictions["prediction"]).mean() > 0.99


def test_leave_one_file_out_covers_every_file_once(table: pd.DataFrame):
    predictions = leave_one_file_out(table)
    truth = table[table["defect_size_in"].isin((None, 0.007))]
    assert set(predictions["file_id"]) == set(truth["file_id"])
    assert len(predictions) == len(truth)
    assert (predictions["label"] == predictions["prediction"]).mean() > 0.99


def test_file_level_predictions_majority_vote():
    segments = pd.DataFrame(
        {
            "file_id": [1, 1, 1, 2, 2],
            "label": ["inner_race", "inner_race", "outer_race", "healthy", "healthy"],
            "defect_size_in": [0.007, 0.007, 0.007, None, None],
            "load_hp": [0.0, 0.0, 0.0, 0.0, 0.0],
            "prediction": ["inner_race", "inner_race", "outer_race", "healthy", "outer_race"],
        }
    )
    files = file_level_predictions(segments)
    assert len(files) == 2
    first = files[files["file_id"] == 1].iloc[0]
    assert first["prediction"] == "inner_race"
    assert first["confidence"] == pytest.approx(2 / 3)
    assert first["correct"]
    assert not files[files["file_id"] == 2].iloc[0]["correct"]


def test_confusion_matrix_and_accuracy():
    predictions = pd.DataFrame(
        {
            "label": ["healthy", "healthy", "inner_race", "outer_race"],
            "prediction": ["healthy", "outer_race", "inner_race", "outer_race"],
        }
    )
    matrix, accuracy = confusion(predictions, "file")
    assert accuracy == pytest.approx(0.75)
    assert matrix.loc["healthy", "healthy"] == 1
    assert matrix.loc["healthy", "outer_race"] == 1
    assert matrix.loc["inner_race", "inner_race"] == 1
    assert matrix.loc["outer_race", "outer_race"] == 1


def test_build_feature_table_without_data_returns_empty():
    with tempfile.TemporaryDirectory() as tmp:
        frame = build_feature_table(Path(tmp))
    assert frame.empty
    assert set(FEATURE_COLUMNS).issubset(frame.columns)
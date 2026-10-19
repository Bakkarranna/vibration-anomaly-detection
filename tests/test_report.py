"""Tests for report writing and plotting."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from vibdiag.model import Evaluation
from vibdiag.report import (
    plot_band_peaks,
    plot_confusion,
    write_metrics,
    write_tables,
)


@pytest.fixture
def table() -> pd.DataFrame:
    rows = []
    for label, peak in (("healthy", 0.01), ("inner_race", 6.0), ("outer_race", 5.0)):
        for index in range(4):
            rows.append(
                {
                    "file_id": 100 + index,
                    "label": label,
                    "defect_size_in": None if label == "healthy" else 0.007,
                    "load_hp": float(index),
                    "rpm": 1750.0,
                    "segment_index": index,
                    "rms": 0.1, "kurtosis": 3.0, "crest_factor": 3.0,
                    "envelope_rms": 0.05,
                    **{f"{n}_peak_ratio": 0.01 for n in ("bpfi", "bpfo", "bsf", "ftf")},
                }
            )
            rows[-1][f"{label.split('_')[0] if label != 'healthy' else 'bsf'}_peak_ratio"] = peak
    return pd.DataFrame(rows)


@pytest.fixture
def evaluations() -> list[Evaluation]:
    predictions = pd.DataFrame(
        {
            "label": ["healthy", "inner_race", "outer_race", "inner_race"],
            "prediction": ["healthy", "outer_race", "outer_race", "inner_race"],
        }
    )
    matrix = pd.DataFrame(
        [[1, 0, 0], [0, 1, 1], [0, 0, 1]],
        index=["healthy", "inner_race", "outer_race"],
        columns=["healthy", "inner_race", "outer_race"],
    )
    return [Evaluation("unseen_size", predictions, predictions, matrix, 0.75, 0.75)]


def test_write_tables_creates_csv_per_protocol(tmp_path: Path, table, evaluations):
    write_tables(table, evaluations, tmp_path)
    assert (tmp_path / "segments.csv").exists()
    assert (tmp_path / "file_predictions_unseen_size.csv").exists()
    assert (tmp_path / "confusion_unseen_size.csv").exists()


def test_write_metrics_round_trips(tmp_path: Path, evaluations):
    metrics = write_metrics(evaluations, tmp_path)
    payload = json.loads((tmp_path / "metrics.json").read_text())
    assert payload == metrics
    assert payload["unseen_size"]["file_accuracy"] == 0.75


def test_plot_band_peaks_writes_a_figure(tmp_path: Path, table):
    plot_band_peaks(table, tmp_path)
    figure = tmp_path / "band_peaks_by_class.png"
    assert figure.exists() and figure.stat().st_size > 1000


def test_plot_confusion_writes_one_figure_per_protocol(tmp_path: Path, evaluations):
    plot_confusion(evaluations, tmp_path)
    assert (tmp_path / "confusion_unseen_size.png").exists()

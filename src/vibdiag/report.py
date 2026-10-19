"""Output generation: tables and figures.

Everything reported goes through this module, so the README numbers always come
from the same run that produced the figures. ``matplotlib`` uses the ``Agg``
backend so the module runs headless in CI.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .bearing import BearingGeometry, defect_frequencies_hz
from .catalogue import load_signal
from .features import DEFECT_NAMES, envelope_spectrum
from .model import Evaluation

CLASS_ORDER = ("healthy", "inner_race", "ball", "outer_race")


def write_tables(
    table: pd.DataFrame,
    evaluations: list[Evaluation],
    output_dir: Path,
) -> None:
    """Write the segment feature table and every prediction table."""
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "segments.csv", index=False)
    for evaluation in evaluations:
        evaluation.files.to_csv(
            output_dir / f"file_predictions_{evaluation.name}.csv", index=False
        )
        evaluation.confusion_matrix.to_csv(
            output_dir / f"confusion_{evaluation.name}.csv"
        )


def write_metrics(evaluations: list[Evaluation], output_dir: Path) -> dict:
    """Write a JSON summary of both protocols and return it."""
    metrics = {
        evaluation.name: {
            "files": len(evaluation.files),
            "segment_accuracy": round(evaluation.segment_accuracy, 4),
            "file_accuracy": round(evaluation.file_accuracy, 4),
        }
        for evaluation in evaluations
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def plot_band_peaks(table: pd.DataFrame, output_dir: Path) -> None:
    """Grouped bars of the four defect band peak ratios, per class."""
    grouped = table.groupby("label")[[f"{n}_peak_ratio" for n in DEFECT_NAMES]].median()
    grouped = grouped.reindex([c for c in CLASS_ORDER if c in grouped.index])
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(DEFECT_NAMES))
    width = 0.8 / len(grouped)
    for index, (label, row) in enumerate(grouped.iterrows()):
        ax.bar(x + index * width, [row[f"{n}_peak_ratio"] for n in DEFECT_NAMES], width, label=label)
    ax.set_xticks(x + width * (len(grouped) - 1) / 2, DEFECT_NAMES)
    ax.set_ylabel("Median band peak / strongest spectral component")
    ax.set_title("Envelope band content at the characteristic defect frequencies")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "band_peaks_by_class.png", dpi=130)
    plt.close(fig)


def plot_envelope_spectra(data_dir: Path, ids: tuple[int, ...], output_dir: Path) -> None:
    """Envelope spectra of representative files, with defect frequencies marked."""
    fig, ax = plt.subplots(figsize=(9, 5))
    geometry = BearingGeometry()
    for file_id in ids:
        data = load_signal(file_id, data_dir)
        signal = data["signal"][: 8192 * 4]
        spectrum, freq = envelope_spectrum(signal, data["sample_rate"])
        label = f"{file_id} {data['label']} {data['load_hp']:g} hp"
        ax.semilogy(freq[:1200], spectrum[:1200], label=label)
        for order_set, style in (
            (["bpfi", "bpfo"], ":"),
            (["bsf"], "--"),
        ):
            for name in order_set:
                centre = defect_frequencies_hz(data["rpm"], geometry)[name]
                if centre < freq[:1200][-1]:
                    ax.axvline(centre, color="grey", linestyle=style, linewidth=0.8)
    ax.set_xlim(0, 400)
    ax.set_xlabel("Envelope frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Envelope spectra of representative files (grey lines: BPFI/BPFO/BSF)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "envelope_spectra.png", dpi=130)
    plt.close(fig)


def plot_confusion(evaluations: list[Evaluation], output_dir: Path) -> None:
    """Confusion matrix figure per evaluation protocol."""
    for evaluation in evaluations:
        matrix = evaluation.confusion_matrix
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(matrix.values, cmap="Blues")
        ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=30, ha="right")
        ax.set_yticks(range(len(matrix.index)), matrix.index)
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                ax.text(col, row, int(matrix.values[row, col]), ha="center", va="center",
                        color="white" if matrix.values[row, col] > matrix.values.max() / 2 else
                        "black")
        ax.set_title(f"{evaluation.name}: file accuracy {evaluation.file_accuracy:.2%}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        fig.tight_layout()
        fig.savefig(output_dir / f"confusion_{evaluation.name}.png", dpi=130)
        plt.close(fig)

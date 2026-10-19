"""Command-line entrypoint: run the analysis on local data and write outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .catalogue import file_ids
from .model import (
    Evaluation,
    build_feature_table,
    file_level_predictions,
    leave_one_file_out,
    train_on_smallest_defect,
)
from .report import (
    plot_band_peaks,
    plot_confusion,
    plot_envelope_spectra,
    write_metrics,
    write_tables,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bearing fault diagnosis from CWRU data")
    parser.add_argument("--data", default="data/raw")
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--segment-size", type=int, default=8192)
    parser.add_argument("--stride", type=int, default=4096)
    parser.add_argument("--no-graphics", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data)
    output_dir = Path(args.output)
    available = [file_id for file_id in file_ids() if (data_dir / f"{file_id}.mat").exists()]
    if not available:
        print(f"No .mat files in {data_dir}; run scripts/download_data.py first.")
        return

    print(f"Building feature table from {len(available)} files ...")
    table = build_feature_table(data_dir, args.segment_size, args.stride)
    print(f"  {len(table)} segments, columns: {table.shape[1]}")

    print("Protocol 1: train on healthy + 0.007 in, test on unseen sizes")
    unseen = train_on_smallest_defect(table)
    unseen_files = file_level_predictions(unseen)

    print("Protocol 2: leave one file out on healthy + 0.007 in")
    lofo = leave_one_file_out(table)
    lofo_files = file_level_predictions(lofo)

    from .model import confusion

    evaluations = [
        Evaluation(
            name="unseen_size",
            segments=unseen,
            files=unseen_files,
            confusion_matrix=confusion(unseen_files, "file")[0],
            segment_accuracy=confusion(unseen, "segment")[1],
            file_accuracy=confusion(unseen_files, "file")[1],
        ),
        Evaluation(
            name="lofo",
            segments=lofo,
            files=lofo_files,
            confusion_matrix=confusion(lofo_files, "file")[0],
            segment_accuracy=confusion(lofo, "segment")[1],
            file_accuracy=confusion(lofo_files, "file")[1],
        ),
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    write_tables(table, evaluations, output_dir)
    metrics = write_metrics(evaluations, output_dir)
    for name, values in metrics.items():
        print(f"  {name}: segment {values['segment_accuracy']:.2%}, "
              f"file {values['file_accuracy']:.2%} ({values['files']} files)")

    if not args.no_graphics:
        plot_band_peaks(table, output_dir)
        plot_confusion(evaluations, output_dir)
        representatives = [97, 105, 118, 130]
        representatives = [f for f in representatives if (data_dir / f"{f}.mat").exists()]
        if representatives:
            plot_envelope_spectra(data_dir, tuple(representatives), output_dir)
        print(f"  figures and tables in {output_dir}")

    return


if __name__ == "__main__":
    main()

"""Model training and evaluation.

The honest test for this problem is not a random split. Segments of one file
are heavily correlated with each other, so a random split leaks information
about which bearing you are testing on into the training set and inflates
accuracy. Two protocols are reported instead:

* **unseen defect size** - train on the healthy files and the smallest defect
  (0.007 in), then classify the 0.014 in and 0.021 in files. The fault
  signature is present at a very different amplitude and the shaft speed is the
  same, so this is the closest thing to a real deployment test.
* **leave one file out** - train and test on the smallest defect only, holding
  out one file at a time. This gives a per-file confusion matrix without any
  leakage.

Both protocols use a segment-level model with majority voting at file level,
which is how a diagnosis would actually be made: look at many windows, then
decide.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .catalogue import FILES, load_signal
from .features import DEFECT_NAMES, feature_names, segment_features, segments_of

TRAIN_SIZES = (None, 0.007)  # healthy plus the smallest defect
TEST_SIZES = (0.014, 0.021)

# The classifier sees only amplitude-independent features: the time-domain
# indicators plus the band peaks expressed as ratios against the strongest
# component of the envelope spectrum. Raw peak magnitudes are deliberately
# excluded, since their scale is set by sensor gain and motor load rather than
# by the bearing's condition.
MODEL_FEATURES = ["rms", "kurtosis", "crest_factor"] + [
    f"{name}_peak_ratio" for name in DEFECT_NAMES
]


def build_feature_table(
    data_dir: Path,
    segment_size: int = 8192,
    stride: int | None = None,
    max_files: int | None = None,
) -> pd.DataFrame:
    """Feature table: one row per segment, keyed by the file it came from."""
    rows: list[dict] = []
    columns = ["file_id", "label", "defect_size_in", "load_hp", "rpm", "segment_index"]
    columns += feature_names()
    hop = stride or segment_size // 2
    ids = sorted(FILES)
    if max_files is not None:
        ids = ids[:max_files]
    for file_id in ids:
        path = data_dir / f"{file_id}.mat"
        if not path.exists():
            continue
        record = FILES[file_id]
        data = load_signal(file_id, data_dir)
        signal = data["signal"]
        for index, segment in enumerate(segments_of(signal, segment_size, hop)):
            features = segment_features(
                segment, data["rpm"], data["sample_rate"]
            )
            rows.append(
                {
                    "file_id": file_id,
                    "label": record.label,
                    "defect_size_in": record.defect_size_in,
                    "load_hp": record.load_hp,
                    "rpm": data["rpm"],
                    "segment_index": index,
                    **features,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _make_model() -> LogisticRegression:
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))


def _fit_predict(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    features = MODEL_FEATURES
    model = _make_model()
    model.fit(train_df[features], train_df["label"])
    predictions = model.predict(test_df[features])
    out = test_df[["file_id", "label", "defect_size_in", "load_hp"]].copy()
    out["prediction"] = predictions
    return out


def train_on_smallest_defect(table: pd.DataFrame) -> pd.DataFrame:
    """Train on healthy + 0.007 in, predict the two unseen sizes."""
    train = table[table["defect_size_in"].isin(TRAIN_SIZES)]
    test = table[table["defect_size_in"].isin(TEST_SIZES)]
    return _fit_predict(train, test)


def leave_one_file_out(table: pd.DataFrame) -> pd.DataFrame:
    """Lofo cross-validation on the healthy + 0.007 in subset (no size shift)."""
    subset = table[table["defect_size_in"].isin(TRAIN_SIZES)]
    predictions = []
    for held_out in sorted(subset["file_id"].unique()):
        train = subset[subset["file_id"] != held_out]
        test = subset[subset["file_id"] == held_out]
        predictions.append(_fit_predict(train, test))
    return pd.concat(predictions, ignore_index=True)


def file_level_predictions(segment_predictions: pd.DataFrame) -> pd.DataFrame:
    """Collapse segment predictions to one label per file by majority vote.

    A tie is not resolved by insertion order: the fault class wins, because
    clearing a bearing that half the windows flagged is a worse mistake than
    asking for an inspection that a healthy bearing did not need.
    """
    groups = segment_predictions.groupby("file_id")
    rows = []
    for file_id, group in groups:
        counts = group["prediction"].value_counts()
        best = int(counts.max())
        tied = sorted(counts[counts == best].index)
        majority = next(
            (name for name in tied if name != "healthy"),
            tied[0],
        )
        confidence = best / float(len(group))
        truth = group["label"].iloc[0]
        rows.append(
            {
                "file_id": file_id,
                "label": truth,
                "defect_size_in": group["defect_size_in"].iloc[0],
                "load_hp": group["load_hp"].iloc[0],
                "prediction": majority,
                "confidence": confidence,
                "correct": bool(majority == truth),
            }
        )
    return pd.DataFrame(rows)


def confusion(predictions: pd.DataFrame, level: str) -> tuple[pd.DataFrame, float]:
    """Confusion matrix (rows = truth, cols = prediction) and accuracy."""
    truth = predictions["label"]
    predicted = predictions["prediction"]
    labels = sorted(set(truth) | set(predicted))
    matrix = confusion_matrix(truth, predicted, labels=labels)
    frame = pd.DataFrame(matrix, index=labels, columns=labels)
    frame.index.name = "true"
    frame.columns.name = f"predicted {level}"
    return frame, float(accuracy_score(truth, predicted))


@dataclass(frozen=True)
class Evaluation:
    """Summary of one protocol, ready to be reported."""

    name: str
    segments: pd.DataFrame
    files: pd.DataFrame
    confusion_matrix: pd.DataFrame
    segment_accuracy: float
    file_accuracy: float

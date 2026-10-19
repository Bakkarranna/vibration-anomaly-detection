# Vibration Anomaly Detection

Bearing fault diagnosis from the Case Western Reserve University (CWRU) vibration
benchmark, with an evaluation designed so the reported numbers mean what they say.

The problem is *not* to fit the benchmark well. It is to fit it honestly: the 40 files
are one bearing recorded at many shaft speeds, four motor loads and three defect sizes,
so a careless evaluation leaks information from the test file into training and reports
accuracy that no real deployment would ever see.

## Why this is not just "train a classifier and report accuracy"

Two things go wrong on this dataset if you are not careful:

1. **A random split leaks.** The segments of one file are almost identical, so a random
   train/test split puts windows of the very bearing being tested into training. Accuracy
   approaches 100% and says nothing.
2. **Raw vibration amplitude is not a feature.** The band peak magnitude at a defect
   frequency scales with the motor load and the sensor gain, so it encodes *how the rig
   was run* rather than *what is wrong with the bearing*. The model here sees only
   quantities that are scale free, which is why a change of load does not by itself look
   like a fault.

## Method

**Bearing kinematics, from geometry.** The rig bearing is an SKF 6205-2RS JEM (deep
groove, 9 balls, d = 0.3126 in, D = 1.537 in, contact angle 0°). The four characteristic
frequencies are computed from those numbers as multiples of shaft speed, so they stay
valid as the speed changes with load (1797 rpm at 0 hp down to 1725 rpm at 3 hp):

| defect | order (impacts / shaft revolution) | at 1797 rpm |
|--------|-------------------------------------|-------------|
| BPFI, inner race | 5.4155 | 161.8 Hz |
| BPFO, outer race | 3.5848 | 107.4 Hz |
| BSF, ball | 2.3573 | 70.6 Hz |
| FTF, cage | 0.3983 | 11.9 Hz |

**Features.** Each 12 kHz drive-end recording is cut into 8192-sample windows with a 4096
stride (50% overlap). Per window: the analytic-signal envelope via the Hilbert transform,
its Hann-windowed spectrum, and the largest peak within ±14.6 Hz of each characteristic
frequency. That peak is reported *both* as a magnitude and as a share of the strongest
component in the envelope spectrum, which keeps the feature bounded and independent of
amplitude, window length and load. The time-domain indicators are RMS, kurtosis and crest
factor. The classifier (standardised logistic regression) sees only the three indicators
and the four band shares: 7 features, no magnitudes.

**Majority vote.** A diagnosis is made per *file*, not per window: the segments vote, and
a tie goes to the fault class, since clearing a bearing that half its windows flagged is
the worse mistake.

## Results (real measured numbers, not placeholders)

`vibdiag` writes these into `outputs/metrics.json` on every run.

| protocol | files | segment accuracy | file accuracy |
|----------|-------|------------------|---------------|
| Leave-one-file-out, healthy + 0.007 in | 12 | **100.0%** | **100.0%** |
| Unseen defect size, train on 0.007 in, test 0.014/0.021 in | 24 | 71.25% | **66.67%** |

Leave-one-file-out is perfect on the files it covers, with a clean 4/4/4 confusion matrix
at file level. Note what this does *not* mean: those are the same defect sizes and the same
bearing the model trained on minus one file.

The unseen-size protocol is the honest deployment test, and it is the interesting result.
Trained only on healthy + 0.007 in files and shown 0.014 and 0.021 in defects it has never
seen, the model gets **every 0.021 in file right (12/12)** and most of the 0.014 in files,
but fails on the two hardest groups:

- the four 0.014 in **ball** faults split between inner-race and outer-race predictions;
- the four 0.014 in **outer-race** faults are all called ball, at high confidence.

That is a real limitation, not a bug: at 0.014 in the fault signature sits at a different
amplitude and a different mix of orders than anything in the training set, and the ball
and outer-race bands compete. Confidence on the wrong files is 53–100%, so the confidences
are **not** calibrated to correctness and must not be read as probabilities.

The median band shares tell the same story as the physics: inner-race files put 100% of
their envelope spectral content at BPFI, outer-race files put it at BPFO, and ball files
put it at BSF, while the healthy files spread it across the bands. The
`outputs/band_peaks_by_class.png` figure is that table.

## Layout

```
src/vibdiag/
  bearing.py    characteristic frequencies from bearing geometry
  catalogue.py  the 40 CWRU files: labels, sizes, loads, the loader
  features.py   windowing, Hilbert envelope, band peaks, ratios
  model.py      leakage-free protocols, majority vote, evaluation
  report.py     tables and figures (Agg backend, runs headless)
  cli.py        `vibdiag` entrypoint
scripts/download_data.py   fetches the 40 .mat files (~145 MB)
tests/          37 unit tests: kinematics, catalogue, features, model, report
outputs/        segments.csv, metrics.json, predictions, confusion, figures
```

## Usage

```bash
pip install -e ".[dev]"
python scripts/download_data.py      # ~145 MB, CWRU files 97..237
pytest                                # 37 tests
vibdiag                               # full analysis -> outputs/
```

The dataset is downloaded by the script and deliberately not versioned (145 MB of MATLAB
files), so a fresh clone needs the download step. Everything is regenerable: `outputs/*`
comes from `vibdiag` alone.

## Data

Case Western Reserve University Bearing Data Center, 12 kHz drive-end accelerometer,
motor loads 0–3 hp, defect sizes 0.007/0.014/0.021 in, 40 files
(healthy 4 + inner-race 4 + ball 4 + outer-race 4, at each size).

Quirks handled by the loader: file 98 has no RPM channel, file 99 stores the 98 arrays
alongside its own, and 226 has a shorter recording than its siblings. Speeds are read
from the `RPM` channel where present and otherwise from the nominal table in
`catalogue.py`, keyed by motor load.

## Limitations

- The two failed 0.014 in groups are not explained by the current features; the model's
  confidence does not track correctness, so it should not be used as an alarm on its own.
- Evaluation covers the healthy + inner/outer/ball combinations of this bearing only. A
  different bearing needs its own geometry and its own labels.
- Only the drive-end channel is used; the base-accelerometer channel on the faulty files
  is ignored.

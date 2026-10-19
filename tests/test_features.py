"""Tests for envelope feature extraction."""

from __future__ import annotations

import numpy as np
import pytest

from vibdiag.bearing import BearingGeometry, defect_frequencies_hz
from vibdiag.features import (
    band_peak,
    envelope_spectrum,
    feature_names,
    segment_features,
    segments_of,
    time_domain_features,
)

FS = 12_000.0
RPM = 1750.0
GEOMETRY = BearingGeometry()
SAMPLE_SIZE = 8192


def _impulse_train(rate_hz: float, decay: float = 60.0) -> np.ndarray:
    """Synthetic bearing impact train: carrier resonance modulated by impulses."""
    t = np.arange(SAMPLE_SIZE) / FS
    carrier = 2000.0 * np.sin(2 * np.pi * 900.0 * t)
    envelope = np.zeros(SAMPLE_SIZE)
    spacing = max(1.0, FS / rate_hz)
    for start in np.arange(0, SAMPLE_SIZE, spacing):
        index = round(start)
        if index >= SAMPLE_SIZE:
            break
        envelope[index] += 1.0
    envelope = np.convolve(envelope, np.exp(-np.arange(200) / decay))
    envelope = envelope[:SAMPLE_SIZE] / envelope.max()
    return carrier * envelope


def test_time_domain_features_on_sinusoid():
    t = np.arange(10_000) / FS
    signal = 3.0 * np.sin(2 * np.pi * 50.0 * t)
    features = time_domain_features(signal)
    assert features["rms"] == pytest.approx(3.0 / np.sqrt(2), rel=1e-3)
    assert features["crest_factor"] == pytest.approx(np.sqrt(2), rel=0.05)
    assert features["kurtosis"] == pytest.approx(1.5, rel=0.05)


def test_time_domain_features_detect_impulsiveness():
    t = np.arange(10_000) / FS
    clean = np.sin(2 * np.pi * 50.0 * t)
    impulse = clean + 20.0 * np.sign(np.sin(2 * np.pi * 50.0 * t * 4)) * 0.01
    assert time_domain_features(impulse)["kurtosis"] > time_domain_features(clean)["kurtosis"]


def test_band_peak_finds_a_known_peak():
    freq = np.arange(0, 400.0)
    spectrum = np.zeros_like(freq)
    spectrum[200] = 5.0
    peak, peak_hz = band_peak(spectrum, freq, centre_hz=200.0, half_width_hz=10.0)
    assert peak == pytest.approx(5.0)
    assert peak_hz == pytest.approx(200.0)


def test_band_peak_returns_zero_outside_range():
    freq = np.arange(0, 400.0)
    spectrum = np.zeros_like(freq)
    assert band_peak(spectrum, freq, centre_hz=100.0, half_width_hz=10.0)[0] == 0.0
    assert band_peak(spectrum, freq, centre_hz=-5.0, half_width_hz=10.0)[0] == 0.0
    assert band_peak(spectrum, freq, centre_hz=20_000.0, half_width_hz=10.0)[0] == 0.0


def test_envelope_spectrum_shapes():
    signal = _impulse_train(defect_frequencies_hz(RPM, GEOMETRY)["bpfi"])
    spectrum, freq = envelope_spectrum(signal, FS)
    assert spectrum.shape == freq.shape
    assert spectrum.size == SAMPLE_SIZE // 2 + 1
    assert np.all(spectrum >= 0)


def test_segments_of_drops_short_tail():
    signal = np.arange(8192 * 3 + 400, dtype=float)
    parts = segments_of(signal, segment_size=8192, stride=4096)
    assert len(parts) == 5
    assert parts[0].size == parts[-1].size == 8192
    # Each window starts one stride after the previous one, and the 400-sample
    # remainder is dropped rather than padded into a short final window.
    assert parts[1][0] == parts[0][0] + 4096
    assert parts[-1][-1] == signal.size - 401


def test_segments_of_rejects_too_short_signal():
    with pytest.raises(ValueError):
        segments_of(np.arange(100.0), segment_size=8192)


def test_segment_features_find_the_injected_fault():
    rates = defect_frequencies_hz(RPM, GEOMETRY)
    features = segment_features(_impulse_train(rates["bpfi"]), RPM, FS, GEOMETRY)
    assert set(features) == set(feature_names())
    assert features["bpfi_peak_ratio"] > features["bpfo_peak_ratio"]
    assert features["bpfi_peak_ratio"] > features["bsf_peak_ratio"]


def test_segment_features_find_each_defect_type():
    rates = defect_frequencies_hz(RPM, GEOMETRY)
    for injected in ("bpfi", "bpfo", "bsf"):
        features = segment_features(_impulse_train(rates[injected]), RPM, FS, GEOMETRY)
        others = [f"{name}_peak_ratio" for name in ("bpfi", "bpfo", "bsf") if name != injected]
        assert all(features[f"{injected}_peak_ratio"] > features[other] for other in others), injected


def test_segment_features_of_a_healthy_signal():
    """A healthy bearing's envelope is nearly flat, so every band scores low.

    The two tones are chosen so their difference frequency (400 Hz) is far
    from every defect order; 160 Hz would beat straight into the BPFI band.
    """
    t = np.arange(SAMPLE_SIZE) / FS
    clean = 3.0 * np.sin(2 * np.pi * 300.0 * t) + 0.5 * np.sin(2 * np.pi * 700.0 * t)
    features = segment_features(clean, RPM, FS, GEOMETRY)
    for name in ("bpfi", "bpfo", "bsf", "ftf"):
        assert features[f"{name}_peak_ratio"] < 0.05, name


def test_segment_features_are_amplitude_invariant():
    """Features are normalised by the envelope RMS, so scaling the signal does not move them."""
    rates = defect_frequencies_hz(RPM, GEOMETRY)
    signal = _impulse_train(rates["bpfi"])
    base = segment_features(signal, RPM, FS, GEOMETRY)
    scaled = segment_features(10.0 * signal, RPM, FS, GEOMETRY)
    assert scaled["rms"] == pytest.approx(10.0 * base["rms"], rel=1e-6)
    assert scaled["bpfi_peak_ratio"] == pytest.approx(base["bpfi_peak_ratio"], rel=1e-6)

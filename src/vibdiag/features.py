"""Feature extraction for envelope analysis.

A localised race defect produces a short impulse each time a rolling element
strikes it, so the impulsive content appears as sidebands around the structural
resonances rather than at the fault frequency itself. Demodulating the signal
with its analytic signal and then looking at the spectrum of the envelope
brings those sidebands down to baseband, where the characteristic defect
frequencies sit on top of a Broadband noise floor.

The feature vector for one segment is therefore:

* a band peak at each of the four defect frequencies (BPFI, BPFO, BSF, FTF),
  searched inside a +/- 0.5 order window around the nominal frequency,
* the RMS, kurtosis and crest factor of the raw segment, which rise with the
  impulsiveness of the signal but say nothing about which race is damaged.

Each band peak is reported both as an absolute magnitude and as a ratio
against the strongest component of the envelope spectrum, i.e. the share of
the envelope's periodic content that sits at that defect frequency. That
normalisation is scale free and bounded, so the features do not depend on
absolute vibration amplitude, on the segment length, or on the motor load,
and a healthy bearing - whose envelope is almost flat - scores near zero on
every band.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert

from .bearing import BearingGeometry, defect_frequencies_hz

DEFECT_NAMES = ("bpfi", "bpfo", "bsf", "ftf")


def analytic_envelope(signal: np.ndarray) -> np.ndarray:
    """Amplitude envelope of a real signal via the analytic signal."""
    return np.abs(hilbert(np.asarray(signal, dtype=float)))


def envelope_spectrum(signal: np.ndarray, sample_rate: float) -> tuple[np.ndarray, np.ndarray]:
    """Envelope of a centred signal and its one-sided spectrum."""
    envelope = analytic_envelope(signal)
    envelope = envelope - envelope.mean()
    window = np.hanning(envelope.size)
    spectrum = np.abs(np.fft.rfft(envelope * window))
    freq = np.fft.rfftfreq(envelope.size, d=1.0 / sample_rate)
    return spectrum, freq


def spectral_reference(spectrum: np.ndarray) -> float:
    """Scale of the strongest periodic content in the envelope spectrum.

    Band peaks are measured in the same units as the whole spectrum, so the
    denominator has to be too: an FFT magnitude grows with the window length
    while a time-domain RMS does not, which makes a peak/RMS ratio depend on
    the segment length as well as on the bearing. The strongest component of
    the spectrum is the natural yardstick, and it keeps every ratio in [0, 1]
    so the features are comparable across segments and files.
    """
    return float(np.max(spectrum))


def band_peak(
    spectrum: np.ndarray,
    freq: np.ndarray,
    centre_hz: float,
    half_width_hz: float,
) -> tuple[float, float]:
    """Peak amplitude and its frequency inside a band around ``centre_hz``.

    Returns (peak, frequency). An empty band returns zero.
    """
    if centre_hz <= 0.0 or centre_hz > freq[-1]:
        return 0.0, 0.0
    mask = (freq > centre_hz - half_width_hz) & (freq < centre_hz + half_width_hz)
    if not mask.any():
        return 0.0, 0.0
    index = np.argmax(np.where(mask, spectrum, -np.inf))
    return float(spectrum[index]), float(freq[index])


def time_domain_features(signal: np.ndarray) -> dict[str, float]:
    """RMS, kurtosis and crest factor of a raw signal."""
    x = np.asarray(signal, dtype=float)
    centred = x - x.mean()
    rms = float(np.sqrt(np.mean(centred ** 2)))
    var = float(np.mean(centred ** 4))
    kurtosis = var / (rms ** 4) if rms > 0 else 0.0
    crest = float(np.max(np.abs(centred)) / rms) if rms > 0 else 0.0
    return {"rms": rms, "kurtosis": kurtosis, "crest_factor": crest}


def segment_features(
    segment: np.ndarray,
    rpm: float,
    sample_rate: float,
    geometry: BearingGeometry | None = None,
    half_width_orders: float = 0.5,
) -> dict[str, float]:
    """Feature vector for one signal segment at a known shaft speed."""
    geometry = geometry or BearingGeometry()
    spectrum, freq = envelope_spectrum(segment, sample_rate)
    envelope = analytic_envelope(segment)
    envelope_rms = float(np.sqrt(np.mean((envelope - envelope.mean()) ** 2)))
    shaft_rate_hz = rpm / 60.0
    half_width_hz = half_width_orders * shaft_rate_hz

    features = time_domain_features(segment)
    features["envelope_rms"] = envelope_rms
    reference = spectral_reference(spectrum)
    for name in DEFECT_NAMES:
        fault_hz = defect_frequencies_hz(rpm, geometry)[name]
        peak, peak_hz = band_peak(spectrum, freq, fault_hz, half_width_hz)
        features[f"{name}_peak"] = peak
        features[f"{name}_peak_hz"] = peak_hz
        features[f"{name}_peak_ratio"] = peak / reference if reference > 0 else 0.0
    return features


def feature_names() -> list[str]:
    """Column names produced by :func:`segment_features`, in fixed order."""
    names = ["rms", "kurtosis", "crest_factor", "envelope_rms"]
    for defect in DEFECT_NAMES:
        names.extend([f"{defect}_peak", f"{defect}_peak_hz", f"{defect}_peak_ratio"])
    return names


def segments_of(
    signal: np.ndarray,
    segment_size: int = 8192,
    stride: int | None = None,
) -> list[np.ndarray]:
    """Split a signal into overlapping windows; drops the tail remainder."""
    stride = stride or segment_size // 2
    signal = np.asarray(signal, dtype=float)
    if signal.size < segment_size:
        raise ValueError(f"signal of {signal.size} samples is shorter than {segment_size}")
    starts = range(0, signal.size - segment_size + 1, stride)
    return [signal[start : start + segment_size] for start in starts]

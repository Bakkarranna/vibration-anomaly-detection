'''vibdiag — bearing fault diagnosis from vibration data, plus its evaluation.

Public API
---------

Bearing kinematics
    ``vibdiag.bearing.BearingGeometry``
    ``vibdiag.bearing.defect_frequencies_hz``

Dataset
    ``vibdiag.catalogue.FILES``
    ``vibdiag.catalogue.load_signal``
    ``vibdiag.catalogue.iter_dataset``
    ``vibdiag.catalogue.download_manifest``

Features
    ``vibdiag.features.segment_features``
    ``vibdiag.features.envelope_spectrum``
    ``vibdiag.features.segments_of``

Model
    ``vibdiag.model.build_feature_table``
    ``vibdiag.model.train_on_smallest_defect``
    ``vibdiag.model.leave_one_file_out``
    ``vibdiag.model.file_level_predictions``
    ``vibdiag.model.confusion``

Outputs
    ``vibdiag.report.write_tables``
    ``vibdiag.report.write_metrics``
    ``vibdiag.report.plot_band_peaks``
    ``vibdiag.report.plot_envelope_spectra``
    ``vibdiag.report.plot_confusion``

CLI
    ``vibdiag.cli.main``
'''

from .bearing import BearingGeometry, defect_frequencies_hz
from .catalogue import FILES, download_manifest, iter_dataset, load_signal
from .features import envelope_spectrum, segment_features, segments_of
from .report import (
    plot_band_peaks,
    plot_confusion,
    plot_envelope_spectra,
    write_metrics,
    write_tables,
)

__all__ = [
    "FILES",
    "BearingGeometry",
    "defect_frequencies_hz",
    "download_manifest",
    "envelope_spectrum",
    "iter_dataset",
    "load_signal",
    "plot_band_peaks",
    "plot_confusion",
    "plot_envelope_spectra",
    "segment_features",
    "segments_of",
    "write_metrics",
    "write_tables",
]

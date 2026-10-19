"""Rolling-element bearing kinematics.

The bearing in the reference rig is the SKF 6205-2RS JEM used by the Case
Western Reserve University data centre, so the characteristic defect
frequencies can be computed from the geometry rather than taken from a table.
All frequencies are expressed as multiples of the shaft rotation rate (orders),
which keeps them valid as the shaft speed changes between motor loads.

For a bearing with ``n`` balls of diameter ``d`` on a pitch circle ``D`` and
contact angle ``beta``:

    BPFI = (n/2) * (1 + (d/D) cos(beta))
    BPFO = (n/2) * (1 - (d/D) cos(beta))
    BSF  = (D/2d) * (1 - ((d/D) cos(beta))^2)
    FTF  = (1/2)  * (1 - (d/D) cos(beta))

BPFI, BPFO and BSF stand for ball-pass frequency inner, ball-pass frequency
outer and ball (cage) spin frequency. FTF is the fundamental train (cage)
frequency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# SKF 6205-2RS JEM geometry, in inches, as specified by the CWRU data centre.
N_BALLS = 9
BALL_DIAMETER = 0.3126
PITCH_DIAMETER = 1.537
CONTACT_ANGLE_RAD = 0.0


@dataclass(frozen=True)
class BearingGeometry:
    """Geometry of a deep-groove ball bearing."""

    n_balls: int = N_BALLS
    ball_diameter: float = BALL_DIAMETER
    pitch_diameter: float = PITCH_DIAMETER
    contact_angle_rad: float = CONTACT_ANGLE_RAD

    @property
    def ball_to_pitch_ratio(self) -> float:
        """(d/D) cos(beta), the coupling between ball motion and case geometry."""
        cos_beta = math.cos(self.contact_angle_rad)
        return (self.ball_diameter / self.pitch_diameter) * cos_beta

    def bpfi_order(self) -> float:
        """Inner-race defect order in impacts per shaft revolution."""
        return (self.n_balls / 2.0) * (1.0 + self.ball_to_pitch_ratio)

    def bpfo_order(self) -> float:
        """Outer-race defect order in impacts per shaft revolution."""
        return (self.n_balls / 2.0) * (1.0 - self.ball_to_pitch_ratio)

    def bsf_order(self) -> float:
        """Rolling-element (ball) defect order in impacts per shaft revolution."""
        return (self.pitch_diameter / (2.0 * self.ball_diameter)) * (
            1.0 - self.ball_to_pitch_ratio ** 2
        )

    def ftf_order(self) -> float:
        """Cage (fundamental train) frequency order in revolutions per revolution."""
        return 0.5 * (1.0 - self.ball_to_pitch_ratio)

    def orders(self) -> dict[str, float]:
        """Return every characteristic frequency order keyed by name."""
        return {
            "bpfi": self.bpfi_order(),
            "bpfo": self.bpfo_order(),
            "bsf": self.bsf_order(),
            "ftf": self.ftf_order(),
        }


def defect_frequencies_hz(rpm: float, geometry: BearingGeometry | None = None) -> dict[str, float]:
    """Convert defect frequency orders to Hz for a given shaft speed."""
    geometry = geometry or BearingGeometry()
    shaft_rate_hz = rpm / 60.0
    return {name: order * shaft_rate_hz for name, order in geometry.orders().items()}

"""Tests for bearing kinematics."""

from __future__ import annotations

import math

import pytest

from vibdiag.bearing import BearingGeometry, defect_frequencies_hz

GEOMETRY = BearingGeometry()
STRAIGHT = BearingGeometry()


@pytest.mark.parametrize(
    "name,expected",
    [
        ("bpfi", 5.4155),
        ("bpfo", 3.5848),
        ("bsf", 2.3573),
        ("ftf", 0.3983),
    ],
)
def test_defect_orders_match_published_constants(name: str, expected: float):
    """The orders for the SKF 6205-2RS JEM match the values in the CWRU papers."""
    assert getattr(GEOMETRY, f"{name}_order")() == pytest.approx(expected, abs=1e-3)


def test_orders_are_positive_and_distinct():
    orders = GEOMETRY.orders()
    assert set(orders) == {"bpfi", "bpfo", "bsf", "ftf"}
    assert all(value > 0 for value in orders.values())
    assert len(set(orders.values())) == 4
    assert orders["bpfi"] > orders["bpfo"] > orders["bsf"]


def test_frequencies_scale_with_shaft_rate():
    orders = GEOMETRY.orders()
    at_1730 = defect_frequencies_hz(1730, GEOMETRY)
    at_3460 = defect_frequencies_hz(3460, GEOMETRY)
    for name, order in orders.items():
        assert at_1730[name] == pytest.approx(order * (1730 / 60), rel=1e-9)
        assert at_3460[name] == pytest.approx(2 * at_1730[name], rel=1e-9)


def test_orders_shift_with_contact_angle():
    """A contact angle shrinks the (d/D)cos(beta) coupling: BPFI drops, BPFO rises."""
    angled = BearingGeometry(contact_angle_rad=math.radians(15))
    straight = STRAIGHT
    assert angled.bpfi_order() < straight.bpfi_order()
    assert angled.bpfo_order() > straight.bpfo_order()


def test_cage_order_matches_coupling_formula():
    assert GEOMETRY.ftf_order() == pytest.approx(0.5 * (1 - GEOMETRY.ball_to_pitch_ratio), rel=1e-9)


def test_outer_and_inner_orders_bracket_the_ball_count():
    """The inner-race order exceeds half the ball count, the outer one stays below it."""
    half = GEOMETRY.n_balls / 2.0
    assert GEOMETRY.bpfi_order() > half + 0.5
    assert GEOMETRY.bpfo_order() < half + 0.5

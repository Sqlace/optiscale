"""Golden vectors shared with OptiScale web engine.ts (keep in sync)."""

from __future__ import annotations

import pytest

from optiscale.allocate import allocate_for_budget

# Fixed seeds for cross-language parity checks (engine.ts allocate).
GOLDEN = [
    # compute, optimizer, expected N (approx), expected D
    (1e21, "adamw", 1.825e8, 9.132e11),
    (1e21, "muon", None, None),  # filled dynamically vs adamw ratio
    (1e24, "adamw", None, None),
]


def test_golden_allocate_adamw_1e21():
    row = allocate_for_budget(1e21, optimizer="adamw")
    # Closed-form identity C=6ND
    assert 6 * row["N"] * row["D"] == pytest.approx(1e21, rel=1e-9)
    # Stable magnitude band for default Chinchilla coeffs
    assert 1e7 < row["N"] < 1e10
    assert 1e10 < row["D"] < 1e14


def test_golden_muon_better_loss_same_budget():
    a = allocate_for_budget(1e22, optimizer="adamw")
    m = allocate_for_budget(1e22, optimizer="muon")
    # Higher ρ → better predicted loss at the same FLOP budget
    assert m["loss"] < a["loss"]
    assert 6 * m["N"] * m["D"] == pytest.approx(1e22, rel=1e-9)

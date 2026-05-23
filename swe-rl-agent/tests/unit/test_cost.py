import pytest

from swe_rl.utils.cost import CostCapExceeded, CostMeter
from swe_rl.settings import settings


def test_cost_meter_accumulates():
    m = CostMeter()
    m.add(100, 50)
    m.add(200, 100)
    assert m.tokens_in == 300
    assert m.tokens_out == 150


def test_cost_meter_caps(monkeypatch):
    monkeypatch.setattr(settings, "dollar_per_1k_tokens_in", 1.0)
    monkeypatch.setattr(settings, "dollar_per_1k_tokens_out", 1.0)
    monkeypatch.setattr(settings, "max_dollars_per_run", 0.05)
    m = CostMeter()
    with pytest.raises(CostCapExceeded):
        m.add(100_000, 0)

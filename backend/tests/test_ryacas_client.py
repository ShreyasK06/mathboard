from unittest.mock import MagicMock, patch

import requests

from ryacas_client import cross_solve, _health_cache


def setup_function(_):
    _health_cache.clear()


def test_returns_none_when_unreachable():
    with patch("ryacas_client.requests.get", side_effect=requests.ConnectionError):
        assert cross_solve("x + 1") is None


def test_returns_none_on_timeout():
    with patch("ryacas_client.requests.get", side_effect=requests.Timeout):
        assert cross_solve("x + 1") is None


def test_returns_failed_on_5xx():
    health_resp = MagicMock(status_code=200, json=lambda: {"status": ["ok"]})
    bad_resp = MagicMock(status_code=500, text="boom")
    with patch("ryacas_client.requests.get", side_effect=[health_resp, bad_resp]):
        r = cross_solve("x + 1")
    assert r is not None
    assert r.status == "failed"
    assert "500" in (r.error or "")


def test_returns_success_on_valid_response():
    health_resp = MagicMock(status_code=200, json=lambda: {"status": ["ok"]})
    ok_resp = MagicMock(
        status_code=200,
        json=lambda: {"status": ["success"], "operation": ["solve"],
                      "solution": ["7"], "latex_result": ["7"]},
    )
    with patch("ryacas_client.requests.get", side_effect=[health_resp, ok_resp]):
        r = cross_solve("x + 5 = 12")
    assert r is not None
    assert r.status == "success"
    assert r.solution == "7"
    assert r.latex_result == "7"


def test_health_cache_avoids_repeated_probes():
    health_resp = MagicMock(status_code=200, json=lambda: {"status": ["ok"]})
    ok_resp = MagicMock(
        status_code=200,
        json=lambda: {"status": ["success"], "operation": ["simplify"],
                      "solution": ["x"], "latex_result": ["x"]},
    )
    with patch("ryacas_client.requests.get", side_effect=[health_resp, ok_resp, ok_resp]) as m:
        cross_solve("x")
        cross_solve("x")
    assert m.call_count == 3

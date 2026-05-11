"""HTTP client for the R Plumber symbolic solver — Ryacas is the primary
solver path; main.py falls back to SymPy only if Plumber is unreachable.
The function is still named ``cross_solve`` for backwards compatibility.
Never raises."""

import logging
import time
from dataclasses import dataclass
from typing import Optional

import requests

logger = logging.getLogger(__name__)

PLUMBER_URL = "http://127.0.0.1:8003"
DEFAULT_TIMEOUT_S = 3.0
HEALTH_CACHE_TTL_S = 30.0

_health_cache: dict[str, float] = {}  # {"last_ok": epoch_seconds}


@dataclass
class RyacasResult:
    status: str                # "success" | "failed"
    solution: Optional[str]
    latex_result: Optional[str]
    error: Optional[str]


def _is_healthy(timeout_s: float) -> bool:
    last_ok = _health_cache.get("last_ok", 0.0)
    if time.time() - last_ok < HEALTH_CACHE_TTL_S:
        return True
    try:
        r = requests.get(f"{PLUMBER_URL}/health", timeout=timeout_s)
        if r.status_code == 200:
            _health_cache["last_ok"] = time.time()
            return True
        return False
    except (requests.ConnectionError, requests.Timeout):
        return False
    except Exception as exc:
        logger.warning("Plumber health probe failed: %s", exc)
        return False


def _unwrap(field):
    """Plumber returns single-element JSON arrays for scalars; unwrap them."""
    if isinstance(field, list) and len(field) == 1:
        return field[0]
    return field


def cross_solve(latex: str, timeout_s: float = DEFAULT_TIMEOUT_S) -> Optional[RyacasResult]:
    """Send latex to /solve_ryacas. Returns None if Plumber is unreachable."""
    if not _is_healthy(timeout_s=timeout_s):
        return None
    try:
        r = requests.get(
            f"{PLUMBER_URL}/solve_ryacas",
            params={"latex": latex},
            timeout=timeout_s,
        )
    except (requests.ConnectionError, requests.Timeout):
        return None
    except Exception as exc:
        logger.warning("Plumber request failed: %s", exc)
        return None

    if r.status_code != 200:
        return RyacasResult(
            status="failed",
            solution=None,
            latex_result=None,
            error=f"HTTP {r.status_code}: {r.text[:200]}",
        )
    try:
        data = r.json()
    except Exception:
        return RyacasResult(status="failed", solution=None, latex_result=None,
                            error="invalid JSON from Plumber")

    status = _unwrap(data.get("status"))
    if status == "success":
        return RyacasResult(
            status="success",
            solution=_unwrap(data.get("solution")),
            latex_result=_unwrap(data.get("latex_result")),
            error=None,
        )
    return RyacasResult(
        status="failed",
        solution=None,
        latex_result=None,
        error=str(_unwrap(data.get("error")) or "unknown ryacas error"),
    )

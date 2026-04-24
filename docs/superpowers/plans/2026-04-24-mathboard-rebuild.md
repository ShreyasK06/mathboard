# MathBoard Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the broken R Plumber/Ryacas solver with SymPy, fix the Gemini SDK call, and add operation badge + collapsible steps + KaTeX solution rendering to the frontend.

**Architecture:** FastAPI gateway (`main.py`) calls Google Gemini (new `google-genai` SDK) for image-to-LaTeX OCR, then passes the LaTeX to `solver.py` (SymPy) which detects the operation type and solves it. R Shiny explorer stays completely untouched. Frontend receives a richer payload and renders an operation badge, steps, and KaTeX-rendered solution.

**Tech Stack:** Python 3.11+, FastAPI, `google-genai` SDK, SymPy, React + Vite, KaTeX, R Shiny (unchanged)

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `backend/solver.py` | LaTeX cleaning, operation detection, SymPy execution, timeout |
| Create | `backend/tests/__init__.py` | Test package marker |
| Create | `backend/tests/conftest.py` | Set fake API key before main imports |
| Create | `backend/tests/test_solver.py` | Unit tests for solver.py |
| Create | `backend/tests/test_main.py` | Integration tests for /convert endpoint |
| Modify | `backend/requirements.txt` | Swap `google-generativeai` → `google-genai`, add `sympy` |
| Modify | `backend/main.py` | New Gemini SDK, remove R Plumber call, add solver call |
| Modify | `frontend/mathboard/src/App.jsx` | Operation badge, steps list, KaTeX solution rendering |
| Modify | `frontend/mathboard/src/App.css` | Styles for badge, steps, solution box |
| Delete | `backend/plumber.r` | Retired — replaced by solver.py |
| Delete | `backend/run_r.r` | Retired — replaced by solver.py |
| Untouched | `backend/shiny_app.R` | R Shiny dataset explorer — do not touch |
| Untouched | `backend/run_shiny.r` | R Shiny launcher — do not touch |

---

## Task 1: Update requirements.txt and install dependencies

**Files:**
- Modify: `backend/requirements.txt`

- [ ] **Step 1: Replace requirements.txt contents**

Write `backend/requirements.txt` with exactly:
```
fastapi
uvicorn
python-multipart
google-genai
python-dotenv
sympy
Pillow
pytest
httpx
```

Note: `google-generativeai` is removed (replaced by `google-genai`). `requests` is removed (no longer calling R API). `sympy` is added.

- [ ] **Step 2: Install the updated dependencies**

```bash
cd backend && .venv/Scripts/pip install -r requirements.txt
```

Expected: Packages install without error. You should see `google-genai` and `sympy` downloaded.

- [ ] **Step 3: Verify imports work**

```bash
cd backend && .venv/Scripts/python -c "import sympy; from google import genai; from google.genai import types; print('OK')"
```

Expected output: `OK`

- [ ] **Step 4: Commit**

```bash
git add backend/requirements.txt
git commit -m "chore: swap google-generativeai for google-genai, add sympy"
```

---

## Task 2: Write failing tests for solver.py (TDD)

**Files:**
- Create: `backend/tests/__init__.py`
- Create: `backend/tests/conftest.py`
- Create: `backend/tests/test_solver.py`

- [ ] **Step 1: Create the test package**

Create `backend/tests/__init__.py` as an empty file.

- [ ] **Step 2: Create conftest.py**

Create `backend/tests/conftest.py`:

```python
import os
os.environ.setdefault("GEMINI_API_KEY", "test_key_for_testing")
```

This must exist so that importing `main` in tests doesn't crash when there's no real API key.

- [ ] **Step 3: Write test_solver.py**

Create `backend/tests/test_solver.py`:

```python
import pytest
from solver import detect_operation, solve_expression


# ── detect_operation ──────────────────────────────────────────────────────────

def test_detect_solve_linear():
    assert detect_operation("x + 2 = 5") == "solve"

def test_detect_solve_quadratic():
    assert detect_operation("x^2 - 4 = 0") == "solve"

def test_detect_integrate_indefinite():
    assert detect_operation(r"\int x^2 dx") == "integrate"

def test_detect_integrate_definite():
    assert detect_operation(r"\int_{0}^{1} x^2 dx") == "integrate"

def test_detect_differentiate_frac():
    assert detect_operation(r"\frac{d}{dx} x^3") == "differentiate"

def test_detect_differentiate_prime():
    assert detect_operation("f'(x)") == "differentiate"

def test_detect_limit():
    assert detect_operation(r"\lim_{x \to 0} x") == "limit"

def test_detect_simplify_default():
    assert detect_operation("x^2 + 2x + 1") == "simplify"

def test_detect_simplify_arithmetic():
    assert detect_operation("3 + 5") == "simplify"


# ── result shape ──────────────────────────────────────────────────────────────

def test_result_has_required_keys_on_success():
    result = solve_expression("3 + 5")
    assert result["status"] == "success"
    for key in ("operation", "operation_label", "steps", "solution", "latex_result"):
        assert key in result, f"Missing key in success result: {key}"

def test_steps_is_nonempty_list():
    result = solve_expression("x + 1")
    assert isinstance(result["steps"], list)
    assert len(result["steps"]) >= 1

def test_result_is_always_a_dict():
    for expr in ["x + 1", "x = 2", r"\int x dx", r"\frac{d}{dx} x^2", "garbage###"]:
        result = solve_expression(expr)
        assert isinstance(result, dict), f"Not a dict for: {expr}"
        assert "status" in result


# ── simplify ──────────────────────────────────────────────────────────────────

def test_simplify_arithmetic():
    result = solve_expression("3 + 5")
    assert result["status"] == "success"
    assert result["operation"] == "simplify"
    assert "8" in result["solution"]

def test_simplify_operation_label():
    result = solve_expression("x + 1")
    assert result["operation_label"] == "Simplifying"


# ── solve ─────────────────────────────────────────────────────────────────────

def test_solve_linear_equation():
    result = solve_expression("x + 2 = 5")
    assert result["status"] == "success"
    assert result["operation"] == "solve"
    assert "3" in result["solution"]

def test_solve_quadratic_equation():
    result = solve_expression("x^2 - 4 = 0")
    assert result["status"] == "success"
    assert result["operation"] == "solve"
    assert "2" in result["solution"]

def test_solve_operation_label():
    result = solve_expression("x = 1")
    assert result["operation_label"] == "Solving equation"


# ── differentiate ─────────────────────────────────────────────────────────────

def test_differentiate_power_rule():
    result = solve_expression(r"\frac{d}{dx} x^2")
    assert result["status"] == "success"
    assert result["operation"] == "differentiate"
    assert "2" in result["solution"]

def test_differentiate_operation_label():
    result = solve_expression(r"\frac{d}{dx} x")
    assert result["operation_label"] == "Differentiating"


# ── integrate ─────────────────────────────────────────────────────────────────

def test_integrate_basic():
    result = solve_expression(r"\int x dx")
    assert result["status"] == "success"
    assert result["operation"] == "integrate"
    assert "x" in result["solution"]

def test_integrate_operation_label():
    result = solve_expression(r"\int x dx")
    assert result["operation_label"] == "Integrating"


# ── error / timeout ───────────────────────────────────────────────────────────

def test_failed_result_has_error_key():
    result = solve_expression(r"\completely_invalid{{{garbage}}}")
    if result["status"] == "failed":
        assert "error" in result

def test_timeout_returns_error_dict(monkeypatch):
    import solver as s

    original_run = s._OP_FUNCS["simplify"]

    def hang(latex, steps):
        import time
        time.sleep(20)
        return original_run(latex, steps)

    monkeypatch.setitem(s._OP_FUNCS, "simplify", hang)
    result = s.solve_expression("x + 1")
    assert result["status"] == "failed"
    assert "complex" in result["error"].lower() or "time" in result["error"].lower()
```

- [ ] **Step 4: Run tests to confirm they fail (solver.py not yet written)**

```bash
cd backend && .venv/Scripts/python -m pytest tests/test_solver.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'solver'` — this confirms tests are wired up and correctly fail before implementation.

- [ ] **Step 5: Commit failing tests**

```bash
git add backend/tests/
git commit -m "test: add failing solver.py tests (TDD)"
```

---

## Task 3: Implement solver.py

**Files:**
- Create: `backend/solver.py`

- [ ] **Step 1: Create solver.py**

Create `backend/solver.py`:

```python
import re
import threading
from typing import Any

import sympy

_x, _y, _z, _t, _n = sympy.symbols("x y z t n")
_LOCALS: dict[str, Any] = {s.name: s for s in (_x, _y, _z, _t, _n)}
_LOCALS.update({
    "pi": sympy.pi, "E": sympy.E, "oo": sympy.oo,
    "sin": sympy.sin, "cos": sympy.cos, "tan": sympy.tan,
    "log": sympy.log, "ln": sympy.log, "exp": sympy.exp,
    "sqrt": sympy.sqrt, "abs": sympy.Abs,
})

_OP_LABELS = {
    "solve": "Solving equation",
    "integrate": "Integrating",
    "differentiate": "Differentiating",
    "limit": "Finding limit",
    "simplify": "Simplifying",
}


def _clean_latex(latex: str) -> str:
    s = latex.strip()
    for start, end in (("\\(", "\\)"), ("$$", "$$"), ("$", "$")):
        if s.startswith(start) and s.endswith(end):
            s = s[len(start):-len(end)].strip()
    s = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"((\1)/(\2))", s)
    s = re.sub(r"\^\{([^{}]*)\}", r"**(\1)", s)
    s = re.sub(r"\^([^{(])", r"**\1", s)
    s = re.sub(r"\\sqrt\{([^{}]*)\}", r"sqrt(\1)", s)
    s = re.sub(r"\\left\s*[\(\[|]", "(", s)
    s = re.sub(r"\\right\s*[\)\]|]", ")", s)
    for fn in ("sin", "cos", "tan", "cot", "sec", "csc",
               "arcsin", "arccos", "arctan",
               "sinh", "cosh", "tanh", "ln", "log", "exp", "abs"):
        s = s.replace(f"\\{fn}", fn)
    for k, v in {
        "\\alpha": "alpha", "\\beta": "beta", "\\gamma": "gamma",
        "\\theta": "theta", "\\phi": "phi", "\\pi": "pi",
        "\\infty": "oo", "\\cdot": "*", "\\times": "*", "\\div": "/",
    }.items():
        s = s.replace(k, v)
    s = re.sub(r"\\[a-zA-Z]+\*?", "", s)
    s = s.replace("{", "(").replace("}", ")")
    s = re.sub(r"(\d)([a-zA-Z(])", r"\1*\2", s)
    return s.strip()


def _parse(latex: str) -> sympy.Expr:
    try:
        from sympy.parsing.latex import parse_latex
        return parse_latex(latex)
    except Exception:
        pass
    return sympy.sympify(_clean_latex(latex), locals=_LOCALS)


def detect_operation(latex: str) -> str:
    if "=" in latex:
        return "solve"
    if r"\int" in latex:
        return "integrate"
    if re.search(r"\\frac\s*\{d", latex) or "'" in latex or r"\prime" in latex:
        return "differentiate"
    if r"\lim" in latex:
        return "limit"
    return "simplify"


def _do_solve(latex: str, steps: list[str]) -> tuple[str, str]:
    steps.append("Detected equation (contains '=')")
    lhs_str, rhs_str = latex.split("=", 1)
    lhs = _parse(lhs_str.strip())
    rhs = _parse(rhs_str.strip())
    expr = lhs - rhs
    steps.append(f"Rewritten as {sympy.latex(expr)} = 0")
    free = sorted(expr.free_symbols, key=str)
    if not free:
        result = sympy.simplify(expr)
        steps.append("No free variables; simplified both sides")
        return str(result), sympy.latex(result)
    var = free[0]
    steps.append(f"Solving for {var}")
    solutions = sympy.solve(expr, var)
    steps.append(f"Found {len(solutions)} solution(s)")
    return (
        ", ".join(str(s) for s in solutions),
        ", ".join(sympy.latex(s) for s in solutions),
    )


def _do_integrate(latex: str, steps: list[str]) -> tuple[str, str]:
    steps.append("Detected integral")
    body = re.sub(r"\\int\s*", "", latex, count=1)
    limits_m = re.match(r"_\{?([^}^\\]+)\}?\s*\^\{?([^}\\]+)\}?\s*(.*)", body)
    if limits_m:
        a_str, b_str, body = limits_m.groups()
        steps.append(f"Definite integral from {a_str.strip()} to {b_str.strip()}")
        var_m = re.search(r"\\,?\s*d([a-z])\s*$", body)
        var = sympy.Symbol(var_m.group(1)) if var_m else _x
        expr_str = re.sub(r"\\,?\s*d[a-z]\s*$", "", body).strip()
        expr = _parse(expr_str)
        a, b = _parse(a_str.strip()), _parse(b_str.strip())
        steps.append(f"Integrating {sympy.latex(expr)} w.r.t. {var} from {a} to {b}")
        result = sympy.simplify(sympy.integrate(expr, (var, a, b)))
    else:
        var_m = re.search(r"\\,?\s*d([a-z])\s*$", body)
        var = sympy.Symbol(var_m.group(1)) if var_m else _x
        expr_str = re.sub(r"\\,?\s*d[a-z]\s*$", "", body).strip()
        steps.append(f"Indefinite integral with respect to {var}")
        expr = _parse(expr_str)
        steps.append(f"Integrating {sympy.latex(expr)}")
        result = sympy.integrate(expr, var)
        steps.append("Constant of integration omitted")
    return str(result), sympy.latex(result)


def _do_differentiate(latex: str, steps: list[str]) -> tuple[str, str]:
    steps.append("Detected derivative")
    tagged = re.sub(
        r"\\frac\s*\{d(?:\^\{?\d+\}?)?\}\s*\{d([a-z])(?:\^\{?\d+\}?)?\}",
        r"__DIFF_\1__ ",
        latex,
    )
    diff_m = re.search(r"__DIFF_([a-z])__\s*(.*)", tagged)
    if diff_m:
        var = sympy.Symbol(diff_m.group(1))
        expr_str = diff_m.group(2).strip()
        steps.append(f"Differentiating with respect to {var}")
        expr = _parse(expr_str)
    else:
        expr = _parse(re.sub(r"'", "", latex).strip())
        var = _x
        steps.append("Differentiating with respect to x (prime notation)")
    result = sympy.diff(expr, var)
    steps.append("Applied differentiation rules")
    return str(result), sympy.latex(result)


def _do_limit(latex: str, steps: list[str]) -> tuple[str, str]:
    steps.append("Detected limit expression")
    lim_m = re.search(r"\\lim_\{?([a-z])\s*\\to\s*([^}\\]+)\}?\s*(.*)", latex)
    if lim_m:
        var_name, approach_str, expr_str = lim_m.groups()
        var = sympy.Symbol(var_name)
        approach = _parse(approach_str.strip())
        expr = _parse(expr_str.strip())
        steps.append(f"Limit as {var} → {approach}")
        result = sympy.limit(expr, var, approach)
    else:
        steps.append("Could not parse limit structure; simplifying instead")
        result = sympy.simplify(_parse(re.sub(r"\\lim[^a-z]*", "", latex)))
    return str(result), sympy.latex(result)


def _do_simplify(latex: str, steps: list[str]) -> tuple[str, str]:
    steps.append("Simplifying expression")
    result = sympy.simplify(_parse(latex))
    steps.append("Applied SymPy simplification rules")
    return str(result), sympy.latex(result)


_OP_FUNCS: dict[str, Any] = {
    "solve": _do_solve,
    "integrate": _do_integrate,
    "differentiate": _do_differentiate,
    "limit": _do_limit,
    "simplify": _do_simplify,
}


def solve_expression(latex: str) -> dict[str, Any]:
    operation = detect_operation(latex)
    steps: list[str] = []
    result_holder: list[dict] = []

    def _run() -> None:
        try:
            sol_str, sol_latex = _OP_FUNCS[operation](latex, steps)
            result_holder.append({
                "status": "success",
                "operation": operation,
                "operation_label": _OP_LABELS[operation],
                "steps": steps,
                "solution": sol_str,
                "latex_result": sol_latex,
            })
        except Exception as primary_exc:
            steps.append("Primary operation failed; falling back to simplify")
            try:
                sol_str, sol_latex = _do_simplify(latex, steps)
                result_holder.append({
                    "status": "success",
                    "operation": "simplify",
                    "operation_label": "Simplifying",
                    "steps": steps,
                    "solution": sol_str,
                    "latex_result": sol_latex,
                })
            except Exception as fallback_exc:
                result_holder.append({
                    "status": "failed",
                    "error": f"Could not parse expression: {fallback_exc}",
                    "operation": operation,
                })

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=10)

    if thread.is_alive():
        return {
            "status": "failed",
            "error": "Expression too complex to solve in time.",
            "operation": operation,
        }

    return result_holder[0] if result_holder else {
        "status": "failed",
        "error": "Solver produced no result.",
        "operation": operation,
    }
```

- [ ] **Step 2: Run tests**

```bash
cd backend && .venv/Scripts/python -m pytest tests/test_solver.py -v
```

Expected: All tests pass. If `test_differentiate_power_rule` or `test_integrate_basic` fail due to `sympy.parsing.latex` not being available, they will fall back to the `_clean_latex` path — check with `-s` flag:

```bash
cd backend && .venv/Scripts/python -m pytest tests/test_solver.py::test_differentiate_power_rule -v -s
```

- [ ] **Step 3: Fix any remaining failures**

If the LaTeX parser path fails for `\frac{d}{dx}` patterns, this is expected — the `_clean_latex` fallback handles `\frac` by converting it to Python division, which confuses the differentiation detector. Debug with:

```bash
cd backend && .venv/Scripts/python -c "from solver import solve_expression; import pprint; pprint.pprint(solve_expression(r'\frac{d}{dx} x^2'))"
```

If it returns `operation: differentiate` and `2*x` or `2` in solution, tests should pass.

- [ ] **Step 4: Commit**

```bash
git add backend/solver.py
git commit -m "feat: add SymPy solver (solve, diff, integrate, limit, simplify)"
```

---

## Task 4: Rewrite main.py

**Files:**
- Modify: `backend/main.py`

- [ ] **Step 1: Replace main.py contents**

Overwrite `backend/main.py` with:

```python
"""
MathBoard FastAPI backend.

Accepts a handwritten-math image, sends it to Google Gemini for OCR → LaTeX,
passes the LaTeX to solver.py (SymPy), and returns the LaTeX + solution.
"""

import os
import sys

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types

import solver

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")

if not GEMINI_API_KEY:
    print(
        "\n"
        "================================================================\n"
        "  WARNING: GEMINI_API_KEY is not set.\n"
        "  Create backend/.env with GEMINI_API_KEY=your_key_here\n"
        "  Generate a key at https://aistudio.google.com\n"
        "================================================================\n",
        file=sys.stderr,
        flush=True,
    )
    _client = None
else:
    _client = genai.Client(api_key=GEMINI_API_KEY)

OCR_PROMPT = (
    "You are an expert mathematical OCR system. "
    "Extract the mathematical expression from this image and return ONLY the "
    "raw LaTeX string. Do not include any markdown formatting, backticks, or "
    "conversational text. Do not include \\( or \\) wrappers. Just the math."
)

app = FastAPI(title="MathBoard Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _strip_latex_fences(latex: str) -> str:
    if not latex:
        return latex
    s = latex.strip()
    if s.startswith("```"):
        first_newline = s.find("\n")
        s = s[first_newline + 1:] if first_newline != -1 else s.lstrip("`")
        if s.endswith("```"):
            s = s[:-3]
        s = s.strip()
    s = s.strip("`").strip()
    if s.startswith("\\(") and s.endswith("\\)"):
        s = s[2:-2].strip()
    if s.startswith("$$") and s.endswith("$$"):
        s = s[2:-2].strip()
    elif s.startswith("$") and s.endswith("$"):
        s = s[1:-1].strip()
    return s


def _classify_gemini_error(exc: Exception) -> str:
    msg = str(exc)
    lowered = msg.lower()
    if "403" in msg or "permission_denied" in lowered or "service_disabled" in lowered:
        return "Gemini API key invalid. Go to aistudio.google.com and create a new API key."
    if "401" in msg or "unauthenticated" in lowered or "api key not valid" in lowered:
        return "Gemini API key is not authorized. Generate a fresh key at aistudio.google.com."
    if "429" in msg or "quota" in lowered or "rate" in lowered:
        return "Gemini quota / rate limit exceeded. Wait a bit and try again."
    if "deadline" in lowered or "timeout" in lowered:
        return "Gemini request timed out. Check your internet connection and retry."
    return f"Gemini API error: {msg}"


@app.get("/health")
def health():
    return {
        "status": "ok",
        "gemini_key_set": bool(GEMINI_API_KEY),
        "gemini_model": GEMINI_MODEL,
    }


@app.post("/convert")
async def convert_equation(file: UploadFile = File(...)):
    if not GEMINI_API_KEY or _client is None:
        return {
            "error": (
                "Gemini API key is not configured on the server. "
                "Add GEMINI_API_KEY=... to backend/.env and restart."
            )
        }

    try:
        image_bytes = await file.read()
    except Exception as exc:
        return {"error": f"Failed to read uploaded file: {exc}"}

    if not image_bytes:
        return {"error": "Uploaded image was empty."}

    print("[Routing] Sending image to Gemini...", flush=True)
    try:
        response = _client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                OCR_PROMPT,
                types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            ],
        )
        raw_latex = (response.text or "").strip()
    except Exception as exc:
        print(f"[Gemini Error] {exc}", flush=True)
        return {"error": _classify_gemini_error(exc)}

    latex_string = _strip_latex_fences(raw_latex)

    if not latex_string:
        return {
            "error": (
                "Gemini returned an empty response. Try drawing a clearer "
                "expression or using a thicker pen."
            )
        }

    print(f"[Gemini Result] {latex_string}", flush=True)
    solution_data = solver.solve_expression(latex_string)
    print(f"[Solver Result] {solution_data}", flush=True)

    return {
        "status": "success",
        "latex": latex_string,
        "solution_data": solution_data,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

- [ ] **Step 2: Verify the backend starts**

```bash
cd backend && .venv/Scripts/python -m uvicorn main:app --port 8000 &
```

Wait 2 seconds, then:

```bash
curl http://127.0.0.1:8000/health
```

Expected:
```json
{"status":"ok","gemini_key_set":true,"gemini_model":"gemini-1.5-flash"}
```

Stop the test server after verifying: `kill %1` (or Ctrl+C if in foreground).

- [ ] **Step 3: Commit**

```bash
git add backend/main.py
git commit -m "feat: update main.py — google-genai SDK, SymPy solver integration"
```

---

## Task 5: Write and run main.py integration tests

**Files:**
- Create: `backend/tests/test_main.py`

- [ ] **Step 1: Write test_main.py**

Create `backend/tests/test_main.py`:

```python
import io
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from PIL import Image

import main

_client = TestClient(main.app)

_MOCK_SOLVER_RESULT = {
    "status": "success",
    "operation": "simplify",
    "operation_label": "Simplifying",
    "steps": ["Simplifying expression", "Applied SymPy simplification rules"],
    "solution": "x**2 + 1",
    "latex_result": "x^{2} + 1",
}


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (100, 100), color="white").save(buf, format="PNG")
    return buf.getvalue()


def test_health_returns_ok():
    resp = _client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "gemini_key_set" in data
    assert "gemini_model" in data


def test_convert_returns_error_when_image_empty():
    resp = _client.post("/convert", files={"file": ("b.png", b"", "image/png")})
    assert resp.status_code == 200
    assert "error" in resp.json()


def test_convert_happy_path():
    mock_response = MagicMock()
    mock_response.text = "x^2 + 1"

    with patch.object(main._client.models, "generate_content", return_value=mock_response), \
         patch.object(main.solver, "solve_expression", return_value=_MOCK_SOLVER_RESULT):
        resp = _client.post("/convert", files={"file": ("b.png", _png_bytes(), "image/png")})

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["latex"] == "x^2 + 1"
    sd = data["solution_data"]
    assert sd["operation"] == "simplify"
    assert sd["operation_label"] == "Simplifying"
    assert isinstance(sd["steps"], list)
    assert sd["latex_result"] == "x^{2} + 1"


def test_convert_gemini_empty_text_returns_error():
    mock_response = MagicMock()
    mock_response.text = ""

    with patch.object(main._client.models, "generate_content", return_value=mock_response):
        resp = _client.post("/convert", files={"file": ("b.png", _png_bytes(), "image/png")})

    assert resp.status_code == 200
    assert "error" in resp.json()


def test_convert_gemini_exception_returns_classified_error():
    with patch.object(
        main._client.models,
        "generate_content",
        side_effect=Exception("429 quota exceeded"),
    ):
        resp = _client.post("/convert", files={"file": ("b.png", _png_bytes(), "image/png")})

    assert resp.status_code == 200
    data = resp.json()
    assert "error" in data
    assert "quota" in data["error"].lower() or "rate" in data["error"].lower()
```

- [ ] **Step 2: Run tests**

```bash
cd backend && .venv/Scripts/python -m pytest tests/test_main.py -v
```

Expected: All 5 tests pass. If `patch.object(main._client.models, ...)` raises `AttributeError` because `_client` is None (key not found in environment), the `conftest.py` env var should have handled it. Verify with:

```bash
cd backend && .venv/Scripts/python -c "import tests.conftest; import main; print(main._client)"
```

Expected: A `genai.Client` object, not `None`.

- [ ] **Step 3: Run all backend tests together**

```bash
cd backend && .venv/Scripts/python -m pytest tests/ -v
```

Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
git add backend/tests/test_main.py
git commit -m "test: add integration tests for /convert endpoint"
```

---

## Task 6: Delete retired R solver files

**Files:**
- Delete: `backend/plumber.r`
- Delete: `backend/run_r.r`

- [ ] **Step 1: Remove files and commit**

```bash
git rm backend/plumber.r backend/run_r.r
git commit -m "chore: remove R Plumber solver (replaced by SymPy in solver.py)"
```

---

## Task 7: Update App.jsx

**Files:**
- Modify: `frontend/mathboard/src/App.jsx`

- [ ] **Step 1: Update the setConvertResult call**

In `App.jsx`, find this block (around line 304–315):

```javascript
      setConvertResult({
        latex: data.latex || "",
        solution: solutionText,
        isSolutionError: Boolean(solData.error),
        yacasParsed: solData.yacas_parsed || "",
      });
```

Replace it with:

```javascript
      setConvertResult({
        latex: data.latex || "",
        solution: solutionText,
        isSolutionError: Boolean(solData.error),
        operationLabel: solData.operation_label || "",
        steps: solData.steps || [],
        latexResult: solData.latex_result || "",
      });
```

- [ ] **Step 2: Replace the result-success JSX block**

Find the entire `<div className="result-success">` block (starts around line 475, ends around line 499) and replace it with:

```jsx
              {!isLoading && convertResult && (
                <div className="result-success">
                  <h3 className="result-label">Recognized LaTeX</h3>
                  <div
                    className="latex-box"
                    dangerouslySetInnerHTML={{
                      __html: katex.renderToString(convertResult.latex || "\\,", {
                        throwOnError: false,
                        displayMode: true,
                      }),
                    }}
                  />
                  <div className="latex-raw">
                    <code>{convertResult.latex}</code>
                  </div>

                  {convertResult.operationLabel && (
                    <div className="operation-badge">{convertResult.operationLabel}</div>
                  )}

                  {convertResult.steps && convertResult.steps.length > 0 && (
                    <details className="steps-details">
                      <summary className="steps-summary">
                        Steps ({convertResult.steps.length})
                      </summary>
                      <ol className="steps-list">
                        {convertResult.steps.map((step, i) => (
                          <li key={i}>{step}</li>
                        ))}
                      </ol>
                    </details>
                  )}

                  <h3 className="result-label">Solution</h3>
                  {convertResult.isSolutionError ? (
                    <div className="result-error">
                      <div className="result-error-body">{convertResult.solution}</div>
                    </div>
                  ) : convertResult.latexResult ? (
                    <div
                      className="latex-box solution-box"
                      dangerouslySetInnerHTML={{
                        __html: katex.renderToString(convertResult.latexResult, {
                          throwOnError: false,
                          displayMode: true,
                        }),
                      }}
                    />
                  ) : (
                    <div className="solution-text">{convertResult.solution}</div>
                  )}
                </div>
              )}
```

- [ ] **Step 3: Commit**

```bash
git add frontend/mathboard/src/App.jsx
git commit -m "feat: add operation badge, collapsible steps, KaTeX solution rendering"
```

---

## Task 8: Update App.css

**Files:**
- Modify: `frontend/mathboard/src/App.css`

- [ ] **Step 1: Append new styles to end of App.css**

Add to the very end of `frontend/mathboard/src/App.css`:

```css
/* ---------- Operation badge ---------- */
.operation-badge {
  display: inline-block;
  padding: 4px 12px;
  background: var(--brand-grad);
  color: #fff;
  border-radius: 20px;
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: 0.4px;
  align-self: flex-start;
}

/* ---------- Steps ---------- */
.steps-details {
  border: 1px solid var(--border);
  border-radius: 10px;
  overflow: hidden;
}

.steps-summary {
  padding: 10px 14px;
  background: #f9fafd;
  cursor: pointer;
  font-size: 0.85rem;
  font-weight: 700;
  color: var(--text-soft);
  list-style: none;
  user-select: none;
}

.steps-summary::-webkit-details-marker {
  display: none;
}

.steps-summary::before {
  content: "▶ ";
  font-size: 0.7em;
}

details[open] .steps-summary::before {
  content: "▼ ";
}

.steps-list {
  margin: 0;
  padding: 10px 14px 10px 30px;
  display: flex;
  flex-direction: column;
  gap: 6px;
  font-size: 0.88rem;
  color: var(--text-soft);
  background: #fff;
  border-top: 1px solid var(--border);
}

.steps-list li {
  line-height: 1.45;
}

/* ---------- Solution KaTeX box ---------- */
.solution-box {
  background: var(--success-bg);
  border-color: var(--success-border);
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/mathboard/src/App.css
git commit -m "feat: add styles for operation badge, steps list, and solution box"
```

---

## Task 9: End-to-end smoke test

- [ ] **Step 1: Start the backend**

```bash
cd backend && .venv/Scripts/python -m uvicorn main:app --reload --port 8000
```

Expected: `Application startup complete.` with no import errors.

- [ ] **Step 2: Verify health endpoint**

In a new terminal:

```bash
curl http://127.0.0.1:8000/health
```

Expected:
```json
{"status":"ok","gemini_key_set":true,"gemini_model":"gemini-1.5-flash"}
```

- [ ] **Step 3: Start the frontend**

```bash
cd frontend/mathboard && npm run dev
```

Expected: `VITE ready on http://localhost:5173`

- [ ] **Step 4: Manual UI test — solve**

Open `http://localhost:5173`. Draw `x + 2 = 5` on the canvas. Click Convert. Verify:
- LaTeX box shows rendered `x + 2 = 5`
- Operation badge shows **"Solving equation"**
- Steps section is visible and expandable (shows ≥ 2 steps)
- Solution box shows `3` rendered in KaTeX

- [ ] **Step 5: Manual UI test — simplify**

Clear canvas. Draw `3 + 5`. Click Convert. Verify solution shows `8`.

- [ ] **Step 6: Run full test suite**

```bash
cd backend && .venv/Scripts/python -m pytest tests/ -v
```

Expected: All tests pass.

- [ ] **Step 7: Final commit**

```bash
git add -A
git commit -m "chore: end-to-end rebuild complete — SymPy solver, google-genai SDK, richer frontend"
```

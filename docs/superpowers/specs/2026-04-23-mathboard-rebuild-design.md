# MathBoard Rebuild Design
**Date:** 2026-04-23
**Status:** Approved

## Problem

The current MathBoard stack has four services (React frontend, FastAPI backend, R Plumber solver, R Shiny explorer). The R Plumber / Ryacas solver is fragile — LaTeX-to-Yacas translation fails on most non-trivial expressions, and Ryacas has limited coverage. The `google-generativeai` SDK API has also shifted, breaking the Gemini OCR call.

## Goals

- Replace the R Plumber solver with SymPy (Python) — full CAS coverage, no separate process
- Fix the Gemini SDK call (updated API)
- Keep R Shiny dataset explorer unchanged
- Add minor frontend improvements: operation badge, collapsible steps, KaTeX solution rendering

## Architecture

```
┌─────────────────────────────────────┐
│  React Frontend (Vite, port 5173)   │
│  Canvas → PNG → POST /convert       │
│  Displays: LaTeX + operation badge  │
│            + steps + solution       │
└──────────────┬──────────────────────┘
               │ HTTP multipart / JSON
┌──────────────▼──────────────────────┐
│  FastAPI Gateway  (port 8000)       │
│  backend/main.py                    │
│  ├─ POST /convert                   │
│  │   1. image → Gemini 1.5 Flash    │
│  │      → raw LaTeX string          │
│  │   2. LaTeX → solver.py → result  │
│  └─ GET /health                     │
│  backend/solver.py  (new)           │
│  ├─ parse_latex(latex) → sympy expr │
│  ├─ detect_operation(latex)         │
│  └─ solve_expression(latex)         │
│      → {operation, steps, solution, │
│          latex_result, status}      │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│  R Shiny  (port 3838) — UNCHANGED   │
│  backend/shiny_app.R                │
│  HASYv2 dataset explorer            │
└─────────────────────────────────────┘
```

**Files retired:** `backend/plumber.r`, `backend/run_r.r`

**Files modified:** `backend/main.py`, `backend/requirements.txt`, `frontend/mathboard/src/App.jsx`, `frontend/mathboard/src/App.css`

**Files added:** `backend/solver.py`

**Files untouched:** `backend/shiny_app.R`, `backend/run_shiny.r`, all frontend config

## Data Flow

### Request
`POST /convert` — multipart form, field `file` = PNG blob (same as current).

### Response
```json
{
  "status": "success",
  "latex": "\\frac{d}{dx} x^2 + 3x",
  "solution_data": {
    "operation": "differentiate",
    "steps": [
      "Detected derivative expression",
      "Applied power rule to x²",
      "Differentiated 3x"
    ],
    "solution": "2*x + 3",
    "latex_result": "2 x + 3",
    "status": "success"
  }
}
```

On error: `{"error": "<human-readable message>"}` (top-level, same as current).

## solver.py — Operation Detection (priority order)

| Priority | Trigger in LaTeX | SymPy operation |
|----------|-----------------|-----------------|
| 1 | `=` present | `sympy.solve()` |
| 2 | `\int` present | `sympy.integrate()` |
| 3 | `\frac{d}{d` or `'` (prime) | `sympy.diff()` |
| 4 | `\lim` present | `sympy.limit()` |
| 5 | fallback | `sympy.simplify()` |

`parse_latex` uses `sympy.parsing.latex.parse_latex()` with a fallback to manual cleaning (strip fences, expand `\frac`, etc.).

## Frontend Changes

- **Operation badge:** small pill above the result showing "Simplifying", "Solving equation", "Differentiating", "Integrating", or "Finding limit"
- **Steps list:** collapsible `<details>` element showing the steps array from `solution_data`
- **Solution rendering:** attempt `katex.renderToString(latex_result)` first; fall back to plain `solution` text if KaTeX throws
- No layout changes; all additions go inside the existing `.result-success` div

## Error Handling

| Scenario | Behavior |
|----------|----------|
| Gemini 401/403 | "API key invalid" message in result panel |
| Gemini 429 | "Rate limit exceeded" message |
| Gemini timeout | "Request timed out" message |
| Gemini returns non-math | SymPy parse fails → "Could not parse expression" |
| SymPy parse error | Return error, suggest clearer drawing |
| SymPy no closed-form | Fall back to `simplify()`, note in steps |
| SymPy timeout (>10s) | `threading.Timer` kill → "Expression too complex" |
| Empty canvas | Frontend blocks submission before sending |
| R Shiny offline | iframe shows connection error naturally |

## Dependencies

**Added to `requirements.txt`:**
- `sympy`

**Removed:** nothing (existing deps kept for compatibility)

**Gemini SDK fix:** Update `genai.configure` + `GenerativeModel` calls to match current `google-generativeai` ≥ 0.8 API. The model stays `gemini-1.5-flash`.

## What Is Not Changing

- R Shiny explorer and its R files
- Canvas drawing UX (pen, eraser, undo/redo, color, thickness)
- Backend CORS config
- Frontend health-check banner logic
- `.env` format (`GEMINI_API_KEY`, `R_API_URL` kept for backwards compat even though R solver is gone)

# MathBoard — Full Design Spec
**Date:** 2026-04-23

## Overview

MathBoard is a web application where users draw mathematical expressions on a canvas, which are then recognized (via Google Gemini), converted to LaTeX, and symbolically solved (via R/Ryacas). This spec covers: fixing all existing bugs, completing the frontend result display, and adding a new R Shiny dataset explorer powered by HASYv2.

---

## Architecture

Four services run concurrently:

| Service | Port | Language | Role |
|---|---|---|---|
| React (Vite) | 5173 | JavaScript | Drawing canvas, results display, nav |
| FastAPI | 8000 | Python | Image upload, Gemini OCR, orchestration |
| R Plumber | 8003 | R | `/solve` endpoint — LaTeX → symbolic solution |
| R Shiny | 3838 | R | HASYv2 dataset explorer dashboard |

### Data Flow — Math Solver

1. User draws on canvas → clicks **Convert**
2. React POSTs canvas image (PNG blob) to `POST /convert` on FastAPI
3. FastAPI sends image to Gemini 1.5 Flash → receives LaTeX string
4. FastAPI POSTs `{latex}` to R Plumber `POST /solve` → receives `{latex, solution, status}`
5. FastAPI returns `{latex, solution}` JSON to React
6. React renders LaTeX via KaTeX and displays the symbolic solution

### Data Flow — Dataset Explorer

- React nav tab "Dataset Explorer" renders a full-height iframe to `http://localhost:3838`
- Shiny app loads independently; no communication with FastAPI required

---

## Frontend (`frontend/mathboard/src/`)

### Layout

- **Top nav bar**: two tabs — "Math Solver" | "Dataset Explorer"
- **Math Solver view** (stacked):
  - Full-width canvas with existing tools (pen, eraser, color, thickness, undo, clear)
  - "Convert" button below canvas
  - Results panel slides in below after conversion:
    - LaTeX rendered via KaTeX
    - Symbolic solution text
    - Error message on failure
  - Loading spinner shown during API call
- **Dataset Explorer view**: full-height `<iframe src="http://localhost:3838" />`

### Changes to `App.jsx`

- Add `activeTab` state (`'solver'` | `'explorer'`)
- Add nav bar component with tab switching
- Fix `handleConvert()`:
  - Wrap in try/catch with loading state
  - POST canvas blob to `http://127.0.0.1:8000/convert`
  - Receive `{latex, solution}` JSON
  - Set result state; display in results panel
- Add KaTeX render call for LaTeX string
- Add `<iframe>` for Dataset Explorer tab

### New Dependencies

- `katex` — LaTeX rendering in browser

---

## Python Backend (`backend/main.py`)

### Changes

- Load Gemini API key from environment variable via `python-dotenv` (not hardcoded)
- Return `{latex, solution}` JSON from `/convert` endpoint (currently returns nothing useful)
- Keep existing CORS, file upload, and R Plumber proxy logic

### `requirements.txt` — Add

```
Pillow
google-generativeai
python-dotenv
```

### `.env` (new file, gitignored)

```
GEMINI_API_KEY=your_key_here
```

---

## R Components (`backend/`)

### Plumber (`plumber.r`) — Improvements

Fix `clean_latex_for_yacas()` to handle:

| LaTeX input | Yacas output |
|---|---|
| `\frac{a}{b}` | `(a)/(b)` |
| `x^{2}` | `x^2` |
| `\sqrt{x}` | `sqrt(x)` |
| `\alpha` | `alpha` |
| `\pi` | `pi` |
| `\times` | `*` |
| `\cdot` | `*` |

Keep existing `/solve` endpoint structure and error handling.

### Shiny App (`backend/shiny_app.R`) — New

**Dataset:** HASYv2 — 168,233 PNG images (32×32px), 369 math symbol classes, CSV labels.
Downloaded from the official HASYv2 release and stored at `backend/data/hasy/`.

**Layout:** `fluidPage` with `dashboardPage` (shinydashboard):

1. **KPI cards row** (top):
   - Total Samples: 168,233
   - Symbol Classes: 369
   - Image Size: 32×32 px

2. **Left panel**: `ggplot2` horizontal bar chart
   - Input: slider for top-N classes (range 10–50, default 20)
   - Input: category filter dropdown (All / Digits / Greek / Operators / Other)
   - Output: frequency bar chart, sorted descending

3. **Right panel**: symbol sample image grid
   - Input: dropdown to select a symbol class
   - Output: 3×3 grid of random sample images from that class rendered via `renderImage`

**R packages required:** `shiny`, `shinydashboard`, `ggplot2`, `dplyr`, `png`, `grid`

Runs on port 3838 via `shiny::runApp(port=3838)`.

---

## Bug Fixes & Security

| Issue | Fix |
|---|---|
| Gemini API key hardcoded in `main.py` | Move to `backend/.env`, load with `python-dotenv` |
| `requirements.txt` missing `Pillow`, `google-generativeai` | Add both + `python-dotenv` |
| `handleConvert()` discards API response | Fix to parse JSON and update result state |
| No results UI in frontend | Add results panel below canvas |
| LaTeX→Yacas regex too simplistic | Expand `clean_latex_for_yacas()` with table above |
| No root `.gitignore` | Add `.gitignore` covering `.env`, `.venv`, `node_modules`, `__pycache__`, `.superpowers`, `backend/data/` |
| No setup instructions | Add `SETUP.md` at repo root with step-by-step startup for all 4 services |

---

## Dataset Setup

HASYv2 is downloaded separately (not committed to git — too large):

1. Download from the official HASYv2 Zenodo page (search "HASYv2 dataset zenodo" — the record contains `HASYv2.tar.bz2`)
2. Extract to `backend/data/hasy/`
3. Expected structure:
   ```
   backend/data/hasy/
   ├── hasy-data/          # 168,233 PNG files
   ├── hasy-data-labels.csv
   └── README.md
   ```

The Shiny app reads `hasy-data-labels.csv` at startup to build the symbol index.

---

## Out of Scope

- Docker/containerization
- User accounts or persistence
- Deploying to a remote server
- Training any ML model locally
- Replacing Gemini with a local inference model

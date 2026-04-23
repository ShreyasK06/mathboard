# MathBoard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix MathBoard end-to-end (canvas → Gemini OCR → R symbolic solver → displayed result) and add a new R Shiny HASYv2 dataset explorer embedded in the UI.

**Architecture:** Four services: React (5173) for the drawing UI, FastAPI (8000) for orchestration, R Plumber (8003) for symbolic solving, R Shiny (3838) for dataset exploration. React embeds Shiny via iframe in a "Dataset Explorer" tab.

**Tech Stack:** React 19 + Vite + KaTeX, FastAPI + python-dotenv + Pillow + google-generativeai, R + Plumber + Ryacas + Shiny + shinydashboard + ggplot2 + dplyr, HASYv2 dataset (168,233 PNGs, 369 math symbol classes).

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| Create | `.gitignore` | Exclude .env, .venv, node_modules, data/, __pycache__, .superpowers |
| Create | `SETUP.md` | Step-by-step startup for all 4 services |
| Create | `backend/.env.example` | Template showing required env vars |
| Create | `backend/.env` | Actual secrets (gitignored) |
| Modify | `backend/requirements.txt` | Add Pillow, google-generativeai, python-dotenv, pytest, httpx |
| Modify | `backend/main.py` | Load API key from env; return {latex, solution_data} |
| Create | `backend/tests/test_main.py` | pytest tests for /convert endpoint |
| Modify | `backend/plumber.r` | Expand clean_latex_for_yacas() |
| Create | `backend/tests/test_plumber.R` | testthat tests for clean_latex_for_yacas |
| Create | `backend/shiny_app.R` | R Shiny HASYv2 dashboard |
| Create | `backend/run_shiny.r` | Starts Shiny on port 3838 |
| Modify | `frontend/mathboard/package.json` | Add katex dependency |
| Modify | `frontend/mathboard/src/App.jsx` | Nav tabs, fix handleConvert, results panel |
| Modify | `frontend/mathboard/src/App.css` | Nav, results panel, explorer iframe styles |

---

## Task 1: Project scaffolding

**Files:**
- Create: `.gitignore`
- Create: `backend/.env.example`
- Create: `backend/.env`

- [ ] **Step 1: Create root .gitignore**

Create `C:\Users\Shreyas Kakkar\Documents\GitHub\mathboard\.gitignore`:

```
# Python
backend/.env
backend/.venv/
backend/__pycache__/
backend/*.pyc
backend/images/

# R
*.Rhistory
*.RData

# Node
frontend/mathboard/node_modules/
frontend/mathboard/dist/

# Dataset (too large for git)
backend/data/

# Brainstorming artifacts
.superpowers/
```

- [ ] **Step 2: Create .env.example**

Create `backend/.env.example`:

```
GEMINI_API_KEY=your_gemini_api_key_here
```

- [ ] **Step 3: Create .env with your actual key**

Create `backend/.env` (this file is gitignored — never commit it):

```
GEMINI_API_KEY=AIzaSyCmJQWrvAgOnJMosyBZvUJFP_S63ZQTvSU
```

- [ ] **Step 4: Commit scaffolding**

```bash
git add .gitignore backend/.env.example
git commit -m "chore: add gitignore and env template"
```

---

## Task 2: Fix backend dependencies

**Files:**
- Modify: `backend/requirements.txt`

- [ ] **Step 1: Replace requirements.txt with complete list**

```
fastapi
uvicorn
python-multipart
requests
Pillow
google-generativeai
python-dotenv
pytest
httpx
```

- [ ] **Step 2: Install the new dependencies**

Run from `backend/` with the venv active:

```bash
cd backend
.venv/Scripts/activate
pip install -r requirements.txt
```

Expected: pip resolves and installs all packages. No errors.

- [ ] **Step 3: Commit**

```bash
git add backend/requirements.txt
git commit -m "fix: add missing backend dependencies"
```

---

## Task 3: Fix backend API key security

**Files:**
- Modify: `backend/main.py` lines 1–22

- [ ] **Step 1: Replace hardcoded key with dotenv loading**

Replace the top of `backend/main.py` (lines 1–22) with:

```python
import os
import io
import requests
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

R_API_URL = "http://localhost:8003/solve"
```

- [ ] **Step 2: Verify the server still starts**

```bash
cd backend
.venv/Scripts/activate
uvicorn main:app --reload
```

Expected output (last line): `INFO:     Application startup complete.`

Stop the server with Ctrl+C.

- [ ] **Step 3: Commit**

```bash
git add backend/main.py
git commit -m "fix: load Gemini API key from .env instead of hardcoding"
```

---

## Task 4: Fix /convert response shape

**Files:**
- Modify: `backend/main.py` lines 30–78

The current `handleConvert()` in React discards the response. The fix: ensure the endpoint returns a clean `{status, latex, solution_data}` JSON — it already does, but we need the `solution_data` to reliably contain a `solution` key the frontend can read. No changes to the existing return are needed — the R plumber already returns `solution` in its response. This task verifies the shape is correct.

- [ ] **Step 1: Verify /convert response shape matches what frontend will expect**

With the backend and R Plumber running, test the endpoint manually:

```bash
# In a second terminal (R Plumber must be running first: Rscript backend/run_r.r)
cd backend && .venv/Scripts/activate
uvicorn main:app --reload
```

```bash
# In a third terminal — send a test image
curl -X POST http://127.0.0.1:8000/convert \
  -F "file=@images/debug_canvas.png"
```

Expected response shape:
```json
{
  "status": "success",
  "latex": "x^2 + 1",
  "solution_data": {
    "original_latex": "x^2 + 1",
    "yacas_parsed": "x^2 + 1",
    "solution": "x^2+1",
    "status": "success"
  }
}
```

If R Plumber is offline, `solution_data` will contain `{"error": "R Compute Engine offline."}` — that is handled correctly by the frontend error display we'll add in Task 9.

- [ ] **Step 2: No code change needed — commit a note instead**

```bash
git commit --allow-empty -m "docs: verify /convert response shape is frontend-compatible"
```

---

## Task 5: Test FastAPI /convert endpoint

**Files:**
- Create: `backend/tests/__init__.py`
- Create: `backend/tests/test_main.py`

- [ ] **Step 1: Create tests package**

Create `backend/tests/__init__.py` (empty file).

- [ ] **Step 2: Write the test file**

Create `backend/tests/test_main.py`:

```python
import os
import io
import sys

os.environ.setdefault("GEMINI_API_KEY", "test-fake-key")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from PIL import Image

from main import app

client = TestClient(app)


def _make_png_bytes():
    img = Image.new("RGB", (50, 50), color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def test_convert_returns_latex_and_solution():
    mock_gemini = MagicMock()
    mock_gemini.text = "x^2 + 1"

    mock_r = MagicMock()
    mock_r.status_code = 200
    mock_r.json.return_value = {"solution": "x^2+1", "status": "success"}
    mock_r.raise_for_status = MagicMock()

    with patch("main.model.generate_content", return_value=mock_gemini), \
         patch("main.requests.post", return_value=mock_r):
        response = client.post(
            "/convert",
            files={"file": ("test.png", _make_png_bytes(), "image/png")},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["latex"] == "x^2 + 1"
    assert "solution_data" in data


def test_convert_returns_error_on_invalid_image():
    response = client.post(
        "/convert",
        files={"file": ("bad.png", io.BytesIO(b"not an image"), "image/png")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "error" in data


def test_convert_handles_r_offline():
    mock_gemini = MagicMock()
    mock_gemini.text = "x + 1"

    with patch("main.model.generate_content", return_value=mock_gemini), \
         patch("main.requests.post", side_effect=Exception("Connection refused")):
        response = client.post(
            "/convert",
            files={"file": ("test.png", _make_png_bytes(), "image/png")},
        )

    assert response.status_code == 200
    data = response.json()
    assert "error" in data["solution_data"]
```

- [ ] **Step 3: Run tests to verify they pass**

```bash
cd backend
.venv/Scripts/activate
pytest tests/test_main.py -v
```

Expected output:
```
tests/test_main.py::test_convert_returns_latex_and_solution PASSED
tests/test_main.py::test_convert_returns_error_on_invalid_image PASSED
tests/test_main.py::test_convert_handles_r_offline PASSED
3 passed in X.XXs
```

- [ ] **Step 4: Commit**

```bash
git add backend/tests/
git commit -m "test: add pytest suite for /convert endpoint"
```

---

## Task 6: Improve R LaTeX→Yacas conversion + testthat

**Files:**
- Modify: `backend/plumber.r` lines 7–15
- Create: `backend/tests/test_plumber.R`

- [ ] **Step 1: Write failing R tests first**

Create `backend/tests/test_plumber.R`:

```r
library(testthat)
source("plumber.r")  # run from backend/ directory

test_that("frac is converted to division", {
  expect_equal(clean_latex_for_yacas("\\frac{x}{2}"), "(x)/(2)")
})

test_that("braced exponent is unwrapped", {
  expect_equal(clean_latex_for_yacas("x^{2}"), "x^2")
})

test_that("sqrt is converted", {
  expect_equal(clean_latex_for_yacas("\\sqrt{x}"), "sqrt(x)")
})

test_that("Greek letters are converted", {
  expect_equal(clean_latex_for_yacas("\\alpha"), "alpha")
  expect_equal(clean_latex_for_yacas("\\pi"), "pi")
  expect_equal(clean_latex_for_yacas("\\theta"), "theta")
})

test_that("multiplication operators are converted", {
  expect_equal(clean_latex_for_yacas("x \\times y"), "x * y")
  expect_equal(clean_latex_for_yacas("x \\cdot y"), "x * y")
})
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd backend
Rscript -e "testthat::test_file('tests/test_plumber.R')"
```

Expected: multiple FAILED results (current regex is too simple).

- [ ] **Step 3: Replace clean_latex_for_yacas in plumber.r**

Replace lines 7–15 of `backend/plumber.r`:

```r
clean_latex_for_yacas <- function(latex_str) {
  s <- latex_str
  # \frac{a}{b} → (a)/(b)  — must run before generic brace removal
  s <- gsub("\\\\frac\\{([^}]*)\\}\\{([^}]*)\\}", "(\\1)/(\\2)", s)
  # x^{2} → x^2
  s <- gsub("\\^\\{([^}]*)\\}", "^\\1", s)
  # \sqrt{x} → sqrt(x)
  s <- gsub("\\\\sqrt\\{([^}]*)\\}", "sqrt(\\1)", s)
  # Greek letters
  s <- gsub("\\\\alpha", "alpha", s)
  s <- gsub("\\\\beta",  "beta",  s)
  s <- gsub("\\\\gamma", "gamma", s)
  s <- gsub("\\\\theta", "theta", s)
  s <- gsub("\\\\pi",    "pi",    s)
  # Operators
  s <- gsub("\\\\times", "*", s)
  s <- gsub("\\\\cdot",  "*", s)
  s <- gsub("\\\\div",   "/", s)
  # Remove remaining unknown backslash commands
  s <- gsub("\\\\[a-zA-Z]+", "", s)
  # Convert remaining braces to parentheses
  s <- gsub("\\{", "(", s)
  s <- gsub("\\}", ")", s)
  return(s)
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd backend
Rscript -e "testthat::test_file('tests/test_plumber.R')"
```

Expected:
```
[ FAIL 0 | WARN 0 | SKIP 0 | PASS 5 ]
```

- [ ] **Step 5: Commit**

```bash
git add backend/plumber.r backend/tests/test_plumber.R
git commit -m "fix: expand LaTeX-to-Yacas conversion; add testthat suite"
```

---

## Task 7: Add KaTeX to frontend

**Files:**
- Modify: `frontend/mathboard/package.json`

- [ ] **Step 1: Install KaTeX**

```bash
cd frontend/mathboard
npm install katex
```

Expected: `package.json` dependencies now includes `"katex": "^0.16.x"`.

- [ ] **Step 2: Verify KaTeX installed**

```bash
ls node_modules/katex/dist/katex.min.css
```

Expected: file exists.

- [ ] **Step 3: Commit**

```bash
git add frontend/mathboard/package.json frontend/mathboard/package-lock.json
git commit -m "feat: add katex for LaTeX rendering in frontend"
```

---

## Task 8: Add nav tabs to frontend

**Files:**
- Modify: `frontend/mathboard/src/App.jsx`
- Modify: `frontend/mathboard/src/App.css`

- [ ] **Step 1: Add activeTab state and nav bar to App.jsx**

Add `activeTab` state after the existing `tool` state (after line 15 of `App.jsx`):

```jsx
const [activeTab, setActiveTab] = useState("solver"); // "solver" | "explorer"
```

Replace the `return (` block opening (line 194) — replace:
```jsx
  return (
    <div className="page">
      <h1>MathBoard</h1>
```

With:
```jsx
  return (
    <div className="page">
      <h1>MathBoard</h1>

      <nav className="nav-tabs">
        <button
          className={`nav-tab ${activeTab === "solver" ? "active" : ""}`}
          onClick={() => setActiveTab("solver")}
        >
          Math Solver
        </button>
        <button
          className={`nav-tab ${activeTab === "explorer" ? "active" : ""}`}
          onClick={() => setActiveTab("explorer")}
        >
          Dataset Explorer
        </button>
      </nav>
```

After the closing `</div>` of `boardWrap` (before the final `</div>`), add the explorer iframe:

```jsx
      {activeTab === "explorer" && (
        <iframe
          src="http://localhost:3838"
          className="explorer-frame"
          title="HASYv2 Dataset Explorer"
        />
      )}
```

Wrap the existing toolbar `<div>` and boardWrap `<div>` (lines 198–274 of the original file) in a `{activeTab === "solver" && ...}` conditional. The structure should be:

```jsx
      {activeTab === "solver" && (
        <>
          <div className="toolbar">
            <button onClick={undo} disabled={historyIndex <= 0}>Undo</button>
            <button onClick={redo} disabled={historyIndex >= history.length - 1}>Redo</button>
            <button onClick={clearCanvas}>Clear</button>
            <div className="divider" />
            <div className="toolGroup">
              <button className={tool === "pen" ? "active" : ""} onClick={() => setTool("pen")}>Pen</button>
              <button className={`${tool === "eraser" ? "active" : ""} eraser-btn`} onClick={() => setTool("eraser")}>Eraser</button>
            </div>
            <div className="divider" />
            <label className="control">
              Thickness
              <input type="range" min="1" max="30"
                value={tool === "eraser" ? eraserSize : penSize}
                onChange={(e) => {
                  const newSize = Number(e.target.value);
                  if (tool === "eraser") setEraserSize(newSize);
                  else setPenSize(newSize);
                }}
              />
              <span className="value">{tool === "eraser" ? eraserSize : penSize}px</span>
            </label>
            <label className="control">
              Color
              <input type="color" value={penColor}
                onChange={(e) => { setPenColor(e.target.value); setTool("pen"); }}
                disabled={tool === "eraser"}
                title={tool === "eraser" ? "Switch to Pen to change color" : ""}
              />
            </label>
            <button className="primary" onClick={handleConvert} disabled={isLoading}>
              {isLoading ? "Converting…" : "Convert"}
            </button>
          </div>

          <div className="boardWrap">
            <canvas
              ref={canvasRef}
              className="board"
              onPointerDown={handlePointerDown}
              onPointerMove={handlePointerMove}
              onPointerUp={handlePointerUp}
              onPointerCancel={handlePointerUp}
              onPointerLeave={handlePointerUp}
            />
          </div>
        </>
      )}
```

- [ ] **Step 2: Add nav tab styles to App.css**

Append to the end of `frontend/mathboard/src/App.css`:

```css
.nav-tabs {
  display: flex;
  gap: 4px;
  padding: 0;
  margin-bottom: 20px;
  border-bottom: 2px solid #e8e8e8;
  width: 100%;
  max-width: calc(100vw - 16px);
}

.nav-tab {
  padding: 10px 28px;
  border: 2px solid transparent;
  border-bottom: none;
  background: #f5f5f5;
  color: #666;
  border-radius: 8px 8px 0 0;
  cursor: pointer;
  font-weight: 500;
  font-size: 14px;
  transition: all 200ms ease;
  box-shadow: none;
}

.nav-tab:hover {
  background: #ededff;
  color: #667eea;
  transform: none;
  box-shadow: none;
  border-color: transparent;
}

.nav-tab.active {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border-color: #667eea;
  box-shadow: 0 -2px 8px rgba(102, 126, 234, 0.2);
}

.explorer-frame {
  width: calc(100vw - 16px);
  height: calc(100vh - 160px);
  border: 2px solid #e8e8e8;
  border-radius: 12px;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.12);
}
```

- [ ] **Step 3: Start dev server and verify tabs render**

```bash
cd frontend/mathboard
npm run dev
```

Open `http://localhost:5173`. Confirm:
- "Math Solver" and "Dataset Explorer" tabs appear
- Clicking "Dataset Explorer" shows an iframe (may show connection refused for now — that's fine)
- Clicking "Math Solver" shows the canvas

- [ ] **Step 4: Commit**

```bash
git add frontend/mathboard/src/App.jsx frontend/mathboard/src/App.css
git commit -m "feat: add Math Solver / Dataset Explorer nav tabs"
```

---

## Task 9: Fix handleConvert and add results panel

**Files:**
- Modify: `frontend/mathboard/src/App.jsx`
- Modify: `frontend/mathboard/src/App.css`

- [ ] **Step 1: Add KaTeX import and result state to App.jsx**

At the top of `App.jsx`, after `import "./App.css";`, add:

```jsx
import katex from "katex";
import "katex/dist/katex.min.css";
```

Add result state after the `activeTab` state:

```jsx
const [isLoading, setIsLoading] = useState(false);
const [result, setResult] = useState(null);      // { latex, solution }
const [resultError, setResultError] = useState(null);
```

- [ ] **Step 2: Replace handleConvert with fixed version**

Replace the entire `handleConvert` function (lines 176–191 of original file):

```jsx
  const handleConvert = async () => {
    const canvas = canvasRef.current;
    setIsLoading(true);
    setResult(null);
    setResultError(null);

    try {
      const blob = await new Promise((resolve) =>
        canvas.toBlob(resolve, "image/png")
      );
      const formData = new FormData();
      formData.append("file", blob);

      const res = await fetch("http://127.0.0.1:8000/convert", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) throw new Error(`Server error: ${res.status}`);

      const data = await res.json();
      if (data.error) throw new Error(data.error);

      const solutionText =
        data.solution_data?.solution ||
        data.solution_data?.error ||
        "No solution returned";

      setResult({ latex: data.latex, solution: solutionText });
    } catch (err) {
      setResultError(err.message);
    } finally {
      setIsLoading(false);
    }
  };
```

- [ ] **Step 3: Add results panel JSX below the boardWrap**

Inside the `{activeTab === "solver" && ...}` block, after the closing `</div>` of `boardWrap`, add:

```jsx
          {isLoading && (
            <div className="results-panel">
              <div className="loading-spinner">Recognizing and solving…</div>
            </div>
          )}

          {resultError && (
            <div className="results-panel">
              <div className="error-display">Error: {resultError}</div>
            </div>
          )}

          {result && (
            <div className="results-panel">
              <h3>LaTeX</h3>
              <div
                className="latex-display"
                dangerouslySetInnerHTML={{
                  __html: katex.renderToString(result.latex, {
                    throwOnError: false,
                    displayMode: true,
                  }),
                }}
              />
              <h3>Solution</h3>
              <div className="solution-display">{result.solution}</div>
            </div>
          )}
```

- [ ] **Step 4: Update the Convert button label**

Find the Convert button in the toolbar:
```jsx
        <button className="primary" onClick={handleConvert}>
          Convert (export PNG)
        </button>
```

Replace with:
```jsx
        <button className="primary" onClick={handleConvert} disabled={isLoading}>
          {isLoading ? "Converting…" : "Convert"}
        </button>
```

- [ ] **Step 5: Add results panel styles to App.css**

Append to `App.css`:

```css
.results-panel {
  width: calc(100vw - 16px);
  max-width: calc(100vw - 16px);
  background: white;
  border: 2px solid #e8e8e8;
  border-radius: 12px;
  padding: 20px 28px;
  margin-top: 16px;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
}

.results-panel h3 {
  margin: 0 0 10px;
  font-size: 0.75em;
  font-weight: 700;
  color: #999;
  text-transform: uppercase;
  letter-spacing: 0.8px;
}

.latex-display {
  background: #f8f7ff;
  border: 1px solid #e0deff;
  border-radius: 8px;
  padding: 16px;
  margin-bottom: 20px;
  text-align: center;
  overflow-x: auto;
  font-size: 1.2em;
}

.solution-display {
  font-size: 1.1em;
  color: #2d7a5e;
  font-weight: 600;
}

.error-display {
  color: #c0392b;
  background: #fdf3f2;
  border: 1px solid #f5c6c3;
  border-radius: 8px;
  padding: 14px 16px;
}

.loading-spinner {
  text-align: center;
  color: #667eea;
  padding: 16px;
  font-style: italic;
}
```

- [ ] **Step 6: Test the full Math Solver flow**

With FastAPI and R Plumber both running, open `http://localhost:5173`:
1. Draw "x^2 + 1" on the canvas
2. Click Convert
3. Confirm loading state appears briefly
4. Confirm LaTeX renders in the results panel
5. Confirm the solution appears below it
6. Draw something invalid; confirm the error displays cleanly

- [ ] **Step 7: Commit**

```bash
git add frontend/mathboard/src/App.jsx frontend/mathboard/src/App.css
git commit -m "feat: fix handleConvert and add results display panel with KaTeX"
```

---

## Task 10: Download HASYv2 dataset

**Files:**
- No code changes — manual data setup step

- [ ] **Step 1: Create the data directory**

```bash
mkdir -p backend/data/hasy
```

- [ ] **Step 2: Download HASYv2**

Search "HASYv2 dataset zenodo" — download `HASYv2.tar.bz2` from the official Zenodo record.

- [ ] **Step 3: Extract and move into place**

```bash
tar -xjf HASYv2.tar.bz2
cp -r HASYv2/hasy-data backend/data/hasy/hasy-data
cp HASYv2/hasy-data-labels.csv backend/data/hasy/hasy-data-labels.csv
```

- [ ] **Step 4: Verify structure**

```bash
ls backend/data/hasy/
# Expected: hasy-data/  hasy-data-labels.csv

head -3 backend/data/hasy/hasy-data-labels.csv
# Expected: path,symbol_id,latex
#           hasy-data/v2-00001.png,31,!
#           hasy-data/v2-00002.png,31,!
```

- [ ] **Step 5: Verify image count**

```bash
ls backend/data/hasy/hasy-data/ | wc -l
# Expected: ~168233
```

No commit — dataset is gitignored.

---

## Task 11: Build R Shiny dataset explorer

**Files:**
- Create: `backend/shiny_app.R`
- Create: `backend/run_shiny.r`

- [ ] **Step 1: Install required R packages**

Open an R console and run:

```r
install.packages(c("shiny", "shinydashboard", "ggplot2", "dplyr"))
```

- [ ] **Step 2: Write shiny_app.R**

Create `backend/shiny_app.R`:

```r
library(shiny)
library(shinydashboard)
library(ggplot2)
library(dplyr)

DATA_DIR <- "data/hasy"
IMAGE_DIR <- file.path(DATA_DIR, "hasy-data")
LABELS_CSV <- file.path(DATA_DIR, "hasy-data-labels.csv")

labels <- read.csv(LABELS_CSV, stringsAsFactors = FALSE)

freq <- labels %>%
  group_by(latex) %>%
  summarise(count = n(), .groups = "drop") %>%
  arrange(desc(count))

addResourcePath("hasy-images", IMAGE_DIR)

ui <- dashboardPage(
  skin = "purple",
  dashboardHeader(title = "HASYv2 Math Symbol Explorer"),
  dashboardSidebar(disable = TRUE),
  dashboardBody(
    tags$head(tags$style(HTML("
      .content-wrapper { background: #f4f6f9; }
      .small-box { border-radius: 10px; }
      .sample-img { width:64px; height:64px; margin:4px;
                    image-rendering:pixelated; border:1px solid #ddd;
                    border-radius:4px; }
    "))),

    fluidRow(
      valueBox(
        format(nrow(labels), big.mark = ","), "Total Samples",
        icon = icon("images"), color = "purple"
      ),
      valueBox(
        nrow(freq), "Symbol Classes",
        icon = icon("shapes"), color = "green"
      ),
      valueBox(
        "32 × 32", "Image Size (px)",
        icon = icon("expand-arrows-alt"), color = "yellow"
      )
    ),

    fluidRow(
      box(
        title = "Symbol Frequency Distribution", status = "primary",
        solidHeader = TRUE, width = 6,
        sliderInput("top_n", "Show top N classes:",
                    min = 10, max = 50, value = 20, step = 5),
        plotOutput("freq_chart", height = "380px")
      ),
      box(
        title = "Symbol Samples", status = "success",
        solidHeader = TRUE, width = 6,
        selectInput("symbol_class", "Select symbol class:",
                    choices = freq$latex, selected = freq$latex[1]),
        uiOutput("sample_grid")
      )
    )
  )
)

server <- function(input, output, session) {

  output$freq_chart <- renderPlot({
    top <- freq %>% head(input$top_n)
    ggplot(top, aes(x = reorder(latex, count), y = count)) +
      geom_bar(stat = "identity", fill = "#667eea", width = 0.7) +
      coord_flip() +
      labs(
        x = "Symbol (LaTeX label)",
        y = "Number of samples",
        title = paste("Top", input$top_n, "Most Frequent Symbols")
      ) +
      theme_minimal(base_size = 13) +
      theme(
        plot.title = element_text(face = "bold", color = "#444"),
        axis.text.y = element_text(size = 10)
      )
  })

  output$sample_grid <- renderUI({
    selected_rows <- labels %>% filter(latex == input$symbol_class)
    n_samples <- min(9, nrow(selected_rows))
    samples <- selected_rows %>% sample_n(n_samples)

    img_tags <- lapply(seq_len(nrow(samples)), function(i) {
      filename <- basename(samples$path[i])
      tags$img(
        src   = paste0("/hasy-images/", filename),
        class = "sample-img",
        title = samples$latex[i]
      )
    })

    tagList(
      p(
        style = "color:#888; font-size:13px; margin-bottom:8px;",
        paste(nrow(selected_rows), "total samples for", input$symbol_class)
      ),
      div(style = "display:flex; flex-wrap:wrap;", img_tags)
    )
  })
}

shinyApp(ui, server, options = list(port = 3838, host = "0.0.0.0", launch.browser = FALSE))
```

- [ ] **Step 3: Write run_shiny.r**

Create `backend/run_shiny.r` with this content (run from `backend/` directory):

```r
shiny::runApp("shiny_app.R", port = 3838, host = "0.0.0.0", launch.browser = FALSE)
```

- [ ] **Step 4: Start the Shiny app and verify**

```bash
cd backend
Rscript run_shiny.r
```

Expected output:
```
Listening on http://0.0.0.0:3838
```

Open `http://localhost:3838`. Confirm:
- Three KPI boxes at the top (168,233 / 369 / 32×32)
- Frequency bar chart renders with top 20 symbols
- Symbol dropdown populates
- Sample image grid shows 9 (or fewer) images for the selected symbol

- [ ] **Step 5: Test Dataset Explorer tab in the main app**

With Shiny running on 3838 and React on 5173, open `http://localhost:5173`. Click "Dataset Explorer" tab. Confirm the Shiny dashboard loads inside the iframe.

- [ ] **Step 6: Commit**

```bash
git add backend/shiny_app.R backend/run_shiny.r
git commit -m "feat: add R Shiny HASYv2 dataset explorer dashboard"
```

---

## Task 12: Write SETUP.md and final commit

**Files:**
- Create: `SETUP.md`

- [ ] **Step 1: Write SETUP.md**

Create `SETUP.md` at the repo root:

```markdown
# MathBoard — Setup & Running

## Prerequisites

- Python 3.10+
- R 4.x with packages: `plumber`, `Ryacas`, `jsonlite`, `shiny`, `shinydashboard`, `ggplot2`, `dplyr`, `testthat`
- Node.js 18+
- A Gemini API key (free at Google AI Studio)

## First-time setup

### 1. Clone and configure secrets

Copy the env template and fill in your key:

```
cp backend/.env.example backend/.env
# edit backend/.env and paste your GEMINI_API_KEY
```

### 2. Install Python dependencies

```
cd backend
python -m venv .venv
.venv/Scripts/activate      # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

### 3. Install Node dependencies

```
cd frontend/mathboard
npm install
```

### 4. Install R packages (first time only)

Open R and run:

```r
install.packages(c("plumber", "Ryacas", "jsonlite",
                   "shiny", "shinydashboard", "ggplot2", "dplyr", "testthat"))
```

### 5. Download HASYv2 dataset

Search "HASYv2 dataset zenodo" and download `HASYv2.tar.bz2`. Extract and place:

```
backend/data/hasy/hasy-data/         ← all PNG files here
backend/data/hasy/hasy-data-labels.csv
```

## Running all 4 services

Open four terminals, each from the repo root:

**Terminal 1 — React frontend:**
```
cd frontend/mathboard && npm run dev
```
→ http://localhost:5173

**Terminal 2 — FastAPI backend:**
```
cd backend && .venv/Scripts/activate && uvicorn main:app --reload
```
→ http://localhost:8000

**Terminal 3 — R Plumber (symbolic solver):**
```
cd backend && Rscript run_r.r
```
→ http://localhost:8003

**Terminal 4 — R Shiny (dataset explorer):**
```
cd backend && Rscript run_shiny.r
```
→ http://localhost:3838

## Running tests

**Python tests:**
```
cd backend && .venv/Scripts/activate && pytest tests/test_main.py -v
```

**R tests:**
```
cd backend && Rscript -e "testthat::test_file('tests/test_plumber.R')"
```
```

- [ ] **Step 2: Commit SETUP.md**

```bash
git add SETUP.md
git commit -m "docs: add SETUP.md with startup instructions for all 4 services"
```

- [ ] **Step 3: Final end-to-end smoke test**

Start all 4 services as described in SETUP.md. Then:

1. Open `http://localhost:5173`
2. Draw a simple expression (e.g., `2+2`) on the canvas
3. Click Convert — loading spinner appears, then results show LaTeX + solution
4. Switch to Dataset Explorer tab — Shiny dashboard loads in iframe
5. Change the symbol class dropdown — image grid updates
6. Adjust the top-N slider — bar chart updates

All 5 checks must pass before this task is complete.
```

- [ ] **Step 4: Tag the working state**

```bash
git tag v1.0-working
```

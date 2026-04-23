# MathBoard — Setup & Running

## What this is

MathBoard lets you draw a mathematical expression on a canvas. It sends the drawing to Google Gemini for OCR (handwriting recognition), converts the result to LaTeX, and passes it to an R symbolic solver. The result (LaTeX + solution) is displayed below the canvas.

A second tab — **Dataset Explorer** — shows an interactive R Shiny dashboard exploring the HASYv2 handwritten math symbol dataset (168,233 samples, 369 symbol classes).

## Prerequisites

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.10+ | FastAPI backend |
| R | 4.x | Plumber solver + Shiny dashboard |
| Node.js | 18+ | React frontend |
| Gemini API key | free | Handwriting OCR |

## First-time setup

### 1. Configure your Gemini API key

```
cp backend/.env.example backend/.env
```

Edit `backend/.env` and paste your key:
```
GEMINI_API_KEY=your_key_here
```

Get a free key at Google AI Studio (search "Google AI Studio API key").

### 2. Install Python dependencies

```
cd backend
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

### 3. Install Node dependencies

```
cd frontend/mathboard
npm install
```

### 4. Install R packages (first time only)

Open an R console and run:

```r
install.packages(c(
  "plumber", "Ryacas", "jsonlite",
  "shiny", "shinydashboard", "ggplot2", "dplyr", "testthat"
))
```

### 5. Download HASYv2 dataset (for Dataset Explorer tab)

Search **"HASYv2 dataset zenodo"** and download `HASYv2.tar.bz2`.

Extract and place the files at:
```
backend/data/hasy/hasy-data/          ← all 168,233 PNG files go here
backend/data/hasy/hasy-data-labels.csv
```

The Shiny app will show a friendly error message if the dataset is missing — the rest of the app works without it.

---

## Running all 4 services

Open **four separate terminals** from the repo root:

**Terminal 1 — React frontend:**
```
cd frontend/mathboard
npm run dev
```
→ Open http://localhost:5173

**Terminal 2 — FastAPI backend (image OCR + orchestration):**
```
cd backend
.venv\Scripts\activate
uvicorn main:app --reload
```
→ Runs at http://localhost:8000

**Terminal 3 — R Plumber (symbolic math solver):**
```
cd backend
Rscript run_r.r
```
→ Runs at http://localhost:8003

**Terminal 4 — R Shiny (dataset explorer):**
```
cd backend
Rscript run_shiny.r
```
→ Runs at http://localhost:3838

---

## Running tests

**Python tests (3 tests):**
```
cd backend
.venv\Scripts\activate
pytest tests/test_main.py -v
```

**R tests (8 tests for LaTeX→Yacas conversion):**
```
cd backend
Rscript -e "testthat::test_file('tests/test_plumber.R')"
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| "Gemini API key not configured" | Check `backend/.env` has a valid key |
| "R Compute Engine offline" | Start Terminal 3 (`Rscript run_r.r`) |
| Dataset Explorer shows "Dataset not found" | Download HASYv2 — see step 5 above |
| KaTeX renders raw LaTeX text | Gemini returned non-LaTeX; try drawing more clearly |

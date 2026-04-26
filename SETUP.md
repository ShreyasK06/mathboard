# MathBoard — Setup & Running

## What this is

MathBoard lets you draw a mathematical expression on a canvas. It sends the drawing to Google Gemini for OCR (handwriting recognition), converts the result to LaTeX, and passes it to a SymPy-based symbolic solver. The result (LaTeX + solution) is displayed below the canvas.

A second tab — **Dataset Explorer** — shows an interactive R Shiny dashboard exploring the HASYv2 handwritten math symbol dataset (168,233 samples, 369 symbol classes).

## Prerequisites

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.10+ | FastAPI backend + SymPy solver |
| R | 4.x | Shiny dataset explorer (optional) |
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

### 4. Install R packages (only if you want the Dataset Explorer tab)

Open an R console and run:

```r
install.packages(c(
  "shiny", "shinydashboard", "ggplot2", "dplyr"
))
```

The core math solver does not need R — it runs on Python/SymPy. R is only required for the optional HASYv2 dataset explorer.

### 5. Download HASYv2 dataset (for Dataset Explorer tab)

Search **"HASYv2 dataset zenodo"** and download `HASYv2.tar.bz2`.

Extract and place the files at:
```
backend/data/hasy/hasy-data/          ← all 168,233 PNG files go here
backend/data/hasy/hasy-data-labels.csv
```

The Shiny app will show a friendly error message if the dataset is missing — the rest of the app works without it.

---

## Running the app

You need **two terminals** (three if you want the Dataset Explorer tab).

**Terminal 1 — React frontend:**
```
cd frontend/mathboard
npm run dev
```
→ Open http://localhost:5173

**Terminal 2 — FastAPI backend (Gemini OCR + SymPy solver):**
```
cd backend
.venv\Scripts\activate
uvicorn main:app --reload
```
→ Runs at http://localhost:8000

**Terminal 3 (optional) — R Shiny dataset explorer:**
```
cd backend
Rscript run_shiny.r
```
→ Runs at http://localhost:3838

The Math Solver tab works without Terminal 3; only the Dataset Explorer tab needs it.

---

## Optional: train the local classifier

The backend has an optional local CNN that handles single-symbol drawings
without calling Gemini, falling back to Gemini for anything else. To enable it:

```
cd backend
.venv\Scripts\activate
python -m ml.train
```

~30 minutes on CPU. Produces `backend/ml/artifacts/{model.pt, classes.json, metrics.json}`.

The app keeps working without this step — every request just goes to Gemini.
See `backend/ml/README.md` for tuning details.

---

## Running tests

**Python tests (45 tests across solver, API, classifier, training):**
```
cd backend
.venv\Scripts\activate
pytest tests/ -v
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| "Gemini API key not configured" | Check `backend/.env` has a valid key |
| "Backend offline" banner in the UI | Start Terminal 2 (`uvicorn main:app --reload`) |
| Dataset Explorer shows "Dataset not found" | Download HASYv2 — see step 5 above |
| Dataset Explorer iframe is blank | Start Terminal 3 (`Rscript run_shiny.r`) |
| KaTeX renders raw LaTeX text | Gemini returned non-LaTeX; try drawing more clearly |

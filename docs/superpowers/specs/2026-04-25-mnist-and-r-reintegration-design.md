# MNIST Hybrid + R Reintegration — Design

**Date:** 2026-04-25
**Status:** Approved (brainstorming complete)
**Owner:** Shreyas Kakkar
**Builds on:** [`2026-04-25-local-symbol-classifier-design.md`](2026-04-25-local-symbol-classifier-design.md)

## Context

The first iteration of the local classifier (now shipped) handles math symbols
from HASYv2's top-100 classes — but those classes are dominated by Greek letters,
operators, and math notation. They contain almost no digits or Latin letters,
so digit drawings still always fall through to Gemini.

Separately, R was removed from the project when the SymPy solver replaced the
old R Plumber solver. The user wants R back in a load-bearing role.

This spec adds both improvements as one cohesive change: extend the local
classifier to recognize digits via MNIST, and bring R back as a parallel
symbolic solver plus a useful Shiny dashboard.

## Goals

- Local classifier confidently classifies handwritten digits 0-9.
- Every solved expression is cross-checked by an independent symbolic solver
  (Ryacas in R), with disagreement surfaced to the user.
- Replace the current static HASYv2 explorer dashboard with a multi-panel
  dashboard showing real, current, useful information about the model and
  the live request stream.

## Non-goals

- Letter recognition (a-z, A-Z). Out of scope for this iteration.
- A dedicated dashboard for HASYv2 itself. The static dataset viewer is
  removed, not just demoted.
- Persisting activity history beyond the most recent 5,000 requests.
- Real-time push (WebSocket) updates to the dashboard. 5-second polling is fine.
- Auth on the dashboard — it runs on localhost only.
- Recomputing the confusion matrix live; only the training-time matrix is shown.

## Architecture

```
┌────────────┐    ┌──────────────────┐    ┌─────────────┐
│  React UI  │───►│  FastAPI (8000)  │───►│ Local CNN   │
│            │    │     /convert     │    │ (preferred) │
└────────────┘    │                  │    └─────────────┘
                  │  parallel solve: │    ┌─────────────┐
                  │  - SymPy         │───►│   Gemini    │
                  │  - Ryacas        │    └─────────────┘
                  │  ↕ logs every    │    ┌─────────────┐
                  │  request         │───►│   SymPy     │
                  │                  │    └─────────────┘
                  │                  │    ┌─────────────┐
                  │                  │───►│ R Plumber   │
                  │                  │    │ (8003)      │
                  │                  │    │  → Ryacas   │
                  └──────────────────┘    └─────────────┘
                       │
                       ▼
                  ┌─────────────┐
                  │ activity.db │
                  │ (SQLite)    │
                  └──────┬──────┘
                         │
                  ┌──────▼──────┐
                  │ R Shiny     │
                  │ (3838)      │
                  │ reads log + │
                  │ metrics.json│
                  │ confusion.json
                  └──────▲──────┘
                         │ iframe
                  ┌──────┴──────┐
                  │ React tab   │
                  │ "Model &    │
                  │  Activity"  │
                  └─────────────┘
```

Two R processes run in parallel, both optional:

- **Plumber on :8003** — `/solve_ryacas` cross-checks every solved expression.
- **Shiny on :3838** — the dashboard.

If either is unreachable, the app degrades gracefully:
- Plumber down → response shows `agreement: "ryacas_unavailable"` and the UI hides the agreement line.
- Shiny down → the iframe in the React tab shows a connection-error placeholder; everything else still works.

## Components

### 1. MNIST hybrid for digits

#### Data

- HASYv2 top-100 classes (existing).
- MNIST 10 digit classes, labels `"0"`–`"9"` matching HASYv2's convention.
- Combined: 110 classes, ~187k training samples.

#### Acquisition

MNIST is downloaded automatically by `train.py` on first run if not present at `backend/data/mnist/`. The standard 4 IDX files (~12 MB total) are fetched from a CDN. We parse IDX with a small in-repo parser to avoid pulling in `torchvision`.

URLs (all `Yann LeCun's MNIST mirror or equivalent`):
- `train-images-idx3-ubyte.gz` (~10 MB)
- `train-labels-idx1-ubyte.gz` (~30 KB)
- `t10k-images-idx3-ubyte.gz` (~1.6 MB)
- `t10k-labels-idx1-ubyte.gz` (~5 KB)

(Train and test splits are merged into a single 70k-sample pool, then re-split with the same stratified 85/15 train/val split as HASYv2 to keep the methodology uniform.)

#### Preprocessing alignment

| Property | HASYv2 | MNIST | After alignment |
|---|---|---|---|
| Size | 32×32 | 28×28 | 32×32 (MNIST resized via bilinear) |
| Polarity | Black ink on white bg | White ink on black bg | Black ink on white bg (MNIST inverted with `255 - pixel`) |
| Pixel range | [0, 1] | [0, 1] | [0, 1] |
| Normalization | mean/std from combined train set | same | same |

The combined dataset's `mean` and `std` are recomputed during training and saved to `metrics.json` so inference normalization stays in sync.

#### Class imbalance

HASYv2 top-100 averages ~1170 samples/class; MNIST is ~7000/class. Without correction, batches would be dominated by digits.

Fix: `torch.utils.data.WeightedRandomSampler` with per-sample weight = `1 / class_count`. Each epoch sees roughly equal samples per class. Sampler is seeded for reproducibility.

#### Module layout

```
backend/ml/
  idx_parser.py     — read MNIST IDX files into numpy arrays (~30 lines)
  datasets.py       — HasyDataset, MnistDataset, CombinedDataset classes
  train.py          — orchestrates download + combine + train; new --include-mnist flag (default true)
```

`MnistDataset` downloads on construction if files missing (idempotent).
`CombinedDataset` concatenates HASYv2 + MNIST and exposes a unified label space.
`train.py`'s training-image loop still uses the same model and same hyperparameters — only the dataset changes.

### 2. R Plumber cross-solver

#### Plumber API (`backend/plumber.R`)

One endpoint plus health probe:

```r
#* @get /health
function() { list(status = "ok") }

#* @get /solve_ryacas
#* @param latex character
function(latex = "") {
  tryCatch({
    yacas_expr <- latex_to_yacas(latex)        # converter from prior R code
    op <- detect_operation(latex)              # "solve"|"integrate"|"differentiate"|"limit"|"simplify"
    raw <- run_ryacas(yacas_expr, op)
    list(
      status = "success",
      operation = op,
      solution = raw$str,
      latex_result = raw$latex
    )
  }, error = function(e) {
    list(status = "failed", error = conditionMessage(e))
  })
}
```

Listens on port 8003. Started via `Rscript run_plumber.r`.

`latex_to_yacas` and `detect_operation` are taken from the prior R Plumber code that was removed in commit 77b31d0. They worked previously and have existing test coverage.

#### Python client (`backend/ryacas_client.py`)

```python
@dataclass
class RyacasResult:
    status: str            # "success" | "failed"
    solution: str | None
    latex_result: str | None
    error: str | None

def cross_solve(latex: str, timeout_s: float = 3.0) -> RyacasResult | None:
    """Returns None if Plumber is unreachable. Never raises."""
```

Behavior:
- HTTP GET `http://127.0.0.1:8003/solve_ryacas?latex=...`
- 3-second timeout. Network failure or timeout → returns `None`.
- 5xx or `status == "failed"` → returns `RyacasResult(status="failed", error=msg)`.
- 200 + `status == "success"` → returns full result.

Module-level: cache the "Plumber-available" check for 30 seconds to avoid hammering health every request.

#### Integration in `main.py`

After local-or-Gemini OCR produces `latex_string`, call SymPy and Ryacas in parallel:

```python
import concurrent.futures

with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
    sympy_future = ex.submit(solver.solve_expression, latex_string)
    ryacas_future = ex.submit(ryacas_client.cross_solve, latex_string)
    solution_data = sympy_future.result()
    ryacas_result = ryacas_future.result()  # respects its own 3s timeout

solution_data["ryacas"] = (
    None if ryacas_result is None
    else {
        "status": ryacas_result.status,
        "solution": ryacas_result.solution,
        "latex_result": ryacas_result.latex_result,
    }
)
solution_data["agreement"] = _compute_agreement(solution_data, ryacas_result)
```

`_compute_agreement` returns one of `"match"`, `"differ"`, `"ryacas_unavailable"`, `"ryacas_error"`. Comparison is on the `latex_result` field after a tiny normalization pass (strip whitespace, normalize `\\frac{...}{...}` form, strip trailing zeros). Differences in trivial formatting count as `match`; differences in actual values count as `differ`.

#### Frontend changes

Below the solution KaTeX block in `App.jsx`:

```jsx
{convertResult.agreement === "match" && (
  <div className="agreement agreement-match">✓ Cross-checked with Ryacas (agree)</div>
)}
{convertResult.agreement === "differ" && (
  <div className="agreement agreement-differ">
    ⚠ Ryacas got: <span dangerouslySetInnerHTML={{ __html: katex.renderToString(convertResult.ryacasLatex) }} />
  </div>
)}
{/* "ryacas_unavailable" and "ryacas_error" render nothing */}
```

`convertResult` gains `agreement` and `ryacasLatex` fields.

### 3. Activity logging

#### Schema — single SQLite table `requests`

```sql
CREATE TABLE IF NOT EXISTS requests (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  timestamp       TEXT    NOT NULL,        -- ISO 8601 UTC
  source          TEXT    NOT NULL,        -- "local" | "gemini"
  recognized_latex TEXT   NOT NULL,
  confidence      REAL,                    -- nullable; only when source="local"
  num_components  INTEGER,
  operation       TEXT,                    -- "solve"|"integrate"|...
  sympy_solution  TEXT,                    -- LaTeX form
  ryacas_solution TEXT,                    -- LaTeX form, null if unavailable
  agreement       TEXT,                    -- "match"|"differ"|"ryacas_unavailable"|"ryacas_error"
  duration_ms     INTEGER,
  thumbnail_b64   TEXT                     -- 64x64 base64 PNG
);

CREATE INDEX idx_timestamp ON requests(timestamp);
CREATE INDEX idx_source ON requests(source);
```

Stored at `backend/activity.db` (gitignored).

#### Python module — `backend/activity_log.py`

```python
def log_request(
    source: str,
    recognized_latex: str,
    confidence: float | None,
    num_components: int | None,
    operation: str | None,
    sympy_solution: str | None,
    ryacas_solution: str | None,
    agreement: str | None,
    duration_ms: int,
    image_bytes: bytes,
) -> None:
    """Append one row. Never raises — logging failures must never break /convert."""
```

- Connection-per-call with `check_same_thread=False`. Low traffic, no pool needed.
- Computes 64×64 base64 PNG thumbnail with Pillow.
- Wraps everything in `try/except` and logs to stderr on failure.
- After insert, trims to last 5,000 rows (`DELETE FROM requests WHERE id NOT IN (SELECT id FROM requests ORDER BY id DESC LIMIT 5000)`).
- Bounds DB size to roughly 50 MB.

Called from `/convert` after the response is built but before returning.

#### Self-recovery

If `activity.db` is corrupt or unreadable, the module logs the error, deletes the file, recreates an empty schema, and continues. The user request still succeeds.

### 4. R Shiny dashboard (`backend/shiny_app.R` — full rewrite)

Replaces the current static HASYv2 viewer entirely. The HASYv2 explorer is removed.

#### Top-of-page summary cards

| Card | Shows | Source |
|---|---|---|
| Local hit rate | `% local / total` over last N requests | `activity.db` |
| Local model accuracy | `accuracy_on_accepted` | `metrics.json` |
| Solver agreement | `% agreement = "match"` over last N requests | `activity.db` |
| Total requests | `count(*)` | `activity.db` |

#### Tab 1 — Recent Activity

Reverse-chronological table of last 50 requests. Columns:
| Time | Thumbnail (64×64) | Recognized LaTeX | Source badge | Confidence | SymPy answer | Ryacas answer | Agreement |

Auto-refreshes every 5 seconds via `reactiveTimer(5000)`. LaTeX rendered with the `katex` Shiny extension if available, fallback to plain text.

#### Tab 2 — Model Performance

- **Confusion matrix heatmap** (ggplot2 `geom_tile`). Read from `backend/ml/artifacts/confusion.json`, computed once during training. Sortable by class frequency.
- **Per-class accuracy bars** — top 30 classes ranked by their `acc_on_accepted` slice. Same data file.
- **Confidence histogram** — distribution of `confidence` values from the live log, only for `source = "local"` rows.

#### Tab 3 — Solver Agreement

- **Stacked bar over time** — daily counts of `match` / `differ` / `ryacas_unavailable` / `ryacas_error`.
- **Disagreement table** — every row where `agreement = "differ"`. Shows recognized LaTeX, SymPy answer, Ryacas answer side-by-side.

#### Tab 4 — Local vs Gemini

- **Daily volume line chart** — local vs gemini count per day, last 30 days.
- **Top-10 symbols handled locally** — bar chart of most-frequently-accepted predicted_latex values.
- **Top-10 symbols falling through to Gemini** — bar chart of most-common LaTeX strings returned by Gemini. Helps decide what to expand training to next.

#### Empty-state handling

| State | Behavior |
|---|---|
| `activity.db` missing or empty | Every card and chart shows: *"No requests logged yet. Use the Math Solver tab to generate data."* |
| `metrics.json` missing | Top cards omit "Local model accuracy" |
| `confusion.json` missing | Tab 2 shows: *"Local classifier not trained yet. Run `python -m ml.train` from backend/."* |

Dashboard never errors — every tab handles missing data gracefully.

#### R dependencies

Already installed in some form: `shiny`, `shinydashboard`, `ggplot2`, `dplyr`.
New: `DBI`, `RSQLite` (read activity.db), `jsonlite` (read metrics.json/confusion.json), `plumber`, `Ryacas` (for the Plumber API, separate process).
Optional: `katex` (for LaTeX rendering in tables — fallback to text if unavailable).

### 5. Confusion matrix export from training

Add to `backend/ml/train.py`:

After the best epoch is identified, run one more eval pass to produce a confusion matrix. Save as `backend/ml/artifacts/confusion.json`:

```json
{
  "classes": ["\\int", "\\sum", ..., "0", "1", ..., "9"],
  "matrix": [[123, 0, 4, ...], [0, 145, 2, ...], ...]
}
```

Only the matrix values are saved; rendering is the dashboard's job.

A small `backend/ml/confusion.py` module isolates this from `train.py`:

```python
def compute_confusion(model, val_loader, num_classes, device) -> list[list[int]]: ...
def save_confusion(path, classes, matrix) -> None: ...
```

## Project layout

```
backend/
  plumber.R                    — NEW
  run_plumber.r                — NEW
  shiny_app.R                  — REWRITTEN (was HASYv2 viewer)
  ryacas_client.py             — NEW
  activity_log.py              — NEW
  activity.db                  — NEW (gitignored)
  ml/
    idx_parser.py              — NEW
    datasets.py                — NEW
    confusion.py               — NEW
    train.py                   — MODIFIED (MNIST + confusion matrix)
    artifacts/
      model.pt                 — regenerated
      classes.json             — regenerated (110 classes now)
      metrics.json             — regenerated
      confusion.json           — NEW (gitignored)
  main.py                      — MODIFIED (parallel solve + logging)
  tests/
    test_ryacas_client.py      — NEW
    test_activity_log.py       — NEW
    test_datasets.py           — NEW
    test_plumber.R             — NEW (LaTeX→Yacas conversion)
  data/mnist/                  — NEW (gitignored, auto-downloaded)

frontend/mathboard/src/
  App.jsx                      — MODIFIED (agreement line + tab label)
  App.css                      — MODIFIED (agreement styling)

docs/superpowers/specs/
  2026-04-25-mnist-and-r-reintegration-design.md — this file
```

## Dependencies added

**Python:**
- No new packages. `requests` may need to be pinned if not already pulled in transitively. SQLite via stdlib `sqlite3`. Pillow + numpy already available.

**R:**
- `plumber`, `Ryacas`, `jsonlite`, `DBI`, `RSQLite`. Reuses existing `shiny`, `shinydashboard`, `ggplot2`, `dplyr`.

## Testing strategy

### `tests/test_ryacas_client.py`
- `test_returns_none_when_plumber_unreachable` — `requests.get` raises ConnectionError → result is `None`.
- `test_returns_none_on_timeout` — patched timeout → `None`.
- `test_returns_failed_on_5xx` — mock 500 response → `RyacasResult(status="failed")`.
- `test_returns_success_on_valid_response` — mock JSON success → fully populated result.
- `test_health_cache_avoids_repeated_probes` — two consecutive calls within 30 seconds make only one health request.

### `tests/test_activity_log.py`
- `test_creates_schema_on_first_use`
- `test_log_then_read_roundtrip`
- `test_thumbnail_is_64x64_base64_png`
- `test_cap_enforced_at_5000_rows`
- `test_recovers_from_corrupt_db` — write garbage to file, log_request still succeeds, schema rebuilt.
- `test_never_raises_on_failure` — patch sqlite3 to raise; log_request returns silently.

### `tests/test_datasets.py`
- `test_idx_parser_reads_mnist_images_shape` — synthetic IDX file; assert (N, 28, 28) shape.
- `test_idx_parser_reads_mnist_labels_shape` — synthetic IDX file; assert (N,) labels.
- `test_mnist_dataset_polarity_inverted_to_match_hasyv2` — feed a known white-ink image; assert output has dark ink.
- `test_mnist_dataset_resized_to_32x32` — assert output tensor shape `(1, 32, 32)`.
- `test_combined_dataset_has_disjoint_label_indices_per_source` — HASYv2 labels ∈ [0..99], MNIST labels ∈ [100..109].
- `test_weighted_sampler_balances_classes` — sample 10000, assert per-class count within 20% of mean.

### `tests/test_main.py` (extended)
- `test_convert_includes_ryacas_when_available` — mock `cross_solve` returning success → response has `solution_data.ryacas` populated.
- `test_convert_marks_unavailable_when_plumber_down` — `cross_solve` returns None → `agreement == "ryacas_unavailable"`.
- `test_convert_marks_differ_when_solvers_disagree` — mock SymPy returns "5", Ryacas returns "6" → `agreement == "differ"`.
- `test_convert_logs_activity_row` — patch `activity_log.log_request`; assert called with right args.
- All existing tests still pass.

### `tests/test_plumber.R`
A few representative LaTeX → Yacas conversion + solve cases:
- `x + 5 = 12` → "7"
- `\int x dx` → contains "x^2"
- `\frac{d}{dx} x^2` → "2x"
- `x^2 + 2x + 1` → "1+2x+x^2" (or equivalent)
- `\lim_{x \to 0} x` → "0"

Run via `Rscript -e "testthat::test_file('tests/test_plumber.R')"`.

## Failure modes (explicit)

| Failure | Behavior |
|---|---|
| MNIST download fails | `train.py` reports the error and exits; the existing model.pt (if any) is untouched. |
| `confusion.json` missing | Dashboard Tab 2 shows "not trained" hint. |
| Plumber not running | `agreement == "ryacas_unavailable"`; UI hides agreement line. |
| Plumber returns invalid JSON | Caught, treated as `ryacas_error`. |
| Shiny not running | Iframe shows browser connection error; nothing else affected. |
| `activity.db` corrupt | Auto-deleted, schema recreated, current request logged anew. |
| SymPy and Ryacas disagree | UI shows ⚠ with both answers; nothing fails. |
| Both SymPy and Ryacas fail | Existing SymPy error path (already handled). |

## Tunable parameters

- Confidence threshold (existing, 0.85)
- Component count cap (existing, 4)
- Activity log cap (5,000 rows)
- Plumber health cache TTL (30 seconds)
- Plumber request timeout (3 seconds)
- Dashboard refresh interval (5 seconds)
- HASYv2 top-N classes (100, unchanged)

## Out of scope / future work

- Letter recognition (a-z, A-Z) via EMNIST
- Browser-side ONNX inference
- Model fine-tuning when the user corrects results (active learning)
- Persisting the dashboard's plot snapshots
- WebSocket-pushed live updates
- A "rollback" workflow if a new training run regresses (manual file restore for now)

## Effort estimate

- Implementation: ~400 lines Python, ~250 lines R, ~30 lines frontend, ~50 lines docs.
- Realistic build time: a few hours of coding + ~30 min training run.

## Decisions made during brainstorming

- **Hybrid HASYv2 + MNIST, not letters.** Lowercase/uppercase Latin letters are out — they're useful but high-variance, and digits are the larger gap.
- **R as parallel solver AND dashboard.** Most "significant" combined role. Drops the static dataset viewer entirely.
- **Cross-solver runs in parallel, not in series.** Hides the latency. 3-second timeout caps worst case.
- **SQLite log capped at 5,000 rows.** Keeps DB <50 MB, retains plenty of recent history.
- **Dashboard refresh polling (not WebSocket).** Simpler, fits Shiny's idioms.
- **No torchvision dep for MNIST.** Custom 30-line IDX parser instead — keeps the dependency surface tight.
- **HASYv2 explorer removed entirely.** User explicitly opted to drop it rather than relegate to a tab.

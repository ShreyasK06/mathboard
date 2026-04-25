# Local Symbol Classifier — Design

**Date:** 2026-04-25
**Status:** Approved (brainstorming complete)
**Owner:** Shreyas Kakkar

## Context

MathBoard currently sends every drawn image to the Google Gemini API for handwriting OCR. Every recognition costs an API call, depends on network connectivity, and exposes the user's drawings to a third-party service. The user wants to reduce reliance on Gemini.

HASYv2 (168,233 single-symbol images, 369 classes, 32×32 grayscale) was downloaded into `backend/data/hasy/` in this session, providing the training data needed to build a local symbol classifier.

## Goals

- Add a local symbol classifier that handles single-symbol drawings without calling Gemini.
- Keep Gemini available as a fallback for anything the local model can't confidently classify (multi-symbol expressions, rare glyphs, low-confidence predictions).
- Preserve the existing app behavior when the local model has not been trained yet — the app must still work end-to-end via Gemini until training is run.
- Make training a one-command operation that finishes on CPU in under an hour.

## Non-goals

- Multi-symbol segmentation. The local model classifies one symbol per image. Anything multi-symbol routes to Gemini.
- 2D layout (fractions, exponents, integrals). Routed to Gemini.
- Browser-side inference. The model runs server-side only. (A future iteration could export to ONNX and run in the browser.)
- Replacing SymPy. The downstream solver pipeline is unchanged.
- A user-facing toggle to bypass the local model.

## Architecture

### Request lifecycle

```
image bytes
    │
    ▼
preprocess (decode → grayscale → threshold → bounding box → component count)
    │
    ▼
gating decision
    │
    ├── num_components > 4  ─────────────┐
    ├── classifier disabled  ────────────┤
    ├── confidence < 0.85   ─────────────┤
    │                                    ▼
    │                                Gemini OCR
    │                                    │
    └── confidence ≥ 0.85 ────► local LaTeX
                                         │
                                         ▼
                                     SymPy solver
                                         │
                                         ▼
                                JSON response (with source field)
```

### Response shape

The `/convert` response gains two fields:

```json
{
  "status": "success",
  "latex": "x",
  "solution_data": { ... },
  "source": "local",       // new — "local" or "gemini"
  "confidence": 0.93        // new — number when source == "local", null otherwise
}
```

## Components

### 1. Training pipeline — `backend/ml/train.py`

Runnable from the `backend/` directory as `python -m ml.train`.

Steps:

1. Load `data/hasy/hasy-data-labels.csv`. Compute frequency per `latex` value.
2. Pick the **top-100 classes** by frequency. Build `classes.json` mapping index → LaTeX (ordered by frequency, ties broken alphabetically). Drop samples outside the top 100.
3. Stratified 85/15 train/val split, stratified by class.
4. Custom `torch.utils.data.Dataset` loads PNGs from `data/hasy/hasy-data/`, converts to single-channel float tensor, normalizes to mean 0 / std 1 using train-set statistics. Light augmentation on train only: random rotation ±10°, random translation ±2 px.
5. Train **`SymbolCNN`** (defined in `model.py`):
   ```
   Conv(1→32,3) → ReLU → Conv(32→32,3) → ReLU → MaxPool(2)
   Conv(32→64,3) → ReLU → Conv(64→64,3) → ReLU → MaxPool(2)
   Flatten → Dropout(0.4) → Linear(→128) → ReLU → Dropout(0.4) → Linear(→100)
   ```
   ~250k params, ~1 MB on disk.
6. Train: AdamW (lr=1e-3, wd=1e-4), cross-entropy, batch_size=128, 10 epochs, cosine LR schedule, early stop on val accuracy. Print per-epoch train/val loss and accuracy.
7. Threshold sweep on the val set: top-1 accuracy, coverage at confidence ≥ 0.85, accuracy on the accepted slice.
8. Save artifacts to `backend/ml/artifacts/`:
   - `model.pt` — `state_dict()` of the best epoch
   - `classes.json` — index → LaTeX mapping
   - `metrics.json` — `{val_top1_acc, coverage_at_0.85, accuracy_on_accepted, train_mean, train_std}`

Reproducibility: `torch.manual_seed(42)`, seeded dataloader shuffle. Re-runs match within ±0.5%.

### 2. Inference module — `backend/ml/classifier.py`

Class `SymbolClassifier`:

```python
class SymbolClassifier:
    def __init__(self, model_path: Path, classes_path: Path, metrics_path: Path): ...
    def is_loaded(self) -> bool: ...
    def classify(self, image_bytes: bytes) -> ClassifyResult | None: ...
```

`ClassifyResult` dataclass: `predicted_latex: str`, `confidence: float`, `num_components: int`, `accepted: bool`.

**Loading.** On FastAPI startup, attempt to load `model.pt`, `classes.json`, `metrics.json`. If any file is missing or `model.pt` is corrupt, the classifier is created in **disabled** state — `is_loaded()` returns False, `classify()` returns None. Backend logs a one-time warning: *"Local classifier disabled. Run `python -m ml.train` to enable."*

**Preprocessing pipeline** (must match training preprocessing):

1. Decode PNG → grayscale numpy array.
2. Otsu threshold → binary ink mask.
3. Count connected components.
   - If `num_components == 0`: return `None`.
   - If `num_components > 4`: return `ClassifyResult(predicted_latex="", confidence=0.0, num_components=N, accepted=False)`.
4. Crop to ink bounding box with 4 px padding.
5. Resize to 32×32 with `Pillow.LANCZOS`. Invert if needed (HASYv2 is black-on-white).
6. Normalize using `train_mean` / `train_std` from `metrics.json`.
7. Forward pass → softmax → argmax. Confidence = top-1 probability.
8. Return `ClassifyResult(latex_label, confidence, num_components, accepted=confidence ≥ 0.85)`.

**Latency budget.** CPU inference of a 32×32 image is single-digit ms; preprocessing is ~5–15 ms. Total local-path overhead ≈ 20 ms. When accepted, this saves 500–1500 ms of Gemini round-trip.

### 3. Gating in `backend/main.py`

Inside `/convert`, immediately before the existing Gemini call:

```python
result = classifier.classify(image_bytes) if classifier.is_loaded() else None

if result and result.accepted:
    latex_string = result.predicted_latex
    source = "local"
    confidence = result.confidence
else:
    # Existing Gemini block runs unchanged.
    source = "gemini"
    confidence = None
```

Solver step is unchanged. Response JSON gains `source` and `confidence` fields.

If the classifier raises during inference, the exception is caught, logged, and the request falls through to the Gemini path so a model bug never breaks the app.

### 4. Frontend — `frontend/mathboard/src/App.jsx` + `App.css`

A second small badge appears next to the existing operation badge in the result panel:

- **`⚡ Recognized locally`** — accent pill, shown when `source === "local"`. Hover tooltip displays confidence (`"Confidence: 93%"`).
- **`☁ Gemini`** — neutral grey pill, shown when `source === "gemini"`. No tooltip.

`convertResult` state grows by `source` and `confidence` fields. Roughly 15 lines added in `App.jsx`, ~30 lines of CSS for the two pill variants in light + dark themes. No new dependencies.

## Project layout

```
backend/
  ml/
    __init__.py
    model.py           # SymbolCNN nn.Module
    train.py           # python -m ml.train
    classifier.py      # SymbolClassifier
    README.md          # train / retrain / artifacts
    artifacts/         # gitignored
      model.pt
      classes.json
      metrics.json
  tests/
    test_classifier.py
    test_train_smoke.py
```

Modified:

- `backend/main.py` — startup load, gating block, response fields
- `backend/requirements.txt` — `torch` (CPU build); install command documented in `backend/ml/README.md`
- `backend/.gitignore` — add `ml/artifacts/`
- `frontend/mathboard/src/App.jsx` and `App.css` — source badge
- `SETUP.md` — short subsection on the optional local classifier

## Dependencies added

- `torch` (CPU build). ~200 MB in the venv. The only real new dep — `numpy` and `Pillow` are already transitive.

## Testing strategy

`tests/test_classifier.py`:

- `test_disabled_when_artifacts_missing` — instantiate with non-existent paths, assert `is_loaded() is False`, `classify(b"...") is None`.
- `test_skips_when_too_many_components` — synthetic image with 5 disjoint dots; assert `accepted=False`, `predicted_latex=""`.
- `test_accepts_high_confidence` — mock model to return ~0.95 on a 1-component image; assert `accepted=True`.
- `test_rejects_low_confidence` — mock model to return 0.5; assert `accepted=False`.
- `test_preprocessing_normalizes_to_32x32` — feed 200×80 image with ink in a corner; assert preprocessed tensor shape is `(1, 1, 32, 32)` and ink is centered.

`tests/test_main.py` (extended):

- `test_convert_uses_local_when_accepted` — mock classifier to accept; assert no Gemini call; response has `source="local"`.
- `test_convert_falls_back_to_gemini_on_low_confidence` — mock classifier to reject; assert Gemini is called; response has `source="gemini"`.
- `test_convert_falls_back_when_classifier_disabled` — `is_loaded()` False → Gemini path runs.

`tests/test_train_smoke.py`:

- Build a tiny synthetic dataset (200 random 32×32 images, 5 classes), run `train.py` for 1 epoch with patched paths, assert `model.pt`, `classes.json`, `metrics.json` are produced and `metrics.json` contains accuracy and coverage keys. Smoke test only — runs in under 10 seconds. Does not assert quality.

## Failure modes (explicit)

| Failure | Behavior |
|---|---|
| `model.pt` / `classes.json` missing | Classifier disabled; Gemini path runs; one-time warning logged |
| `model.pt` corrupt | Caught at load; classifier disabled |
| Image with zero ink pixels | `classify()` returns `None`; existing "empty image" error path |
| `num_components > 4` | Classifier returns `accepted=False`; falls through to Gemini |
| Classifier exception during inference | Caught, logged, falls through to Gemini |
| Gemini API error | Existing classifier error in `_classify_gemini_error` (unchanged) |

## Tunable parameters

- Confidence threshold (initial: 0.85)
- Component count cap (initial: 4)
- Top-N classes (initial: 100)

Initial values picked from `metrics.json` after training. Adjust after observing real-world inputs.

## Out of scope / future work

- Multi-symbol segmentation
- 2D layout (fractions, exponents)
- Browser-side ONNX inference
- Active learning (let users correct results to improve the model)
- Pretrained model fine-tuning (option C from brainstorm)

## Effort estimate

- Implementation: ~350 lines Python (training + classifier + tests), ~50 lines frontend, ~30 lines docs.
- Realistic build time: a few hours of coding + ~30 min training run on CPU.

## Decisions made during brainstorming

- **Hybrid, not full replacement.** Local model first; Gemini fallback for anything the local model can't handle.
- **Single-symbol scope.** No segmentation.
- **Server-side only.** Browser inference is a separate future project.
- **Top-100 classes.** Drop the long tail to keep the model focused on symbols users actually draw.
- **Custom small CNN, not pretrained.** Better fit for 32×32 grayscale than ImageNet transfer.
- **Artifacts gitignored.** Training is fast enough to require a one-time run; keeps the repo lean.
- **No "Force Gemini" toggle in the UI.** Out of scope — handled via logs and tests.

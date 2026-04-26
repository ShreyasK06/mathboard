# Local Symbol Classifier

A small CNN that runs server-side and handles single-symbol drawings before
falling back to Gemini. Trained on the top-100 most-common symbol classes
in HASYv2.

## Train

From `backend/`:

```
.venv/Scripts/python -m ml.train
```

~30 minutes on CPU. Produces three files in `ml/artifacts/`:

| File | Purpose |
|---|---|
| `model.pt` | Trained `state_dict` for `SymbolCNN` (~2 MB) |
| `classes.json` | Index → LaTeX label mapping (100 entries) |
| `metrics.json` | Accuracy, coverage at confidence threshold, train-set mean/std |

`ml/artifacts/` is gitignored. Re-run training to refresh.

## How it gates

For each `/convert` request:

1. Decode image, count connected components.
2. If `> 4 components` → skip local model, go to Gemini.
3. Else: crop, resize 32×32, run model, take softmax max as confidence.
4. If `confidence ≥ 0.85` → accept locally, return without calling Gemini.
5. Else → fall through to Gemini.

If `model.pt` is missing or corrupt, the classifier reports
`is_loaded() == False` and every request goes to Gemini — the app keeps
working before training is done.

## Tuning

After training, look at `metrics.json`:

```json
{
  "val_top1_acc": 0.85,
  "coverage_at_threshold": 0.71,
  "accuracy_on_accepted": 0.95,
  ...
}
```

- `val_top1_acc` — how often the model's top-1 prediction is right on the held-out 15%.
- `coverage_at_threshold` — fraction of val samples accepted at the 0.85 confidence threshold.
- `accuracy_on_accepted` — accuracy on just the accepted slice. **This is the user-facing accuracy**; aim for ≥ 0.90.

If `accuracy_on_accepted < 0.90`, raise `CONFIDENCE_THRESHOLD` in
`backend/ml/classifier.py` (and in `train.py` so the metrics
match) — fewer accepts, but the ones we do make are right more often.

## Files

| File | Purpose |
|---|---|
| `model.py` | `SymbolCNN` nn.Module |
| `preprocessing.py` | Otsu, components, crop, resize, normalize — used by both training and inference |
| `train.py` | CLI entry point, training loop, threshold sweep |
| `classifier.py` | `SymbolClassifier` — load artifacts, classify image bytes |
| `artifacts/` | Gitignored — produced by training |

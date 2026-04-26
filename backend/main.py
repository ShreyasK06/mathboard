"""
MathBoard FastAPI backend.

Accepts a handwritten-math image. Tries a local SymbolClassifier first; if it
declines (artifacts missing, multi-symbol, or low confidence), falls back to
Google Gemini for OCR. The result (LaTeX) goes through solver.py (SymPy).
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types

import solver
from ml.classifier import SymbolClassifier

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

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

_ARTIFACTS = Path(__file__).parent / "ml" / "artifacts"
classifier = SymbolClassifier(
    model_path=_ARTIFACTS / "model.pt",
    classes_path=_ARTIFACTS / "classes.json",
    metrics_path=_ARTIFACTS / "metrics.json",
)
print(
    f"[Classifier] {'enabled' if classifier.is_loaded() else 'disabled (run python -m ml.train to enable)'}",
    flush=True,
)

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
    if "service_disabled" in lowered or "has not been used in project" in lowered:
        import re as _re
        m = _re.search(r"https://console\.developers\.google\.com/[^\s'\"]+", msg)
        url = m.group(0) if m else "https://aistudio.google.com"
        return (
            "Your Gemini key is a Google Cloud key whose project does not have the "
            "Generative Language API enabled. Either (a) enable it here: "
            f"{url} (then wait ~2 min), or (b) replace the key in backend/.env "
            "with an AI Studio key from https://aistudio.google.com (works immediately)."
        )
    if "403" in msg or "permission_denied" in lowered:
        return "Gemini API key invalid or unauthorized. Generate a new AI Studio key at https://aistudio.google.com."
    if "401" in msg or "unauthenticated" in lowered or "api key not valid" in lowered:
        return "Gemini API key is not authorized. Generate a fresh key at https://aistudio.google.com."
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
        "classifier_loaded": classifier.is_loaded(),
    }


@app.post("/convert")
async def convert_equation(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
    except Exception as exc:
        return {"error": f"Failed to read uploaded file: {exc}"}

    if not image_bytes:
        return {"error": "Uploaded image was empty."}

    # Try local classifier first.
    try:
        local_result = classifier.classify(image_bytes) if classifier.is_loaded() else None
    except Exception as exc:
        print(f"[Classifier Error] {exc}", flush=True)
        local_result = None

    if local_result and local_result.accepted:
        latex_string = local_result.predicted_latex
        source = "local"
        confidence: float | None = local_result.confidence
        print(f"[Classifier] Accepted '{latex_string}' (conf={confidence:.3f})", flush=True)
    else:
        if not GEMINI_API_KEY or _client is None:
            return {
                "error": (
                    "Gemini API key is not configured on the server. "
                    "Add GEMINI_API_KEY=... to backend/.env and restart."
                )
            }
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
        source = "gemini"
        confidence = None
        print(f"[Gemini Result] {latex_string}", flush=True)

    solution_data = solver.solve_expression(latex_string)
    print(f"[Solver Result] {solution_data}", flush=True)

    return {
        "status": "success",
        "latex": latex_string,
        "solution_data": solution_data,
        "source": source,
        "confidence": confidence,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

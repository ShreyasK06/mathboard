
import concurrent.futures
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types

import activity_log
import ryacas_client
import solver
from ml.classifier import SymbolClassifier

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

if not GEMINI_API_KEY:
    print(
        "\n================================================================\n"
        "  WARNING: GEMINI_API_KEY is not set.\n"
        "  Create backend/.env with GEMINI_API_KEY=your_key_here\n"
        "  Generate a key at https://aistudio.google.com\n"
        "================================================================\n",
        file=sys.stderr, flush=True,
    )
    _client = None
else:
    _client = genai.Client(api_key=GEMINI_API_KEY)

_BACKEND_DIR = Path(__file__).parent
_ARTIFACTS = _BACKEND_DIR / "ml" / "artifacts"
_ACTIVITY_DB = _BACKEND_DIR / "activity.db"

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

SOLVE_PROMPT = (
    "You are a math solver used as an independent cross-check. "
    "Solve the following expression and return ONLY the final answer in LaTeX. "
    "Rules:\n"
    "- For an equation, return just the value(s) of the unknown — no 'x =' prefix.\n"
    "- For multiple solutions, separate with commas (e.g., '-2, 2').\n"
    "- For simplification or differentiation, return the simplified form only.\n"
    "- For indefinite integration, omit the constant of integration.\n"
    "- No explanation, no $, no \\( \\), no markdown, no backticks.\n\n"
    "Expression: {latex}"
)

GEMINI_SOLVE_TIMEOUT_S = 6.0

app = FastAPI(title="MathBoard Backend")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
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
        m = re.search(r"https://console\.developers\.google\.com/[^\s'\"]+", msg)
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


def _normalize_for_compare(s: str | None) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", "", s)


def _as_solution_set(s: str | None) -> frozenset[str]:
    """Split a comma-separated solution list into a whitespace-normalized set
    so '-2, 2' compares equal to '2,-2'."""
    if not s:
        return frozenset()
    return frozenset(_normalize_for_compare(p) for p in s.split(",") if p.strip())


def _compute_agreement(primary_latex: str | None, crosscheck_result) -> str:
    """Compare the displayed (primary) answer against the cross-checker.
    Returns 'match' | 'differ' | 'crosscheck_unavailable' | 'crosscheck_error'.
    """
    if crosscheck_result is None:
        return "crosscheck_unavailable"
    if crosscheck_result.status != "success":
        return "crosscheck_error"
    if _as_solution_set(primary_latex) == _as_solution_set(crosscheck_result.latex_result):
        return "match"
    return "differ"


@dataclass
class GeminiSolveResult:
    status: str                 # "success" | "failed"
    solution: Optional[str]
    latex_result: Optional[str]
    error: Optional[str]


def solve_with_gemini(latex_string: str) -> Optional[GeminiSolveResult]:
    """Ask Gemini to solve the expression independently. Returns None when the
    API key isn't configured; never raises."""
    if _client is None or not GEMINI_API_KEY:
        return None
    try:
        response = _client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[SOLVE_PROMPT.format(latex=latex_string)],
        )
        raw = (response.text or "").strip()
    except Exception as exc:
        return GeminiSolveResult(
            status="failed", solution=None, latex_result=None, error=str(exc)
        )
    cleaned = _strip_latex_fences(raw)
    # Strip a leading "x =" / "x=" prefix Gemini sometimes adds despite the prompt.
    cleaned = re.sub(r"^[a-zA-Z]\s*=\s*", "", cleaned).strip()
    if not cleaned:
        return GeminiSolveResult(
            status="failed", solution=None, latex_result=None, error="empty response"
        )
    return GeminiSolveResult(
        status="success", solution=cleaned, latex_result=cleaned, error=None
    )


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
    t0 = time.monotonic()
    try:
        image_bytes = await file.read()
    except Exception as exc:
        return {"error": f"Failed to read uploaded file: {exc}"}
    if not image_bytes:
        return {"error": "Uploaded image was empty."}

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
            return {"error": "Gemini returned an empty response. Try drawing a clearer expression."}
        source = "gemini"
        confidence = None
        print(f"[Gemini Result] {latex_string}", flush=True)

    # Solve in parallel: Ryacas (primary), Gemini (cross-check), SymPy (fallback
    # for when Ryacas can't be reached). SymPy also supplies the operation_label
    # and step list we display either way.
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        ryacas_future = ex.submit(ryacas_client.cross_solve, latex_string)
        gemini_future = ex.submit(solve_with_gemini, latex_string)
        sympy_future = ex.submit(solver.solve_expression, latex_string)
        ryacas_result = ryacas_future.result()
        sympy_result = sympy_future.result()
        try:
            gemini_result = gemini_future.result(timeout=GEMINI_SOLVE_TIMEOUT_S)
        except concurrent.futures.TimeoutError:
            gemini_result = GeminiSolveResult(
                status="failed", solution=None, latex_result=None,
                error="Gemini cross-check timed out",
            )

    # Pick the displayed answer: Ryacas if it succeeded, otherwise SymPy fallback.
    if ryacas_result is not None and ryacas_result.status == "success":
        primary_solver = "ryacas"
        solution_data = {
            "status": "success",
            "operation": sympy_result.get("operation"),
            "operation_label": sympy_result.get("operation_label"),
            "steps": sympy_result.get("steps", []),
            "solution": ryacas_result.solution,
            "latex_result": ryacas_result.latex_result,
        }
    else:
        primary_solver = "sympy"
        solution_data = sympy_result

    solution_data["primary_solver"] = primary_solver
    solution_data["ryacas"] = (
        None if ryacas_result is None
        else {
            "status": ryacas_result.status,
            "solution": ryacas_result.solution,
            "latex_result": ryacas_result.latex_result,
        }
    )
    solution_data["crosscheck"] = (
        None if gemini_result is None
        else {
            "solver": "gemini",
            "status": gemini_result.status,
            "solution": gemini_result.solution,
            "latex_result": gemini_result.latex_result,
        }
    )
    solution_data["agreement"] = _compute_agreement(
        solution_data.get("latex_result"), gemini_result
    )

    duration_ms = int((time.monotonic() - t0) * 1000)

    activity_log.log_request(
        db_path=_ACTIVITY_DB,
        source=source,
        recognized_latex=latex_string,
        confidence=confidence,
        num_components=local_result.num_components if local_result else None,
        operation=solution_data.get("operation"),
        primary_solver=primary_solver,
        primary_solution=solution_data.get("latex_result"),
        crosscheck_solution=(solution_data["crosscheck"] or {}).get("latex_result"),
        agreement=solution_data["agreement"],
        duration_ms=duration_ms,
        image_bytes=image_bytes,
    )

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

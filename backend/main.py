"""
MathBoard FastAPI backend.

Accepts a handwritten-math image, sends it to Google Gemini for OCR -> LaTeX,
passes the LaTeX to solver.py (SymPy), and returns the LaTeX + solution.
"""

import os
import sys

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types

import solver

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")

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
    if "403" in msg or "permission_denied" in lowered or "service_disabled" in lowered:
        return "Gemini API key invalid. Go to aistudio.google.com and create a new API key."
    if "401" in msg or "unauthenticated" in lowered or "api key not valid" in lowered:
        return "Gemini API key is not authorized. Generate a fresh key at aistudio.google.com."
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
    }


@app.post("/convert")
async def convert_equation(file: UploadFile = File(...)):
    if not GEMINI_API_KEY or _client is None:
        return {
            "error": (
                "Gemini API key is not configured on the server. "
                "Add GEMINI_API_KEY=... to backend/.env and restart."
            )
        }

    try:
        image_bytes = await file.read()
    except Exception as exc:
        return {"error": f"Failed to read uploaded file: {exc}"}

    if not image_bytes:
        return {"error": "Uploaded image was empty."}

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

    print(f"[Gemini Result] {latex_string}", flush=True)
    solution_data = solver.solve_expression(latex_string)
    print(f"[Solver Result] {solution_data}", flush=True)

    return {
        "status": "success",
        "latex": latex_string,
        "solution_data": solution_data,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

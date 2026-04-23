import os
import io
import requests
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from dotenv import load_dotenv
from google import genai

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
client = genai.Client(api_key=GEMINI_API_KEY)

R_API_URL = "http://localhost:8003/solve"

@app.post("/convert")
async def convert_equation(file: UploadFile = File(...)):
    if not GEMINI_API_KEY:
        return {"error": "Gemini API key is not configured on the server."}

    image_bytes = await file.read()
    try:
        img = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        return {"error": f"Invalid image format: {e}"}

    print("[Routing] Sending image to Gemini Cloud...")
    prompt = (
        "You are an expert mathematical OCR system. "
        "Extract the mathematical expression from this image and return ONLY the raw LaTeX string. "
        "Do not include any markdown formatting, backticks, or conversational text. "
        "Do not include \\( or \\) wrappers. Just the math."
    )

    try:
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=[prompt, img]
        )
        latex_string = response.text.strip()
        print(f"[Gemini Result] Raw LaTeX: {latex_string}")
    except Exception as e:
        return {"error": f"Gemini API Error: {str(e)}"}

    r_payload = {"latex": latex_string}
    print("[Routing] Sending LaTeX to local R Plumber API...")

    try:
        r_resp = requests.post(R_API_URL, json=r_payload, timeout=10)
        r_resp.raise_for_status()
        r_result = r_resp.json()
        print(f"[R Response] {r_result}")
    except requests.exceptions.ConnectionError:
        print("[Error] Cannot connect to R. Is Plumber running on port 8003?")
        r_result = {"error": "R Compute Engine offline."}
    except Exception as e:
        r_result = {"error": str(e)}

    return {
        "status": "success",
        "latex": latex_string,
        "solution_data": r_result
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

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
    mock_response = MagicMock()
    mock_response.text = "x^2 + 1"

    mock_r = MagicMock()
    mock_r.status_code = 200
    mock_r.json.return_value = {"solution": "x^2+1", "status": "success"}
    mock_r.raise_for_status = MagicMock()

    with patch("main.client.models.generate_content", return_value=mock_response), \
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
    mock_response = MagicMock()
    mock_response.text = "x + 1"

    with patch("main.client.models.generate_content", return_value=mock_response), \
         patch("main.requests.post", side_effect=Exception("Connection refused")):
        response = client.post(
            "/convert",
            files={"file": ("test.png", _make_png_bytes(), "image/png")},
        )

    assert response.status_code == 200
    data = response.json()
    assert "error" in data["solution_data"]

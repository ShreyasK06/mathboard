import io
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from PIL import Image

import main

_client = TestClient(main.app)

_MOCK_SOLVER_RESULT = {
    "status": "success",
    "operation": "simplify",
    "operation_label": "Simplifying",
    "steps": ["Simplifying expression", "Applied SymPy simplification rules"],
    "solution": "x**2 + 1",
    "latex_result": "x^{2} + 1",
}


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (100, 100), color="white").save(buf, format="PNG")
    return buf.getvalue()


def test_health_returns_ok():
    resp = _client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "gemini_key_set" in data
    assert "gemini_model" in data


def test_health_has_no_r_url():
    resp = _client.get("/health")
    assert "r_url" not in resp.json()


def test_convert_returns_error_when_image_empty():
    resp = _client.post("/convert", files={"file": ("b.png", b"", "image/png")})
    assert resp.status_code == 200
    assert "error" in resp.json()


def test_convert_happy_path():
    mock_response = MagicMock()
    mock_response.text = "x^2 + 1"

    with patch.object(main._client.models, "generate_content", return_value=mock_response), \
         patch.object(main.solver, "solve_expression", return_value=_MOCK_SOLVER_RESULT):
        resp = _client.post("/convert", files={"file": ("b.png", _png_bytes(), "image/png")})

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["latex"] == "x^2 + 1"
    sd = data["solution_data"]
    assert sd["operation"] == "simplify"
    assert sd["operation_label"] == "Simplifying"
    assert isinstance(sd["steps"], list)
    assert sd["latex_result"] == "x^{2} + 1"


def test_convert_gemini_empty_text_returns_error():
    mock_response = MagicMock()
    mock_response.text = ""

    with patch.object(main._client.models, "generate_content", return_value=mock_response):
        resp = _client.post("/convert", files={"file": ("b.png", _png_bytes(), "image/png")})

    assert resp.status_code == 200
    assert "error" in resp.json()


def test_convert_gemini_exception_classified():
    with patch.object(
        main._client.models,
        "generate_content",
        side_effect=Exception("429 quota exceeded"),
    ):
        resp = _client.post("/convert", files={"file": ("b.png", _png_bytes(), "image/png")})

    assert resp.status_code == 200
    data = resp.json()
    assert "error" in data
    assert "quota" in data["error"].lower() or "rate" in data["error"].lower()


def test_convert_solution_data_has_operation_label():
    mock_response = MagicMock()
    mock_response.text = "x + 2 = 5"

    solve_result = {
        "status": "success",
        "operation": "solve",
        "operation_label": "Solving equation",
        "steps": ["Detected equation", "Solving for x"],
        "solution": "3",
        "latex_result": "3",
    }

    with patch.object(main._client.models, "generate_content", return_value=mock_response), \
         patch.object(main.solver, "solve_expression", return_value=solve_result):
        resp = _client.post("/convert", files={"file": ("b.png", _png_bytes(), "image/png")})

    data = resp.json()
    assert data["solution_data"]["operation_label"] == "Solving equation"
    assert data["solution_data"]["steps"] == ["Detected equation", "Solving for x"]


from ml.classifier import ClassifyResult


def test_convert_uses_local_when_classifier_accepts():
    """When the local classifier accepts, Gemini is NOT called and source='local'."""
    accepted = ClassifyResult(
        predicted_latex="x", confidence=0.93, num_components=1, accepted=True
    )

    class FakeClassifier:
        def is_loaded(self): return True
        def classify(self, _): return accepted

    with patch.object(main, "classifier", FakeClassifier()), \
         patch.object(main._client.models, "generate_content") as gemini, \
         patch.object(main.solver, "solve_expression", return_value=_MOCK_SOLVER_RESULT):
        resp = _client.post("/convert", files={"file": ("b.png", _png_bytes(), "image/png")})

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["latex"] == "x"
    assert data["source"] == "local"
    assert 0.9 < data["confidence"] < 1.0
    gemini.assert_not_called()


def test_convert_falls_back_to_gemini_on_low_confidence():
    rejected = ClassifyResult(
        predicted_latex="x", confidence=0.40, num_components=1, accepted=False
    )

    class FakeClassifier:
        def is_loaded(self): return True
        def classify(self, _): return rejected

    mock_response = MagicMock()
    mock_response.text = "x^2 + 1"

    with patch.object(main, "classifier", FakeClassifier()), \
         patch.object(main._client.models, "generate_content", return_value=mock_response), \
         patch.object(main.solver, "solve_expression", return_value=_MOCK_SOLVER_RESULT):
        resp = _client.post("/convert", files={"file": ("b.png", _png_bytes(), "image/png")})

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["latex"] == "x^2 + 1"
    assert data["source"] == "gemini"
    assert data["confidence"] is None


def test_convert_falls_back_when_classifier_disabled():
    class DisabledClassifier:
        def is_loaded(self): return False
        def classify(self, _): return None

    mock_response = MagicMock()
    mock_response.text = "y"

    with patch.object(main, "classifier", DisabledClassifier()), \
         patch.object(main._client.models, "generate_content", return_value=mock_response), \
         patch.object(main.solver, "solve_expression", return_value=_MOCK_SOLVER_RESULT):
        resp = _client.post("/convert", files={"file": ("b.png", _png_bytes(), "image/png")})

    assert resp.json()["source"] == "gemini"
    assert resp.json()["confidence"] is None


from ryacas_client import RyacasResult


def test_convert_includes_ryacas_when_available():
    """When Plumber is reachable and agrees with SymPy, response has agreement='match'."""
    mock_response = MagicMock()
    mock_response.text = "x + 5 = 12"

    sympy_result = {
        "status": "success", "operation": "solve", "operation_label": "Solving equation",
        "steps": [], "solution": "7", "latex_result": "7",
    }
    ryacas_result = RyacasResult(status="success", solution="7", latex_result="7", error=None)

    class FakeClassifier:
        def is_loaded(self): return False
        def classify(self, _): return None

    with patch.object(main, "classifier", FakeClassifier()), \
         patch.object(main._client.models, "generate_content", return_value=mock_response), \
         patch.object(main.solver, "solve_expression", return_value=sympy_result), \
         patch.object(main.ryacas_client, "cross_solve", return_value=ryacas_result):
        resp = _client.post("/convert", files={"file": ("b.png", _png_bytes(), "image/png")})

    data = resp.json()
    sd = data["solution_data"]
    assert sd["ryacas"]["latex_result"] == "7"
    assert sd["agreement"] == "match"


def test_convert_marks_unavailable_when_plumber_down():
    mock_response = MagicMock()
    mock_response.text = "x"
    sympy_result = {
        "status": "success", "operation": "simplify", "operation_label": "Simplifying",
        "steps": [], "solution": "x", "latex_result": "x",
    }

    class FakeClassifier:
        def is_loaded(self): return False
        def classify(self, _): return None

    with patch.object(main, "classifier", FakeClassifier()), \
         patch.object(main._client.models, "generate_content", return_value=mock_response), \
         patch.object(main.solver, "solve_expression", return_value=sympy_result), \
         patch.object(main.ryacas_client, "cross_solve", return_value=None):
        resp = _client.post("/convert", files={"file": ("b.png", _png_bytes(), "image/png")})

    sd = resp.json()["solution_data"]
    assert sd["ryacas"] is None
    assert sd["agreement"] == "ryacas_unavailable"


def test_convert_marks_differ_when_solvers_disagree():
    mock_response = MagicMock()
    mock_response.text = "x + 5 = 12"
    sympy_result = {
        "status": "success", "operation": "solve", "operation_label": "Solving equation",
        "steps": [], "solution": "7", "latex_result": "7",
    }
    ryacas_result = RyacasResult(status="success", solution="6", latex_result="6", error=None)

    class FakeClassifier:
        def is_loaded(self): return False
        def classify(self, _): return None

    with patch.object(main, "classifier", FakeClassifier()), \
         patch.object(main._client.models, "generate_content", return_value=mock_response), \
         patch.object(main.solver, "solve_expression", return_value=sympy_result), \
         patch.object(main.ryacas_client, "cross_solve", return_value=ryacas_result):
        resp = _client.post("/convert", files={"file": ("b.png", _png_bytes(), "image/png")})

    sd = resp.json()["solution_data"]
    assert sd["agreement"] == "differ"
    assert sd["ryacas"]["latex_result"] == "6"


def test_convert_logs_activity_row():
    mock_response = MagicMock()
    mock_response.text = "x"
    sympy_result = {
        "status": "success", "operation": "simplify", "operation_label": "Simplifying",
        "steps": [], "solution": "x", "latex_result": "x",
    }

    class FakeClassifier:
        def is_loaded(self): return False
        def classify(self, _): return None

    with patch.object(main, "classifier", FakeClassifier()), \
         patch.object(main._client.models, "generate_content", return_value=mock_response), \
         patch.object(main.solver, "solve_expression", return_value=sympy_result), \
         patch.object(main.ryacas_client, "cross_solve", return_value=None), \
         patch.object(main.activity_log, "log_request") as log_mock:
        _client.post("/convert", files={"file": ("b.png", _png_bytes(), "image/png")})

    assert log_mock.call_count == 1
    kwargs = log_mock.call_args.kwargs
    assert kwargs["source"] == "gemini"
    assert kwargs["recognized_latex"] == "x"
    assert kwargs["agreement"] == "ryacas_unavailable"

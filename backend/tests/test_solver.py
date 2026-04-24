from solver import detect_operation, solve_expression


# ── detect_operation ──────────────────────────────────────────────────────────

def test_detect_solve_linear():
    assert detect_operation("x + 2 = 5") == "solve"

def test_detect_solve_quadratic():
    assert detect_operation("x^2 - 4 = 0") == "solve"

def test_detect_integrate_indefinite():
    assert detect_operation(r"\int x^2 dx") == "integrate"

def test_detect_integrate_definite():
    assert detect_operation(r"\int_{0}^{1} x^2 dx") == "integrate"

def test_detect_differentiate_frac():
    assert detect_operation(r"\frac{d}{dx} x^3") == "differentiate"

def test_detect_differentiate_prime():
    assert detect_operation("f'(x)") == "differentiate"

def test_detect_limit():
    assert detect_operation(r"\lim_{x \to 0} x") == "limit"

def test_detect_simplify_default():
    assert detect_operation("x^2 + 2x + 1") == "simplify"

def test_detect_simplify_arithmetic():
    assert detect_operation("3 + 5") == "simplify"


# ── result shape ──────────────────────────────────────────────────────────────

def test_result_has_required_keys_on_success():
    result = solve_expression("3 + 5")
    assert result["status"] == "success"
    for key in ("operation", "operation_label", "steps", "solution", "latex_result"):
        assert key in result, f"Missing key in success result: {key}"

def test_steps_is_nonempty_list():
    result = solve_expression("x + 1")
    assert isinstance(result["steps"], list)
    assert len(result["steps"]) >= 1

def test_result_is_always_a_dict():
    for expr in ["x + 1", "x = 2", r"\int x dx", r"\frac{d}{dx} x^2", "garbage###"]:
        result = solve_expression(expr)
        assert isinstance(result, dict), f"Not a dict for: {expr}"
        assert "status" in result


# ── simplify ──────────────────────────────────────────────────────────────────

def test_simplify_arithmetic():
    result = solve_expression("3 + 5")
    assert result["status"] == "success"
    assert result["operation"] == "simplify"
    assert "8" in result["solution"]

def test_simplify_operation_label():
    result = solve_expression("x + 1")
    assert result["operation_label"] == "Simplifying"


# ── solve ─────────────────────────────────────────────────────────────────────

def test_solve_linear_equation():
    result = solve_expression("x + 2 = 5")
    assert result["status"] == "success"
    assert result["operation"] == "solve"
    assert "3" in result["solution"]

def test_solve_quadratic_equation():
    result = solve_expression("x^2 - 4 = 0")
    assert result["status"] == "success"
    assert result["operation"] == "solve"
    assert "2" in result["solution"]

def test_solve_operation_label():
    result = solve_expression("x = 1")
    assert result["operation_label"] == "Solving equation"


# ── differentiate ─────────────────────────────────────────────────────────────

def test_differentiate_power_rule():
    result = solve_expression(r"\frac{d}{dx} x^2")
    assert result["status"] == "success"
    assert result["operation"] == "differentiate"
    assert "2" in result["solution"]

def test_differentiate_operation_label():
    result = solve_expression(r"\frac{d}{dx} x")
    assert result["operation_label"] == "Differentiating"


# ── integrate ─────────────────────────────────────────────────────────────────

def test_integrate_basic():
    result = solve_expression(r"\int x dx")
    assert result["status"] == "success"
    assert result["operation"] == "integrate"
    assert "x" in result["solution"]

def test_integrate_operation_label():
    result = solve_expression(r"\int x dx")
    assert result["operation_label"] == "Integrating"


# ── limit ─────────────────────────────────────────────────────────────────────

def test_limit_basic():
    result = solve_expression(r"\lim_{x \to 0} x")
    assert result["status"] == "success"
    assert result["operation"] == "limit"
    assert "0" in result["solution"]


# ── error / timeout ───────────────────────────────────────────────────────────

def test_failed_result_has_error_key():
    result = solve_expression(r"\completely_invalid{{{garbage}}}")
    assert result["status"] in ("success", "failed")
    if result["status"] == "failed":
        assert "error" in result

def test_timeout_returns_error_dict(monkeypatch):
    import solver as s

    original_run = s._OP_FUNCS["simplify"]

    def hang(latex, steps):
        import time
        time.sleep(20)
        return original_run(latex, steps)

    monkeypatch.setitem(s._OP_FUNCS, "simplify", hang)
    result = s.solve_expression("x + 1")
    assert result["status"] == "failed"
    assert "complex" in result["error"].lower() or "time" in result["error"].lower()

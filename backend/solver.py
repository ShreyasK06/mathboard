import re
import threading
from typing import Any

import sympy

_x, _y, _z, _t, _n = sympy.symbols("x y z t n")
_LOCALS: dict[str, Any] = {s.name: s for s in (_x, _y, _z, _t, _n)}
_LOCALS.update({
    "pi": sympy.pi, "E": sympy.E, "oo": sympy.oo,
    "sin": sympy.sin, "cos": sympy.cos, "tan": sympy.tan,
    "log": sympy.log, "ln": sympy.log, "exp": sympy.exp,
    "sqrt": sympy.sqrt, "abs": sympy.Abs,
})

_OP_LABELS = {
    "solve": "Solving equation",
    "integrate": "Integrating",
    "differentiate": "Differentiating",
    "limit": "Finding limit",
    "simplify": "Simplifying",
}


def _clean_latex(latex: str) -> str:
    s = latex.strip()
    for start, end in (("\\(", "\\)"), ("$$", "$$"), ("$", "$")):
        if s.startswith(start) and s.endswith(end):
            s = s[len(start):-len(end)].strip()
    s = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"((\1)/(\2))", s)
    s = re.sub(r"\^\{([^{}]*)\}", r"**(\1)", s)
    s = re.sub(r"\^([^{(])", r"**\1", s)
    s = re.sub(r"\\sqrt\{([^{}]*)\}", r"sqrt(\1)", s)
    s = re.sub(r"\\left\s*[\(\[|]", "(", s)
    s = re.sub(r"\\right\s*[\)\]|]", ")", s)
    for fn in ("sin", "cos", "tan", "cot", "sec", "csc",
               "arcsin", "arccos", "arctan",
               "sinh", "cosh", "tanh", "ln", "log", "exp", "abs"):
        s = s.replace(f"\\{fn}", fn)
    for k, v in {
        "\\alpha": "alpha", "\\beta": "beta", "\\gamma": "gamma",
        "\\theta": "theta", "\\phi": "phi", "\\pi": "pi",
        "\\infty": "oo", "\\cdot": "*", "\\times": "*", "\\div": "/",
    }.items():
        s = s.replace(k, v)
    s = re.sub(r"\\[a-zA-Z]+\*?", "", s)
    s = s.replace("{", "(").replace("}", ")")
    s = re.sub(r"(\d)([a-zA-Z(])", r"\1*\2", s)
    return s.strip()


def _parse(latex: str) -> sympy.Expr:
    try:
        import antlr4  # noqa: F401 — only attempt parse_latex if antlr4 is present
        from sympy.parsing.latex import parse_latex
        return parse_latex(latex)
    except Exception:
        pass
    return sympy.sympify(_clean_latex(latex), locals=_LOCALS)


def detect_operation(latex: str) -> str:
    if "=" in latex:
        return "solve"
    if r"\int" in latex:
        return "integrate"
    if re.search(r"\\frac\s*\{d", latex) or "'" in latex or r"\prime" in latex:
        return "differentiate"
    if r"\lim" in latex:
        return "limit"
    return "simplify"


def _do_solve(latex: str, steps: list[str]) -> tuple[str, str]:
    steps.append("Detected equation (contains '=')")
    lhs_str, rhs_str = latex.split("=", 1)
    lhs = _parse(lhs_str.strip())
    rhs = _parse(rhs_str.strip())
    expr = lhs - rhs
    steps.append(f"Rewritten as {sympy.latex(expr)} = 0")
    free = sorted(expr.free_symbols, key=str)
    if not free:
        result = sympy.simplify(expr)
        steps.append("No free variables; simplified both sides")
        return str(result), sympy.latex(result)
    var = free[0]
    steps.append(f"Solving for {var}")
    solutions = sympy.solve(expr, var)
    steps.append(f"Found {len(solutions)} solution(s)")
    return (
        ", ".join(str(s) for s in solutions),
        ", ".join(sympy.latex(s) for s in solutions),
    )


def _do_integrate(latex: str, steps: list[str]) -> tuple[str, str]:
    steps.append("Detected integral")
    # Remove \int prefix (use str.replace to avoid regex backslash issues)
    body = latex.replace(r"\int", "", 1).strip()
    limits_m = re.match(r"_\{?([^}^]+)\}?\s*\^\{?([^}]+)\}?\s*(.*)", body)
    if limits_m:
        a_str, b_str, body = limits_m.groups()
        steps.append(f"Definite integral from {a_str.strip()} to {b_str.strip()}")
        var_m = re.search(r"\s*d([a-z])\s*$", body)
        var = sympy.Symbol(var_m.group(1)) if var_m else _x
        expr_str = re.sub(r"\s*d[a-z]\s*$", "", body).strip()
        expr = _parse(expr_str)
        a, b = _parse(a_str.strip()), _parse(b_str.strip())
        steps.append(f"Integrating {sympy.latex(expr)} w.r.t. {var} from {a} to {b}")
        result = sympy.simplify(sympy.integrate(expr, (var, a, b)))
    else:
        var_m = re.search(r"\s*d([a-z])\s*$", body)
        var = sympy.Symbol(var_m.group(1)) if var_m else _x
        expr_str = re.sub(r"\s*d[a-z]\s*$", "", body).strip()
        steps.append(f"Indefinite integral with respect to {var}")
        expr = _parse(expr_str)
        steps.append(f"Integrating {sympy.latex(expr)}")
        result = sympy.integrate(expr, var)
        steps.append("Constant of integration omitted")
    return str(result), sympy.latex(result)


def _do_differentiate(latex: str, steps: list[str]) -> tuple[str, str]:
    steps.append("Detected derivative")
    tagged = re.sub(
        r"\\frac\s*\{d(?:\^\{?\d+\}?)?\}\s*\{d([a-z])(?:\^\{?\d+\}?)?\}",
        r"__DIFF_\1__ ",
        latex,
    )
    diff_m = re.search(r"__DIFF_([a-z])__\s*(.*)", tagged)
    if diff_m:
        var = sympy.Symbol(diff_m.group(1))
        expr_str = diff_m.group(2).strip()
        steps.append(f"Differentiating with respect to {var}")
        expr = _parse(expr_str)
    else:
        expr = _parse(re.sub(r"'", "", latex).strip())
        var = _x
        steps.append("Differentiating with respect to x (prime notation)")
    result = sympy.diff(expr, var)
    steps.append("Applied differentiation rules")
    return str(result), sympy.latex(result)


def _do_limit(latex: str, steps: list[str]) -> tuple[str, str]:
    steps.append("Detected limit expression")
    # Strip \lim prefix with str operations (avoids \l bad-escape on Python 3.13+)
    lim_body = latex
    lim_idx = latex.find(r"\lim")
    if lim_idx >= 0:
        lim_body = latex[lim_idx + 4:].strip()
    # Parse limit variable and approach from _{x \to value} or _x\to value
    lim_m = re.search(r"_\{?([a-z])\s*\\to\s*([^}]+)\}?\s*(.*)", lim_body)
    if lim_m:
        var_name, approach_str, expr_str = lim_m.groups()
        var = sympy.Symbol(var_name)
        approach = _parse(approach_str.strip())
        expr = _parse(expr_str.strip())
        steps.append(f"Limit as {var} → {approach}")
        result = sympy.limit(expr, var, approach)
    else:
        steps.append("Could not parse limit structure; simplifying instead")
        result = sympy.simplify(_parse(lim_body))
    return str(result), sympy.latex(result)


def _do_simplify(latex: str, steps: list[str]) -> tuple[str, str]:
    steps.append("Simplifying expression")
    result = sympy.simplify(_parse(latex))
    steps.append("Applied SymPy simplification rules")
    return str(result), sympy.latex(result)


_OP_FUNCS: dict[str, Any] = {
    "solve": _do_solve,
    "integrate": _do_integrate,
    "differentiate": _do_differentiate,
    "limit": _do_limit,
    "simplify": _do_simplify,
}


def solve_expression(latex: str) -> dict[str, Any]:
    operation = detect_operation(latex)
    steps: list[str] = []
    result_holder: list[dict] = []

    def _run() -> None:
        try:
            sol_str, sol_latex = _OP_FUNCS[operation](latex, steps)
            result_holder.append({
                "status": "success",
                "operation": operation,
                "operation_label": _OP_LABELS[operation],
                "steps": steps,
                "solution": sol_str,
                "latex_result": sol_latex,
            })
        except Exception as primary_exc:
            steps.append("Primary operation failed; falling back to simplify")
            try:
                sol_str, sol_latex = _do_simplify(latex, steps)
                result_holder.append({
                    "status": "success",
                    "operation": "simplify",
                    "operation_label": "Simplifying",
                    "steps": steps,
                    "solution": sol_str,
                    "latex_result": sol_latex,
                })
            except Exception as fallback_exc:
                result_holder.append({
                    "status": "failed",
                    "error": f"Could not parse expression: {fallback_exc}",
                    "operation": operation,
                })

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=10)

    if thread.is_alive():
        return {
            "status": "failed",
            "error": "Expression too complex to solve in time.",
            "operation": operation,
        }

    return result_holder[0] if result_holder else {
        "status": "failed",
        "error": "Solver produced no result.",
        "operation": operation,
    }

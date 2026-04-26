# Plumber API exposing Ryacas as a cross-check symbolic solver.
# Run via: Rscript run_plumber.r  -> http://127.0.0.1:8003

library(plumber)
library(Ryacas)
library(jsonlite)

# ---- LaTeX -> Yacas string conversion ------------------------------------

latex_to_yacas <- function(latex) {
  s <- latex
  # \frac{a}{b} -> ((a)/(b))
  s <- gsub("\\\\frac\\{([^{}]*)\\}\\{([^{}]*)\\}", "((\\1)/(\\2))", s, perl = TRUE)
  # \sqrt{a} -> Sqrt(a)
  s <- gsub("\\\\sqrt\\{([^{}]*)\\}", "Sqrt(\\1)", s, perl = TRUE)
  # \cdot, \times -> *
  s <- gsub("\\\\(cdot|times)", "*", s, perl = TRUE)
  # \div -> /
  s <- gsub("\\\\div", "/", s, perl = TRUE)
  # \pi, \infty
  s <- gsub("\\\\pi", "Pi", s, perl = TRUE)
  s <- gsub("\\\\infty", "Infinity", s, perl = TRUE)
  # strip remaining backslashes from generic functions/letters (\sin, \alpha, ...)
  s <- gsub("\\\\([a-zA-Z]+)", "\\1", s, perl = TRUE)
  # remove all braces (we've already handled the structured ones above)
  s <- gsub("\\{|\\}", "", s, perl = TRUE)
  # implicit multiplication: 2x -> 2*x
  s <- gsub("([0-9])([a-zA-Z(])", "\\1*\\2", s, perl = TRUE)
  trimws(s)
}

detect_operation <- function(latex) {
  if (grepl("(?<![<>!])=(?!=)", latex, perl = TRUE)) return("solve")
  if (grepl("\\\\int", latex)) return("integrate")
  if (grepl("\\\\frac\\s*\\{d", latex) || grepl("'", latex) || grepl("\\\\prime", latex)) {
    return("differentiate")
  }
  if (grepl("\\\\lim", latex)) return("limit")
  "simplify"
}

# ---- per-operation solvers -----------------------------------------------

run_yacas <- function(expr_str) {
  trimws(as.character(yac(expr_str)))
}

solve_with_ryacas <- function(latex) {
  tryCatch({
    op <- detect_operation(latex)
    yacas_expr <- latex_to_yacas(latex)
    if (op == "solve") {
      sides <- strsplit(yacas_expr, "=", fixed = TRUE)[[1]]
      lhs <- sides[1]; rhs <- sides[2]
      sol <- run_yacas(sprintf("Solve(%s == %s, x)", lhs, rhs))
    } else if (op == "integrate") {
      body <- gsub("int", "", yacas_expr, fixed = TRUE)
      body <- trimws(gsub("dx$", "", body))
      sol <- run_yacas(sprintf("Integrate(x) %s", body))
    } else if (op == "differentiate") {
      body <- gsub("frac\\{d\\}\\{dx\\}", "", yacas_expr, perl = TRUE)
      body <- gsub("d/dx", "", body, fixed = TRUE)
      sol <- run_yacas(sprintf("D(x) %s", trimws(body)))
    } else if (op == "limit") {
      sol <- run_yacas(sprintf("Limit(x, 0) %s", yacas_expr))
    } else {
      sol <- run_yacas(sprintf("Simplify(%s)", yacas_expr))
    }
    list(
      status = "success",
      operation = op,
      solution = sol,
      latex_result = sol
    )
  }, error = function(e) {
    list(status = "failed", error = conditionMessage(e))
  })
}

# ---- HTTP endpoints ------------------------------------------------------

#* Health probe
#* @get /health
function() {
  list(status = "ok", solver = "ryacas")
}

#* Solve an expression
#* @get /solve_ryacas
#* @param latex character The LaTeX expression
function(latex = "") {
  if (nchar(latex) == 0) {
    return(list(status = "failed", error = "Empty latex"))
  }
  solve_with_ryacas(latex)
}

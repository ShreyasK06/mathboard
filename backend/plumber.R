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

# Split a string on commas that are at the top level (not inside parens or braces).
split_top_level_commas <- function(s) {
  out <- character()
  buf <- ""
  depth <- 0L
  chars <- strsplit(s, "", fixed = TRUE)[[1]]
  for (ch in chars) {
    if (ch == "(" || ch == "{" || ch == "[") {
      depth <- depth + 1L; buf <- paste0(buf, ch)
    } else if (ch == ")" || ch == "}" || ch == "]") {
      depth <- depth - 1L; buf <- paste0(buf, ch)
    } else if (ch == "," && depth == 0L) {
      out <- c(out, buf); buf <- ""
    } else {
      buf <- paste0(buf, ch)
    }
  }
  if (nchar(buf) > 0) out <- c(out, buf)
  out
}

# Parse Yacas Solve output like "{x==7}" or "{x==2,x==(-2)}" into a vector of
# raw value strings ("7", "2", "-2"). If Yacas couldn't solve, returns the
# input unchanged in a single-element vector.
parse_solve_values <- function(raw, var = "x") {
  s <- trimws(raw)
  if (!grepl("^\\{.*\\}$", s)) return(s)
  inner <- substr(s, 2, nchar(s) - 1L)
  parts <- split_top_level_commas(inner)
  vals <- vapply(parts, function(p) {
    p <- trimws(p)
    p <- sub(paste0("^", var, "\\s*==\\s*"), "", p, perl = TRUE)
    # Strip a single pair of fully-enclosing parens, e.g. "(-2)" -> "-2"
    if (grepl("^\\(.+\\)$", p)) {
      depth <- 0L; encloses <- TRUE
      chars <- strsplit(p, "", fixed = TRUE)[[1]]
      for (i in seq_along(chars)) {
        if (chars[i] == "(") depth <- depth + 1L
        else if (chars[i] == ")") depth <- depth - 1L
        if (depth == 0L && i < length(chars)) { encloses <- FALSE; break }
      }
      if (encloses) p <- substr(p, 2, nchar(p) - 1L)
    }
    trimws(p)
  }, character(1), USE.NAMES = FALSE)
  vals
}

# Take a raw yacas value string and produce its LaTeX form via TeXForm.
# Falls back to the raw string if TeXForm fails.
to_tex <- function(value_str) {
  out <- tryCatch(run_yacas(sprintf("TeXForm(%s)", value_str)),
                  error = function(e) value_str)
  if (length(out) == 0 || is.na(out) || nchar(out) == 0) return(value_str)
  out
}

solve_with_ryacas <- function(latex) {
  tryCatch({
    op <- detect_operation(latex)
    yacas_expr <- latex_to_yacas(latex)
    if (op == "solve") {
      sides <- strsplit(yacas_expr, "=", fixed = TRUE)[[1]]
      lhs <- trimws(sides[1]); rhs <- trimws(sides[2])
      raw <- run_yacas(sprintf("Solve(%s == %s, x)", lhs, rhs))
      vals <- parse_solve_values(raw, var = "x")
      solution <- paste(vals, collapse = ", ")
      latex_result <- paste(vapply(vals, to_tex, character(1)), collapse = ", ")
    } else if (op == "integrate") {
      body <- gsub("int", "", yacas_expr, fixed = TRUE)
      body <- trimws(gsub("dx$", "", body))
      raw <- run_yacas(sprintf("Integrate(x) %s", body))
      solution <- raw
      latex_result <- to_tex(raw)
    } else if (op == "differentiate") {
      body <- gsub("frac\\{d\\}\\{dx\\}", "", yacas_expr, perl = TRUE)
      body <- gsub("d/dx", "", body, fixed = TRUE)
      raw <- run_yacas(sprintf("D(x) %s", trimws(body)))
      solution <- raw
      latex_result <- to_tex(raw)
    } else if (op == "limit") {
      raw <- run_yacas(sprintf("Limit(x, 0) %s", yacas_expr))
      solution <- raw
      latex_result <- to_tex(raw)
    } else {
      raw <- run_yacas(sprintf("Simplify(%s)", yacas_expr))
      solution <- raw
      latex_result <- to_tex(raw)
    }
    list(
      status = "success",
      operation = op,
      solution = solution,
      latex_result = latex_result
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

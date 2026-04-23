# plumber.R
library(plumber)
library(Ryacas)
library(jsonlite)

# Basic translation from LaTeX format to Yacas syntax
clean_latex_for_yacas <- function(latex_str) {
  s <- latex_str
  # \frac{a}{b} -> (a)/(b)  — must run before generic brace removal
  s <- gsub("\\\\frac\\{([^}]*)\\}\\{([^}]*)\\}", "(\\1)/(\\2)", s)
  # x^{2} -> x^2
  s <- gsub("\\^\\{([^}]*)\\}", "^\\1", s)
  # \sqrt{x} -> sqrt(x)
  s <- gsub("\\\\sqrt\\{([^}]*)\\}", "sqrt(\\1)", s)
  # Greek letters
  s <- gsub("\\\\alpha", "alpha", s)
  s <- gsub("\\\\beta",  "beta",  s)
  s <- gsub("\\\\gamma", "gamma", s)
  s <- gsub("\\\\theta", "theta", s)
  s <- gsub("\\\\pi",    "pi",    s)
  # Operators
  s <- gsub("\\\\times", "*", s)
  s <- gsub("\\\\cdot",  "*", s)
  s <- gsub("\\\\div",   "/", s)
  # Remove remaining unknown backslash commands
  s <- gsub("\\\\[a-zA-Z]+", "", s)
  # Convert remaining braces to parentheses
  s <- gsub("\\{", "(", s)
  s <- gsub("\\}", ")", s)
  return(s)
}

#* @apiTitle Mathboard R Compute Engine
#* @post /solve
function(req) {
  # Safely parse the incoming JSON
  body <- tryCatch(jsonlite::fromJSON(req$postBody), error = function(e) NULL)
  
  if (is.null(body) || is.null(body$latex) || body$latex == "") {
    return(list(error = "No LaTeX string provided by Python gateway."))
  }
  
  latex_str <- body$latex
  yacas_expr <- clean_latex_for_yacas(latex_str)
  
  # Attempt symbolic evaluation
  result <- tryCatch({
    # "Simplify" instructs Yacas to evaluate algebra and arithmetic
    evaluated <- yac_str(paste0("Simplify(", yacas_expr, ")"))
    
    list(
      original_latex = latex_str,
      yacas_parsed = yacas_expr,
      solution = evaluated,
      status = "success"
    )
  }, error = function(e) {
    list(
      original_latex = latex_str,
      error = paste("R Evaluation Failed:", e$message),
      status = "failed"
    )
  })
  
  return(result)
}
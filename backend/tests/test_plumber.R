library(testthat)
source("../plumber.R")

test_that("latex_to_yacas converts basic operators", {
  expect_equal(latex_to_yacas("x + 5"), "x + 5")
  # \frac{1}{2} -> ((1)/(2)) (parens preserved for operator precedence)
  expect_match(latex_to_yacas("\\frac{1}{2}"), "1.*/.*2")
  expect_match(latex_to_yacas("x^2"), "x^2", fixed = TRUE)
})

test_that("detect_operation classifies expressions", {
  expect_equal(detect_operation("x + 5 = 12"), "solve")
  expect_equal(detect_operation("\\int x dx"), "integrate")
  expect_equal(detect_operation("\\frac{d}{dx} x^2"), "differentiate")
  expect_equal(detect_operation("\\lim_{x \\to 0} x"), "limit")
  expect_equal(detect_operation("x^2 + 2x + 1"), "simplify")
})

test_that("solve_with_ryacas returns a valid result for x + 5 = 12", {
  result <- solve_with_ryacas("x + 5 = 12")
  expect_equal(result$status, "success")
  expect_equal(result$operation, "solve")
  expect_match(result$solution, "7", fixed = TRUE)
})

test_that("solve_with_ryacas returns a valid result for simplification", {
  result <- solve_with_ryacas("x^2 + 2*x + 1")
  expect_equal(result$status, "success")
})

test_that("solve_with_ryacas returns failed status for malformed input", {
  # Unbalanced parens after sanitization break Yacas's parser.
  result <- solve_with_ryacas("(((")
  expect_equal(result$status, "failed")
})

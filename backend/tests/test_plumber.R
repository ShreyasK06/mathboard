library(testthat)
source("../plumber.r")  # path relative to tests/ directory

test_that("frac is converted to division", {
  expect_equal(clean_latex_for_yacas("\\frac{x}{2}"), "(x)/(2)")
})

test_that("braced exponent is unwrapped", {
  expect_equal(clean_latex_for_yacas("x^{2}"), "x^2")
})

test_that("sqrt is converted", {
  expect_equal(clean_latex_for_yacas("\\sqrt{x}"), "sqrt(x)")
})

test_that("Greek letters are converted", {
  expect_equal(clean_latex_for_yacas("\\alpha"), "alpha")
  expect_equal(clean_latex_for_yacas("\\pi"), "pi")
  expect_equal(clean_latex_for_yacas("\\theta"), "theta")
})

test_that("multiplication operators are converted", {
  expect_equal(clean_latex_for_yacas("x \\times y"), "x * y")
  expect_equal(clean_latex_for_yacas("x \\cdot y"), "x * y")
})

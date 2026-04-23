# run_r.R
library(plumber)
pr("plumber.R") |> pr_run(port=8003)
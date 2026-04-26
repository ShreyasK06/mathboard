# Launch the Plumber API on port 8003.
library(plumber)
pr <- plumb("plumber.R")
pr$run(host = "127.0.0.1", port = 8003)

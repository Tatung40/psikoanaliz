# Independent R mirt 1.48 ML calibration of a 3PL or 4PL (validation only).
# Usage: Rscript reference.R data.csv 3PL|4PL out_dir
# ML, no priors, theta ~ N(0, 1) fixed; 101 equally spaced points on [-6, 6]; TOL 1e-8; SQUAREM; Oakes SEs.
# Physical a, b, c (= g), d (= u) and their delta-method SEs come from coef(IRTpars = TRUE, printSE = TRUE).
suppressPackageStartupMessages({library(mirt); library(jsonlite)})
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 3, as.character(packageVersion("mirt")) == "1.48", args[2] %in% c("3PL", "4PL"))
dat <- read.csv(args[1]); model <- args[2]; out <- args[3]
fit <- mirt(dat, 1, itemtype = model, quadpts = 101, theta_lim = c(-6, 6), TOL = 1e-8, SE = TRUE,
            SE.type = "Oakes", accelerate = "squarem", verbose = FALSE, technical = list(NCYCLES = 100000))
items <- coef(fit, IRTpars = TRUE, printSE = TRUE)[seq_len(ncol(dat))]
pick <- function(row, p) unname(sapply(items, function(m) if (p %in% colnames(m)) m[row, p] else NA_real_))
names_r <- c(a = "a", b = "b", c = "g", d = "u")
par <- lapply(names_r, function(p) pick("par", p))
se <- lapply(names_r, function(p) pick("SE", p))
if (model == "3PL") { par$d <- rep(1, ncol(dat)); se$d <- rep(NA_real_, ncol(dat)) }
result <- list(model = model, mirt = as.character(packageVersion("mirt")), R = R.version.string,
               converged = extract.mirt(fit, "converged"), iterations = extract.mirt(fit, "iterations"),
               loglik = extract.mirt(fit, "logLik"), par = par, se = se)
write_json(result, file.path(out, "r_result.json"), digits = NA, auto_unbox = TRUE, na = "null")
cat(sprintf("%s converged=%s iterations=%d logLik=%.10f\n", model, result$converged, result$iterations, result$loglik))

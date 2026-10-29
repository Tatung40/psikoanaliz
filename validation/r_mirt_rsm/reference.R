# Independent R mirt 1.48 rating scale model (validation only).
# Usage: Rscript reference.R data.csv out_dir
# itemtype "rsm": slopes 1 (fixed), common step parameters, item locations; latent mean 0 fixed and latent
# variance free (the Rasch identification of core.irt_rsm). ML, 101 points on [-6, 6], TOL 1e-8.
suppressPackageStartupMessages({library(mirt); library(jsonlite)})
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2, as.character(packageVersion("mirt")) == "1.48")
dat <- read.csv(args[1], check.names = FALSE); out <- args[2]
sv <- mirt(dat, 1, itemtype = "rsm", pars = "values")
sv$value[sv$name == "a1"] <- 1; sv$est[sv$name == "a1"] <- FALSE
sv$est[sv$name == "COV_11"] <- TRUE; sv$value[sv$name == "MEAN_1"] <- 0; sv$est[sv$name == "MEAN_1"] <- FALSE
fit <- mirt(dat, 1, itemtype = "rsm", pars = sv, quadpts = 101, theta_lim = c(-6, 6), TOL = 1e-8, SE = FALSE,
            verbose = FALSE, technical = list(NCYCLES = 100000))
theta <- matrix(read.csv(file.path(out, "theta.csv"))$theta)
trace <- lapply(seq_len(ncol(dat)), function(j) unname(probtrace(extract.item(fit, j), theta)))
result <- list(mirt = as.character(packageVersion("mirt")), R = R.version.string,
               converged = extract.mirt(fit, "converged"), iterations = extract.mirt(fit, "iterations"),
               loglik = extract.mirt(fit, "logLik"), nest = extract.mirt(fit, "nest"),
               sd = sqrt(coef(fit, simplify = TRUE)$cov[1, 1]), probtrace = trace)
write_json(result, file.path(out, "r_result.json"), digits = NA, auto_unbox = TRUE)
cat(sprintf("RSM converged=%s iterations=%d logLik=%.10f nest=%d sd=%.8f\n", result$converged,
            result$iterations, result$loglik, result$nest, result$sd))

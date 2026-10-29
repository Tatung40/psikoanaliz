# Independent R mirt 1.48 calibration of one unidimensional polytomous model (validation only).
# Usage: Rscript reference.R data.csv GRM|PCM|GPCM|NRM out_dir
# ML, no priors, theta ~ N(0, 1) (mean 0, variance 1 fixed), 101 equally spaced points on [-6, 6].
# GRM: graded (a1, d1..d3); GPCM: gpcm with default scoring ak = 0..3 and d0 = 0; PCM: gpcm with a1 fixed 1;
# NRM: nominal with default identification ak0 = 0, ak3 = 3, d0 = 0 (category 0 reference).
suppressPackageStartupMessages({library(mirt); library(jsonlite)})
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 3, as.character(packageVersion("mirt")) == "1.48")
dat <- read.csv(args[1], check.names = FALSE)
model <- args[2]; out <- args[3]
stopifnot(model %in% c("GRM", "PCM", "GPCM", "NRM"), all(sapply(dat, function(v) setequal(unique(v), 0:3))))
itemtype <- switch(model, GRM = "graded", PCM = "gpcm", GPCM = "gpcm", NRM = "nominal")
sv <- mirt(dat, 1, itemtype = itemtype, pars = "values")
if (model == "PCM") { sv$value[sv$name == "a1"] <- 1; sv$est[sv$name == "a1"] <- FALSE }
stopifnot(!any(sv$est[sv$name %in% c("MEAN_1", "COV_11")]))
# NRM: EM with default Ramsay acceleration did not reach TOL in 20000 cycles (first CI run); SQUAREM and
# more cycles change only the path to R's own ML solution, never the convergence tolerance.
fit <- mirt(dat, 1, itemtype = itemtype, pars = sv, quadpts = 101, theta_lim = c(-6, 6), TOL = 1e-8,
            SE = FALSE, verbose = FALSE, accelerate = if (model == "NRM") "squarem" else "Ramsay",
            technical = list(NCYCLES = if (model == "NRM") 100000 else 20000))
coefs <- coef(fit, IRTpars = FALSE, simplify = TRUE)$items
theta <- matrix(read.csv(file.path(out, "theta.csv"))$theta)
trace <- lapply(seq_len(ncol(dat)), function(j) unname(probtrace(extract.item(fit, j), theta)))
scores <- fscores(fit, method = "EAP", full.scores.SE = TRUE, quadpts = 101, theta_lim = c(-6, 6))
result <- list(model = model, mirt = as.character(packageVersion("mirt")), R = R.version.string,
               converged = extract.mirt(fit, "converged"), iterations = extract.mirt(fit, "iterations"),
               loglik = extract.mirt(fit, "logLik"), nest = extract.mirt(fit, "nest"),
               coef = as.data.frame(coefs), probtrace = trace,
               eap = unname(scores[, "F1"]), eap_se = unname(scores[, "SE_F1"]))
write_json(result, file.path(out, "r_result.json"), digits = NA, auto_unbox = TRUE, dataframe = "columns")
cat(sprintf("%s converged=%s iterations=%d logLik=%.10f nest=%d\n", model, result$converged,
            result$iterations, result$loglik, result$nest))

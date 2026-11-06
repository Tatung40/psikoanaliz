# Rscript fit.R data.csv outdir — PROTOCOL.json strategy: R mirt 1.48 ML, no priors.
# EM on GH61 (identical to PR #146) -> native unbounded BFGS BL (mirt optim, analytic gradient), warm-started on
# GH61, GH101, GH161; at most 3 BL calls per grid while R reports non-convergence (mirt fixes BFGS maxit = 100).
# mirt 1.48 bounded BL is unusable (invalid optim method "nlminb"); equivalence to the bounded MLE holds only if the
# final estimate is inside the common box, which evaluate.py checks.
args <- commandArgs(trailingOnly = TRUE); stopifnot(length(args) == 2L)
suppressPackageStartupMessages(library(mirt)); stopifnot(as.character(packageVersion("mirt")) == "1.48")
dat <- read.csv(args[1], check.names = FALSE); out <- args[2]; dir.create(out, recursive = TRUE, showWarnings = FALSE)
gh <- function(n) { J <- matrix(0, n, n); k <- seq_len(n - 1); J[cbind(k, k + 1)] <- sqrt(k / 2); J[cbind(k + 1, k)] <- sqrt(k / 2)
  e <- eigen(J, symmetric = TRUE); o <- order(e$values); list(node = e$values[o] * sqrt(2), weight = e$vectors[1, o]^2) }
tech <- function(n) { g <- gh(n); list(NCYCLES = 20000, customTheta = matrix(g$node, ncol = 1),
                                       customPriorFun = function(Theta, Etable) g$weight) }
sv <- mirt(dat, 1, itemtype = "4PL", pars = "values", verbose = FALSE)
box <- function(sv) { for (r in list(c("a1", .1, 5), c("g", 0, .5), c("u", .5, 1))) {
  i <- sv$name == r[1] & sv$est; lo <- as.numeric(r[2]); hi <- as.numeric(r[3])
  sv$lbound[i] <- lo; sv$ubound[i] <- hi; sv$value[i] <- pmin(pmax(sv$value[i], lo + 1e-3), hi - 1e-3) }; sv }
sv <- box(sv); meta <- list()
save <- function(fit, stage, q, secs) {
  raw <- coef(fit, simplify = TRUE, IRTpars = FALSE)$items
  write.csv(data.frame(item = colnames(dat), a = raw[, "a1"], b = -raw[, "d"] / raw[, "a1"], c = raw[, "g"], d = raw[, "u"]),
            file.path(out, paste0(stage, ".csv")), row.names = FALSE)
  meta[[stage]] <<- data.frame(stage = stage, quadpts = q, converged = extract.mirt(fit, "converged"),
                               iterations = extract.mirt(fit, "iterations"), loglik = extract.mirt(fit, "logLik"), seconds = secs)
  write.csv(do.call(rbind, meta), file.path(out, "metadata.csv"), row.names = FALSE) }
t0 <- proc.time()[["elapsed"]]
fit <- mirt(dat, 1, itemtype = "4PL", pars = sv, method = "EM", TOL = 1e-8, SE = FALSE, verbose = FALSE, technical = tech(61))
save(fit, "EM_GH61", 61, proc.time()[["elapsed"]] - t0)
for (q in c(61L, 101L, 161L)) {
  for (r in 1:3) {
    t0 <- proc.time()[["elapsed"]]; v <- mod2values(fit); v$lbound[v$est] <- -Inf; v$ubound[v$est] <- Inf
    fit <- mirt(dat, 1, itemtype = "4PL", pars = v, method = "BL", TOL = 1e-12, SE = FALSE, verbose = FALSE, technical = tech(q))
    save(fit, sprintf("BL_GH%d_r%d", q, r), q, proc.time()[["elapsed"]] - t0)
    if (extract.mirt(fit, "converged")) break }
  file.copy(file.path(out, sprintf("BL_GH%d_r%d.csv", q, r)), file.path(out, sprintf("BL_GH%d.csv", q)), overwrite = TRUE) }
capture.output(sessionInfo(), file = file.path(out, "sessionInfo.txt"))

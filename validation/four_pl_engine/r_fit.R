# Rscript r_fit.R data.csv n_quadpts outdir  — one bounded 4PL ML fit with R mirt 1.48 on a fixed GH grid.
args <- commandArgs(trailingOnly=TRUE); stopifnot(length(args) == 3L)
suppressPackageStartupMessages(library(mirt)); stopifnot(as.character(packageVersion("mirt")) == "1.48")
dat <- read.csv(args[1], check.names=FALSE); q <- as.integer(args[2]); out <- args[3]
dir.create(out, recursive=TRUE, showWarnings=FALSE)
gh <- function(n) { # Golub-Welsch, physicists' Hermite -> N(0,1)
  J <- matrix(0, n, n); k <- seq_len(n - 1); J[cbind(k, k + 1)] <- sqrt(k / 2); J[cbind(k + 1, k)] <- sqrt(k / 2)
  e <- eigen(J, symmetric=TRUE); o <- order(e$values)
  list(node=e$values[o] * sqrt(2), weight=(e$vectors[1, o]^2))
}
g <- gh(q)
sv <- mirt(dat, 1, itemtype="4PL", pars="values", verbose=FALSE)
box <- function(name, lo, hi) { i <- sv$name == name & sv$est; sv$lbound[i] <<- lo; sv$ubound[i] <<- hi
  sv$value[i] <<- pmin(pmax(sv$value[i], lo + 1e-3), hi - 1e-3) }
box("a1", .1, 5); box("g", 0, .5); box("u", .5, 1)
t0 <- proc.time()[["elapsed"]]
fit <- mirt(dat, 1, itemtype="4PL", pars=sv, method="EM", TOL=1e-8, SE=FALSE, verbose=FALSE,
            technical=list(NCYCLES=20000, customTheta=matrix(g$node, ncol=1),
                           customPriorFun=function(Theta, Etable) g$weight))
secs <- proc.time()[["elapsed"]] - t0
raw <- coef(fit, simplify=TRUE, IRTpars=FALSE)$items
write.csv(data.frame(item=colnames(dat), a=raw[, "a1"], b=-raw[, "d"] / raw[, "a1"], c=raw[, "g"], d=raw[, "u"]),
          file.path(out, "parameters.csv"), row.names=FALSE)
write.csv(data.frame(engine="R mirt", version=as.character(packageVersion("mirt")), quadpts=q,
                     converged=extract.mirt(fit, "converged"), cycles=extract.mirt(fit, "iterations"),
                     loglik=extract.mirt(fit, "logLik"), seconds=secs), file.path(out, "metadata.csv"), row.names=FALSE)
capture.output(sessionInfo(), file=file.path(out, "sessionInfo.txt"))

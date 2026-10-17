# R mirt reference for FREE 4PL estimation (B5).
#
# Usage (repository root):  Rscript scripts/mirt_free_4pl.R [fixture_dir]
# Reads cases_free4pl.csv, gh21.csv, data_*/items_* (scripts/make_mirt_fixtures.py)
# and writes mirt_free4pl_results.csv and mirt_free4pl_session.txt.
#
# Matched to the Python fit_irt / fit_irt_multistart set-up:
#  * same data; P = g + (u-g)/(1+exp(-(a1*theta+d))), Python a=a1, b=-d/a1, c=g, d=u
#  * latent N(0,1) fixed (mirt default MEAN_1=0, COV_11=1 not estimated; recorded)
#  * identical 21-node Gauss-Hermite nodes/weights: technical customTheta/customPriorFun
#    (used by the EM E-step and logLik; mirt 1.48 R/utils.R::updatePrior)
#  * all a1, d, g, u free; no item priors (prior.type recorded)
#  * starting values = Python initial values for initial_shift 0, -1, +1
#  * unbounded fit (mirt default: g, u on the logit scale) AND a fit restricted to the
#    Python box a1 in [.15, 4], g in [0, .35], u in [.65, 1] via the documented pars
#    data.frame lbound/ubound columns (g/u bounds given raw; mirt applies logit,
#    R/utils.R line ~1329). Python's b in [-4, 4] has no mirt counterpart (d = -a b).
#  * mirt stops when max |parameter change| < TOL (R/utils.R::hasConverged), Python
#    when |dlogL| < tolerance; both run tight and solutions are compared through the
#    same objective (logLik, gradient, KKT) rather than through the stopping rule.
# Only documented interfaces are used: mirt(pars=, TOL=, technical=list(customTheta,
# customPriorFun, NCYCLES)), mod2values, extract.mirt, logLik, numerical_deriv.
suppressPackageStartupMessages(library(mirt))

args <- commandArgs(trailingOnly = TRUE)
dir <- if (length(args)) args[1] else "tests/fixtures/mirt"
cases <- read.csv(file.path(dir, "cases_free4pl.csv"), stringsAsFactors = FALSE)
gh <- read.csv(file.path(dir, "gh21.csv"))
tech <- list(customTheta = matrix(gh$theta),
             customPriorFun = function(Theta, Etable) gh$weight,
             warn = FALSE, message = FALSE)
TOL <- 1e-7
out <- list()
add <- function(...) out[[length(out) + 1L]] <<- data.frame(...)
clip <- function(v, lo, hi) pmin(pmax(v, lo), hi)

for (r in seq_len(nrow(cases))) {
  case <- cases$case[r]
  dat <- read.csv(file.path(dir, paste0("data_", case, ".csv")))
  items <- colnames(dat)
  sv0 <- mirt(dat, 1, itemtype = "4PL", pars = "values", technical = tech)
  add(case = case, run = "setup", quantity = "prior_types",
      row = paste(unique(sv0$prior.type), collapse = ";"), col = "", value = NA_real_)
  add(case = case, run = "setup", quantity = "latent_estimated",
      row = paste(sv0$name[sv0$item == "GROUP"], sv0$est[sv0$item == "GROUP"], collapse = ";"),
      col = "", value = NA_real_)
  b0 <- clip(-qlogis(colMeans(dat)), -3, 3)          # Python _initial()
  est <- sv0$est
  logit_par <- sv0$name[est] %in% c("g", "u")

  starts <- function(shift) {                         # Python fit_irt initial values
    a <- if (shift == 0) 1 else clip(1 + .4 * shift, .16, 3.99)
    g <- if (shift == 0) .1 else clip(.1 + .08 * shift, .001, .34)
    u <- if (shift == 0) .95 else clip(.95 - .15 * shift, .66, .99)
    v <- sv0
    for (j in seq_along(items)) {
      sel <- v$item == items[j]
      v$value[sel & v$name == "a1"] <- a
      v$value[sel & v$name == "d"] <- -a * b0[j]
      v$value[sel & v$name == "g"] <- g
      v$value[sel & v$name == "u"] <- u
    }
    v
  }
  fixed_ll <- function(p, ...) {
    v <- sv0
    p[logit_par] <- plogis(p[logit_par])
    v$value[est] <- p
    as.numeric(logLik(mirt(dat, 1, itemtype = "4PL", pars = v, TOL = NaN,
                           technical = tech, verbose = FALSE)))
  }
  runs <- list()
  for (shift in c(0, -1, 1)) runs[[paste0("free_s", shift)]] <- list(sv = starts(shift), shift = shift)
  boxed <- starts(0)
  boxed$lbound[boxed$name == "a1"] <- .15; boxed$ubound[boxed$name == "a1"] <- 4
  boxed$lbound[boxed$name == "g"] <- 0;    boxed$ubound[boxed$name == "g"] <- .35
  boxed$lbound[boxed$name == "u"] <- .65;  boxed$ubound[boxed$name == "u"] <- 1
  runs[["box_s0"]] <- list(sv = boxed, shift = 0)

  for (run in names(runs)) {
    t0 <- proc.time()[3]
    fit <- mirt(dat, 1, itemtype = "4PL", pars = runs[[run]]$sv, TOL = TOL,
                technical = c(tech, list(NCYCLES = 20000)), verbose = FALSE)
    secs <- proc.time()[3] - t0
    add(case = case, run = run, quantity = "start_shift", row = "", col = "", value = runs[[run]]$shift)
    add(case = case, run = run, quantity = "converged", row = "", col = "",
        value = as.numeric(extract.mirt(fit, "converged")))
    add(case = case, run = run, quantity = "iterations", row = "", col = "",
        value = extract.mirt(fit, "iterations"))
    add(case = case, run = run, quantity = "seconds", row = "", col = "", value = secs)
    add(case = case, run = run, quantity = "loglik", row = "", col = "", value = as.numeric(logLik(fit)))
    mv <- mod2values(fit)
    for (j in seq_along(items)) for (nm in c("a1", "d", "g", "u"))
      add(case = case, run = run, quantity = "estimate", row = items[j], col = nm,
          value = mv$value[mv$item == items[j] & mv$name == nm])
    internal <- mv$value[est]
    internal[logit_par] <- qlogis(internal[logit_par])
    pname <- paste0(mv$name[est], ".", mv$item[est])
    if (all(is.finite(internal))) {
      g <- numerical_deriv(internal, fixed_ll, gradient = TRUE, type = "Richardson")
      for (i in seq_along(g)) add(case = case, run = run, quantity = "gradient_internal",
                                  row = pname[i], col = "", value = g[i])
      if (run == "free_s0") {
        H <- numerical_deriv(internal, fixed_ll, gradient = FALSE, type = "central", delta = 1e-4)
        for (i in seq_len(nrow(H))) for (k in seq_len(ncol(H)))
          add(case = case, run = run, quantity = "hessian_internal_central",
              row = pname[i], col = pname[k], value = H[i, k])
      }
    } else {
      add(case = case, run = run, quantity = "gradient_status",
          row = "non-finite internal value (g or u on the 0/1 boundary)", col = "", value = NA_real_)
    }
  }
}
res <- do.call(rbind, out)
write.csv(res, file.path(dir, "mirt_free4pl_results.csv"), row.names = FALSE)
writeLines(c(R.version.string,
             paste("mirt", as.character(packageVersion("mirt"))),
             paste("SimDesign", as.character(packageVersion("SimDesign"))),
             paste("Deriv", as.character(packageVersion("Deriv"))),
             paste("mirt source:", Sys.getenv("MIRT_SOURCE", "CRAN/unspecified")),
             paste("EM TOL:", TOL, "| NCYCLES: 20000 | accelerate: default"),
             paste("generated:", format(Sys.time(), tz = "UTC", usetz = TRUE))),
           file.path(dir, "mirt_free4pl_session.txt"))
cat("Wrote", nrow(res), "rows\n")

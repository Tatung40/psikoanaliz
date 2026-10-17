# R mirt reference values for the Python 3PL/4PL comparison (B3).
#
# Usage (from the repository root):
#   Rscript scripts/mirt_reference_3pl4pl.R [fixture_dir]
# Reads tests/fixtures/mirt/{cases,gh21,data_*,items_*}.csv (made by
# scripts/make_mirt_fixtures.py) and writes mirt_results.csv and
# mirt_session.txt to the same directory.
#
# Documented mirt interfaces only:
#  * pars='values' / pars=<data.frame>: supply parameters (g, u given raw).
#  * TOL=NaN: evaluate the model at the supplied (starting) values, no M-step.
#  * technical=list(customTheta, customPriorFun): integration grid and
#    normalized prior weights -> identical Gauss-Hermite nodes/weights to Python.
#  * SE=TRUE, SE.type='Richardson'/'central': numerical observed Hessian.
#  * numerical_deriv(): gradient of logLik.
#  * vcov(), logLik(), probtrace(extract.item()).
# Parameterization (mirt 1.48 source, src/traceLinePts.cpp::P_dich):
#   P = g + (u - g) / (1 + exp(-(a1 * theta + d))), g and u stored internally on
#   the logit scale. Python: a = a1, b = -d / a1, c = g, d_python = u.
suppressPackageStartupMessages(library(mirt))

args <- commandArgs(trailingOnly = TRUE)
dir <- if (length(args)) args[1] else "tests/fixtures/mirt"
cases <- read.csv(file.path(dir, "cases.csv"), stringsAsFactors = FALSE)
gh <- read.csv(file.path(dir, "gh21.csv"))
tech <- list(customTheta = matrix(gh$theta),
             customPriorFun = function(Theta, Etable) gh$weight,
             warn = FALSE, message = FALSE)
out <- list()
add <- function(case, quantity, row, col, value)
  out[[length(out) + 1L]] <<- data.frame(case = case, quantity = quantity,
                                         row = row, col = col, value = value)

for (r in seq_len(nrow(cases))) {
  case <- cases$case[r]; model <- cases$model[r]
  dat <- read.csv(file.path(dir, paste0("data_", case, ".csv")))
  if (cases$design[r] == "est") {
    # Free ML estimation on the same GH-21 grid (EM, no priors, tight tolerance).
    sv0 <- mirt(dat, 1, itemtype = model, pars = "values", technical = tech)
    add(case, "prior_types", paste(unique(sv0$prior.type), collapse = ";"), "", NA_real_)
    fit <- mirt(dat, 1, itemtype = model, TOL = 1e-8,
                technical = c(tech, list(NCYCLES = 5000)), verbose = FALSE)
    add(case, "fit_converged", "", "", as.numeric(extract.mirt(fit, "converged")))
    add(case, "fit_iterations", "", "", extract.mirt(fit, "iterations"))
    add(case, "fit_loglik", "", "", as.numeric(logLik(fit)))
    cf <- coef(fit, IRTpars = TRUE, simplify = TRUE)$items
    for (j in seq_len(nrow(cf))) for (p in c("a", "b", "g"))
      add(case, "fit_irtpars", rownames(cf)[j], p, cf[j, p])
    next
  }
  it <- read.csv(file.path(dir, paste0("items_", case, ".csv")), stringsAsFactors = FALSE)
  sv <- mirt(dat, 1, itemtype = model, pars = "values", technical = tech)
  for (j in seq_len(nrow(it))) {
    item <- it$Madde[j]
    sv$value[sv$item == item & sv$name == "a1"] <- it$a[j]
    sv$value[sv$item == item & sv$name == "d"] <- -it$a[j] * it$b[j]
    sv$value[sv$item == item & sv$name == "g"] <- it$c[j]
    if (model == "4PL") sv$value[sv$item == item & sv$name == "u"] <- it$d[j]
  }
  est <- sv$est
  pname <- paste0(sv$name[est], ".", sv$item[est])
  logit_par <- sv$name[est] %in% c("g", "u")
  internal <- sv$value[est]
  internal[logit_par] <- qlogis(internal[logit_par])
  add(case, "parameter_internal", pname, "", internal)

  fixed_ll <- function(p, ...) {
    v <- sv
    p[logit_par] <- plogis(p[logit_par])
    v$value[est] <- p
    as.numeric(logLik(mirt(dat, 1, itemtype = model, pars = v, TOL = NaN,
                           technical = tech, verbose = FALSE)))
  }
  add(case, "loglik_gh21", "", "", fixed_ll(internal))
  for (type in c("Richardson", "central"))
    add(case, paste0("gradient_internal_", type), pname, "",
        numerical_deriv(internal, fixed_ll, gradient = TRUE, type = type))
  # Hessian of mirt's own GH-21 logLik (documented numerical_deriv; works for
  # indefinite matrices too). Two difference schemes give a numerical error bound.
  add_matrix <- function(quantity, H)
    for (i in seq_len(nrow(H))) for (k in seq_len(ncol(H)))
      add(case, quantity, pname[i], pname[k], H[i, k])
  add_matrix("hessian_internal_Richardson",
             numerical_deriv(internal, fixed_ll, gradient = FALSE, type = "Richardson"))
  add_matrix("hessian_internal_central",
             numerical_deriv(internal, fixed_ll, gradient = FALSE, type = "central", delta = 1e-4))

  # mirt's default grid: 61 equally spaced nodes on [-6, 6], prior dnorm/sum.
  # Here logLik and the SE.type numerical Hessian use the same weights.
  vdef <- mirt(dat, 1, itemtype = model, pars = "values")
  vdef$value <- sv$value
  default_ll <- function(p, ...) {
    v <- vdef
    p[logit_par] <- plogis(p[logit_par])
    v$value[est] <- p
    as.numeric(logLik(mirt(dat, 1, itemtype = model, pars = v, TOL = NaN, verbose = FALSE)))
  }
  add(case, "loglik_default61", "", "", default_ll(internal))
  add(case, "gradient_internal_default61", pname, "",
      numerical_deriv(internal, default_ll, gradient = TRUE, type = "Richardson"))

  # SE.type numerical Hessian -> vcov (mirt refuses non-invertible information).
  # NOTE (mirt 1.48 source, R/utils.R::BL.LL): for the Gaussian density this
  # Hessian re-weights nodes by dnorm(theta)/sum and does not use customPriorFun,
  # so with a custom GH grid it differs from the logLik quadrature. The
  # "customgh" run is kept only to document that behaviour.
  for (grid in c("default61", "customgh")) for (type in c("Richardson", "central")) {
    res <- tryCatch({
      m <- if (grid == "default61")
        mirt(dat, 1, itemtype = model, pars = vdef, TOL = NaN, SE = TRUE,
             SE.type = type, verbose = FALSE)
      else mirt(dat, 1, itemtype = model, pars = sv, TOL = NaN, SE = TRUE,
                SE.type = type, technical = tech, verbose = FALSE)
      V <- vcov(m)
      if (ncol(V) != length(pname)) stop("vcov not available (information not invertible / not PD)")
      add(case, paste0("vcov_colnames_", grid, "_", type), pname, colnames(V), NA_real_)
      V
    }, error = function(e) e)
    if (inherits(res, "error") || any(!is.finite(res))) {
      add(case, paste0("vcov_status_", grid, "_", type),
          if (inherits(res, "error")) conditionMessage(res) else "non-finite", "", NA_real_)
    } else add_matrix(paste0("vcov_internal_", grid, "_", type), res)
  }

  m0 <- mirt(dat, 1, itemtype = model, pars = sv, TOL = NaN, technical = tech, verbose = FALSE)
  th <- matrix(seq(-3, 3, by = 1))
  for (j in seq_len(nrow(it))) {
    P <- probtrace(extract.item(m0, j), th)[, 2]
    for (q in seq_along(P)) add(case, "probtrace", it$Madde[j], as.character(th[q]), P[q])
  }
}
res <- do.call(rbind, out)
write.csv(res, file.path(dir, "mirt_results.csv"), row.names = FALSE)
writeLines(c(R.version.string,
             paste("mirt", as.character(packageVersion("mirt"))),
             paste("SimDesign", as.character(packageVersion("SimDesign"))),
             paste("Deriv", as.character(packageVersion("Deriv"))),
             paste("Rcpp", as.character(packageVersion("Rcpp"))),
             paste("RcppArmadillo", as.character(packageVersion("RcppArmadillo"))),
             paste("mirt source:", Sys.getenv("MIRT_SOURCE", "CRAN/unspecified")),
             paste("generated:", format(Sys.time(), tz = "UTC", usetz = TRUE))),
           file.path(dir, "mirt_session.txt"))
cat("Wrote", nrow(res), "rows to", file.path(dir, "mirt_results.csv"), "\n")

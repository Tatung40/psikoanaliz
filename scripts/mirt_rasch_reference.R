# R mirt reference for the dichotomous Rasch module (B8).
#
# Usage (repository root): Rscript scripts/mirt_rasch_reference.R [fixture_dir]
# Reads tests/fixtures/rasch/{cases,gh41,data_*}.csv and writes mirt_rasch_results.csv
# and mirt_rasch_session.txt.
#
# Matched estimation conditions (core.rasch.fit_rasch):
#  * Rasch with latent mean 0 and free latent SD is fitted as the equivalent
#    2PL with all slopes constrained equal and theta ~ N(0, 1) fixed:
#    logistic(a * z + d_i) with a = sigma, d_i = -b_i, z = theta / sigma.
#    (itemtype = 'Rasch' frees COV_11, but a fixed customPriorFun would hold the
#    latent density fixed, so that route is not equivalent on a custom grid.)
#  * identical 41-node Gauss-Hermite grid via technical customTheta/customPriorFun;
#  * missing responses (NA) are skipped in the likelihood (mirt default).
# Documented interfaces only: mirt(pars=, constrain=, TOL=, technical=),
# mod2values, logLik, numerical_deriv, fscores(method = 'WLE'),
# itemfit(fit_stats = 'infit', Theta = ), personfit(Theta = ).
# An itemtype = 'Rasch' fit on mirt's default grid is recorded for information only.
suppressPackageStartupMessages(library(mirt))

args <- commandArgs(trailingOnly = TRUE)
dir <- if (length(args)) args[1] else "tests/fixtures/rasch"
cases <- read.csv(file.path(dir, "cases.csv"), stringsAsFactors = FALSE)
gh <- read.csv(file.path(dir, "gh41.csv"))
tech <- list(customTheta = matrix(gh$theta),
             customPriorFun = function(Theta, Etable) gh$weight,
             warn = FALSE, message = FALSE)
out <- list()
add <- function(case, quantity, row, col, value)
  out[[length(out) + 1L]] <<- data.frame(case = case, quantity = quantity, row = row,
                                         col = col, value = value)

for (case in cases$case) {
  dat <- read.csv(file.path(dir, paste0("data_", case, ".csv")), na.strings = "NA")
  items <- colnames(dat)
  sv <- mirt(dat, 1, itemtype = "2PL", pars = "values", technical = tech)
  slopes <- sv$parnum[sv$name == "a1"]
  add(case, "prior_types", paste(unique(sv$prior.type), collapse = ";"), "", NA_real_)
  fit <- mirt(dat, 1, itemtype = "2PL", constrain = list(slopes), TOL = 1e-9,
              technical = c(tech, list(NCYCLES = 20000)), verbose = FALSE)
  mv <- mod2values(fit)
  a <- mv$value[mv$name == "a1"][1]
  d <- mv$value[mv$name == "d"]
  add(case, "converged", "", "", as.numeric(extract.mirt(fit, "converged")))
  add(case, "iterations", "", "", extract.mirt(fit, "iterations"))
  add(case, "loglik", "", "", as.numeric(logLik(fit)))
  add(case, "slope", "", "", a)
  for (j in seq_along(items)) add(case, "intercept", items[j], "", d[j])

  # Hessian of mirt's own logLik in (a, d_1..d_k) at the mirt solution.
  fixed_ll <- function(p, ...) {
    v <- sv
    v$value[v$name == "a1"] <- p[1]
    v$value[v$name == "d"] <- p[-1]
    as.numeric(logLik(mirt(dat, 1, itemtype = "2PL", pars = v, TOL = NaN,
                           technical = tech, verbose = FALSE)))
  }
  par <- c(a, d)
  g <- numerical_deriv(par, fixed_ll, gradient = TRUE, type = "Richardson")
  H <- numerical_deriv(par, fixed_ll, gradient = FALSE, type = "Richardson")
  names <- c("a", paste0("d.", items))
  for (i in seq_along(g)) add(case, "gradient", names[i], "", g[i])
  for (i in seq_along(g)) for (k in seq_along(g)) add(case, "hessian", names[i], names[k], H[i, k])

  # Person WLE and Infit/Outfit at the mirt solution, using mirt's own WLE as Theta.
  wle <- fscores(fit, method = "WLE", full.scores = TRUE, full.scores.SE = TRUE)
  for (n in seq_len(nrow(wle))) {
    add(case, "wle", as.character(n), "z", wle[n, 1])
    add(case, "wle", as.character(n), "se_z", wle[n, 2])
  }
  th <- matrix(wle[, 1])
  ifit <- itemfit(fit, fit_stats = "infit", Theta = th)
  for (j in seq_len(nrow(ifit))) for (s in c("outfit", "z.outfit", "infit", "z.infit"))
    add(case, "itemfit", as.character(ifit$item[j]), s, ifit[j, s])
  pfit <- personfit(fit, Theta = th)
  for (n in seq_len(nrow(pfit))) for (s in c("outfit", "z.outfit", "infit", "z.infit"))
    add(case, "personfit", as.character(n), s, pfit[n, s])

  # Information only: itemtype = 'Rasch' on mirt's default grid (61 nodes on [-6, 6]).
  ras <- mirt(dat, 1, itemtype = "Rasch", TOL = 1e-9, technical = list(NCYCLES = 20000),
              verbose = FALSE)
  rv <- mod2values(ras)
  add(case, "rasch_default_loglik", "", "", as.numeric(logLik(ras)))
  add(case, "rasch_default_sd", "", "", sqrt(rv$value[rv$name == "COV_11"]))
  for (j in seq_along(items))
    add(case, "rasch_default_intercept", items[j], "", rv$value[rv$name == "d"][j])
}
res <- do.call(rbind, out)
write.csv(res, file.path(dir, "mirt_rasch_results.csv"), row.names = FALSE)
writeLines(c(R.version.string, paste("mirt", as.character(packageVersion("mirt"))),
             paste("mirt source:", Sys.getenv("MIRT_SOURCE", "CRAN/unspecified")),
             "EM TOL 1e-9, NCYCLES 20000, GH-41 custom grid; WLE via fscores(method='WLE')",
             paste("generated:", format(Sys.time(), tz = "UTC", usetz = TRUE))),
           file.path(dir, "mirt_rasch_session.txt"))
cat("Wrote", nrow(res), "rows\n")

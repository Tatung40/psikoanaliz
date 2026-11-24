# G1 reference fits with R mirt 1.48 (PROTOCOL.json). Usage: Rscript reference.R manifest.json out_dir
# 1PL fixed_a: 2PL with a1 fixed at 1; 1PL common_a: 2PL with all a1 constrained equal; 2PL: itemtype 2PL (ML).
# 3PL/4PL: MAP with createItem() in the IRT parameterization; Python mirt default priors in the M-step (amendment A3).
suppressPackageStartupMessages({library(mirt); library(jsonlite)})
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2, as.character(packageVersion("mirt")) == "1.48")
cases <- fromJSON(args[1], simplifyVector = FALSE); out <- args[2]
clip <- function(p) pmin(pmax(p, 1e-12), 1 - 1e-12)
P4 <- function(par, Theta, ncat) { p <- clip(par[3] + (par[4] - par[3]) / (1 + exp(-par[1] * (Theta[, 1] - par[2])))); cbind(1 - p, p) }
P3 <- function(par, Theta, ncat) { p <- clip(par[3] + (1 - par[3]) / (1 + exp(-par[1] * (Theta[, 1] - par[2])))); cbind(1 - p, p) }
eps <- 1e-6
# Amendment A3: the registered priors enter the M-step objective of each custom item (parprior is not used by R mirt's
# M-step for createItem items). Objective per item: sum(expected counts * log P) + log prior, gradient/Hessian by numDeriv.
lp <- function(par) {
  out <- dlnorm(par[1], 0, .5, log = TRUE) + dnorm(par[2], 0, 2, log = TRUE) + dbeta(par[3], 2, 8, log = TRUE)
  if (length(par) == 4) out <- out + dbeta(par[4], 8, 2, log = TRUE)
  if (is.finite(out)) out else -1e10
}
make <- function(name, par, P, lb, ub) {
  obj <- function(p, x, Theta) sum(x@dat * log(P(p, Theta))) + lp(p)
  createItem(name, par = par, est = rep(TRUE, length(par)), P = P, lbound = lb, ubound = ub,
             gr = function(x, Theta) numDeriv::grad(function(p) obj(p, x, Theta), x@par),
             hss = function(x, Theta) numDeriv::hessian(function(p) obj(p, x, Theta), x@par))
}
items <- list(
  irt3 = make("irt3", c(a = 1, b = 0, c = .15), P3, c(eps, -Inf, eps), c(Inf, Inf, 1 - eps)),
  irt4 = make("irt4", c(a = 1, b = 0, c = .15, d = .95), P4, c(eps, -Inf, eps, eps), c(Inf, Inf, 1 - eps, 1 - eps)))
settings <- list(quadpts = 101, theta_lim = c(-6, 6), TOL = 1e-8, verbose = FALSE, SE = TRUE, SE.type = "Oakes",
                 accelerate = "squarem", technical = list(NCYCLES = 5000))   # SQUAREM: speed only, same objective
fit_with <- function(args) {
  setTimeLimit(elapsed = 300, transient = TRUE); on.exit(setTimeLimit(elapsed = Inf), add = TRUE)   # per-fit budget
  fit <- tryCatch(do.call(mirt, c(args, settings)), error = function(e) { message("fit error: ", conditionMessage(e)); NULL })
  se_type <- "Oakes"
  if (is.null(fit) || !isTRUE(extract.mirt(fit, "secondordertest"))) {   # pre-registered fallback, recorded
    s2 <- settings; s2$SE.type <- "Richardson"; fit <- do.call(mirt, c(args, s2)); se_type <- "Richardson"
  }
  list(fit = fit, se_type = se_type)
}
for (case in cases) {
  t0 <- Sys.time()
  dat <- read.csv(case$path); k <- ncol(dat); model <- case$model
  ok <- tryCatch({
  if (model %in% c("1PL", "2PL")) {
    sv <- mirt(dat, 1, itemtype = "2PL", pars = "values")
    extra <- list()
    if (identical(case$variant, "fixed_a")) { sv$value[sv$name == "a1"] <- 1; sv$est[sv$name == "a1"] <- FALSE }
    if (identical(case$variant, "common_a")) extra$constrain <- list(sv$parnum[sv$name == "a1"])
    res <- fit_with(c(list(data = dat, model = 1, itemtype = "2PL", pars = sv), extra))
    cf <- coef(res$fit, IRTpars = TRUE, printSE = TRUE)[seq_len(k)]
    get <- function(row, p) unname(sapply(cf, function(m) if (p %in% colnames(m) && row %in% rownames(m)) m[row, p] else NA_real_))
    par <- list(a = get("par", "a"), b = get("par", "b")); se <- list(a = get("SE", "a"), b = get("SE", "b"))
  } else {
    type <- if (model == "3PL") "irt3" else "irt4"; qs <- if (model == "3PL") c("a", "b", "c") else c("a", "b", "c", "d")
    sv <- mirt(dat, 1, itemtype = type, customItems = items[type], pars = "values")
    sv$value[sv$name == "b"] <- -qlogis(pmin(pmax(colMeans(dat), .02), .98))
    res <- fit_with(list(data = dat, model = 1, itemtype = type, customItems = items[type], pars = sv))
    cf <- coef(res$fit, printSE = TRUE)[seq_len(k)]
    get <- function(row, p) unname(sapply(cf, function(m) if (row %in% rownames(m)) m[row, p] else NA_real_))
    par <- setNames(lapply(qs, function(p) get("par", p)), qs); se <- setNames(lapply(qs, function(p) get("SE", p)), qs)
  }
  result <- list(id = case$id, converged = extract.mirt(res$fit, "converged"), iterations = extract.mirt(res$fit, "iterations"),
                 loglik = extract.mirt(res$fit, "logLik"), logPrior = tryCatch(extract.mirt(res$fit, "logPrior"), error = function(e) NA),
                 se_type = res$se_type, par = par, se = se)
  write_json(result, file.path(out, paste0(case$id, ".R.json")), digits = NA, auto_unbox = TRUE, na = "null")
  cat(sprintf("%s converged=%s iterations=%d logLik=%.8f se=%s seconds=%.0f\n", case$id, result$converged, result$iterations,
              result$loglik, res$se_type, as.numeric(difftime(Sys.time(), t0, units = "secs"))))
  TRUE }, error = function(e) { cat(sprintf("%s R FAILED: %s\n", case$id, conditionMessage(e))); FALSE })
}
write_json(list(R = R.version.string, R_mirt = as.character(packageVersion("mirt"))), file.path(out, "versions.json"), auto_unbox = TRUE)

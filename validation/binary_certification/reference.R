# G1 reference fits with R mirt 1.48 (PROTOCOL.json). Usage: Rscript reference.R manifest.json out_dir
# 1PL fixed_a: 2PL with a1 fixed at 1; 1PL common_a: 2PL with all a1 constrained equal; 2PL: itemtype 2PL (ML).
# 3PL/4PL: MAP with createItem() in the IRT parameterization and the Python mirt default priors via parprior.
suppressPackageStartupMessages({library(mirt); library(jsonlite)})
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2, as.character(packageVersion("mirt")) == "1.48")
cases <- fromJSON(args[1], simplifyVector = FALSE); out <- args[2]
clip <- function(p) pmin(pmax(p, 1e-12), 1 - 1e-12)
P4 <- function(par, Theta, ncat) { p <- clip(par[3] + (par[4] - par[3]) / (1 + exp(-par[1] * (Theta[, 1] - par[2])))); cbind(1 - p, p) }
P3 <- function(par, Theta, ncat) { p <- clip(par[3] + (1 - par[3]) / (1 + exp(-par[1] * (Theta[, 1] - par[2])))); cbind(1 - p, p) }
eps <- 1e-6
items <- list(
  irt3 = createItem("irt3", par = c(a = 1, b = 0, c = .15), est = rep(TRUE, 3), P = P3,
                    lbound = c(eps, -Inf, eps), ubound = c(Inf, Inf, 1 - eps)),
  irt4 = createItem("irt4", par = c(a = 1, b = 0, c = .15, d = .95), est = rep(TRUE, 4), P = P4,
                    lbound = c(eps, -Inf, eps, eps), ubound = c(Inf, Inf, 1 - eps, 1 - eps)))
settings <- list(quadpts = 101, theta_lim = c(-6, 6), TOL = 1e-8, verbose = FALSE, SE = TRUE, SE.type = "Oakes",
                 technical = list(NCYCLES = 5000))
fit_with <- function(args) {
  fit <- tryCatch(do.call(mirt, c(args, settings)), error = function(e) NULL)
  se_type <- "Oakes"
  if (is.null(fit) || !isTRUE(extract.mirt(fit, "secondordertest"))) {   # pre-registered fallback, recorded
    s2 <- settings; s2$SE.type <- "Richardson"; fit <- do.call(mirt, c(args, s2)); se_type <- "Richardson"
  }
  list(fit = fit, se_type = se_type)
}
for (case in cases) {
  dat <- read.csv(case$path); k <- ncol(dat); model <- case$model
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
    pn <- function(nm) sv$parnum[sv$name == nm]
    prior <- list(c(pn("a"), "lnorm", 0, .5), c(pn("b"), "norm", 0, 2), c(pn("c"), "beta", 2, 8))
    if (model == "4PL") prior[[4]] <- c(pn("d"), "beta", 8, 2)
    res <- fit_with(list(data = dat, model = 1, itemtype = type, customItems = items[type], pars = sv, parprior = prior))
    cf <- coef(res$fit, printSE = TRUE)[seq_len(k)]
    get <- function(row, p) unname(sapply(cf, function(m) if (row %in% rownames(m)) m[row, p] else NA_real_))
    par <- setNames(lapply(qs, function(p) get("par", p)), qs); se <- setNames(lapply(qs, function(p) get("SE", p)), qs)
  }
  result <- list(id = case$id, converged = extract.mirt(res$fit, "converged"), iterations = extract.mirt(res$fit, "iterations"),
                 loglik = extract.mirt(res$fit, "logLik"), logPrior = tryCatch(extract.mirt(res$fit, "logPrior"), error = function(e) NA),
                 se_type = res$se_type, par = par, se = se)
  write_json(result, file.path(out, paste0(case$id, ".R.json")), digits = NA, auto_unbox = TRUE, na = "null")
  cat(sprintf("%s converged=%s iterations=%d logLik=%.8f se=%s\n", case$id, result$converged, result$iterations, result$loglik, res$se_type))
}
write_json(list(R = R.version.string, R_mirt = as.character(packageVersion("mirt"))), file.path(out, "versions.json"), auto_unbox = TRUE)

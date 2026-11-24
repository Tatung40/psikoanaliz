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
# M-step for createItem items). Objective per item: sum(expected counts * log P) + log prior.
# Analytic gradient/Hessian of (expected complete-data item log-likelihood + log prior); verified against central
# differences (relative error <= 2e-7) in validation/binary_certification (Python mirror of these lines).
derivs <- function(p, x, Theta) {
  four <- length(p) == 4; a <- p[1]; b <- p[2]; c <- p[3]; d <- if (four) p[4] else 1
  th <- Theta[, 1]; r0 <- x@dat[, 1]; r1 <- x@dat[, 2]
  s <- plogis(a * (th - b)); u <- s * (1 - s); w <- u * (1 - 2 * s); t <- th - b; P <- c + (d - c) * s
  g <- r1 / P - r0 / (1 - P); h <- -r1 / P^2 - r0 / (1 - P)^2
  D <- list((d - c) * u * t, -(d - c) * u * a, 1 - s); if (four) D[[4]] <- s
  m <- length(D); zero <- 0 * th; D2 <- rep(list(rep(list(zero), m)), m)
  D2[[1]][[1]] <- (d - c) * w * t^2; D2[[1]][[2]] <- D2[[2]][[1]] <- (d - c) * (-w * a * t - u); D2[[2]][[2]] <- (d - c) * w * a^2
  D2[[1]][[3]] <- D2[[3]][[1]] <- -u * t; D2[[2]][[3]] <- D2[[3]][[2]] <- u * a
  if (four) { D2[[1]][[4]] <- D2[[4]][[1]] <- u * t; D2[[2]][[4]] <- D2[[4]][[2]] <- -u * a }
  gp <- c(-1 / a - log(a) / (.25 * a), -b / 4, 1 / c - 7 / (1 - c)); if (four) gp <- c(gp, 7 / d - 1 / (1 - d))
  hp <- c((log(a) - 1) / (.25 * a^2) + 1 / a^2, -1 / 4, -1 / c^2 - 7 / (1 - c)^2); if (four) hp <- c(hp, -7 / d^2 - 1 / (1 - d)^2)
  G <- sapply(D, function(Di) sum(g * Di)) + gp
  H <- matrix(0, m, m); for (i in 1:m) for (j in 1:m) H[i, j] <- sum(h * D[[i]] * D[[j]] + g * D2[[i]][[j]])
  list(G = G, H = H + diag(hp))
}
make <- function(name, par, P, lb, ub)
  createItem(name, par = par, est = rep(TRUE, length(par)), P = P, lbound = lb, ubound = ub,
             gr = function(x, Theta) derivs(x@par, x, Theta)$G, hss = function(x, Theta) derivs(x@par, x, Theta)$H)
items <- list(
  # Amendment A4: the Python mirt library box (mirt.estimation.base._ITEM_PARAMETER_BOUNDS).
  irt3 = make("irt3", c(a = 1, b = 0, c = .15), P3, c(.1, -6, eps), c(5, 6, .5)),
  irt4 = make("irt4", c(a = 1, b = 0, c = .15, d = .95), P4, c(.1, -6, eps, .5), c(5, 6, .5, 1 - eps)))
settings <- list(quadpts = 101, theta_lim = c(-6, 6), TOL = 1e-8, verbose = FALSE, SE = TRUE, SE.type = "Oakes",
                 accelerate = "squarem", technical = list(NCYCLES = 5000))   # SQUAREM: speed only, same objective
fit_with <- function(args) {
  setTimeLimit(elapsed = 240, transient = TRUE); on.exit(setTimeLimit(elapsed = Inf), add = TRUE)   # per-fit budget
  use <- settings; use[names(args)[names(args) %in% names(use)]] <- NULL
  fit <- tryCatch(do.call(mirt, c(args, use)), error = function(e) { message("fit error: ", conditionMessage(e)); NULL })
  se_type <- if (isFALSE(args$SE)) "none (MAP: descriptive only, A5 addendum)" else "Oakes"
  if (!isFALSE(args$SE) && (is.null(fit) || !isTRUE(extract.mirt(fit, "secondordertest")))) {   # pre-registered fallback, recorded
    s2 <- use; s2$SE.type <- "Richardson"; fit <- do.call(mirt, c(args, s2)); se_type <- "Richardson"
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
    # Custom items: Newton-Raphson M-step with the supplied (analytic) Hessian; the default quasi-Newton
    # M-step stopped EM after 8-14 cycles below the posterior maximum (run 4ce2ee1). Objective and TOL unchanged.
    run <- function(start) fit_with(list(data = dat, model = 1, itemtype = type, customItems = items[type], pars = start,
                                         optimizer = "NR", accelerate = "squarem", SE = FALSE))   # A5 addendum
    logpost <- function(res) {          # R-computed log-posterior at R's own estimate (amendment A5)
      cf <- coef(res$fit)[seq_len(k)]
      extract.mirt(res$fit, "logLik") + sum(sapply(cf, function(m) { p <- m["par", ]
        dlnorm(p[["a"]], 0, .5, log = TRUE) + dnorm(p[["b"]], 0, 2, log = TRUE) + dbeta(p[["c"]], 2, 8, log = TRUE) +
          (if ("d" %in% names(p)) dbeta(p[["d"]], 8, 2, log = TRUE) else 0) }))
    }
    # Start 2 (amendment A5): R's own nested-model fit on the same data; Python estimates are never used.
    sv2 <- sv
    if (model == "4PL") {
      nested <- tryCatch({
        s3 <- mirt(dat, 1, itemtype = "irt3", customItems = items["irt3"], pars = "values")
        s3$value[s3$name == "b"] <- -qlogis(pmin(pmax(colMeans(dat), .02), .98))
        coef(fit_with(list(data = dat, model = 1, itemtype = "irt3", customItems = items["irt3"], pars = s3,
                           optimizer = "NR", accelerate = "squarem", SE = FALSE))$fit)[seq_len(k)] }, error = function(e) NULL)
      if (!is.null(nested)) for (q in c("a", "b", "c"))
        sv2$value[sv2$name == q] <- pmin(pmax(sapply(nested, function(m) m["par", q]), c(a = .1, b = -6, c = 1e-6)[[q]]), c(a = 5, b = 6, c = .5)[[q]])
      sv2$value[sv2$name == "d"] <- .95
    } else {
      nested <- tryCatch(coef(mirt(dat, 1, itemtype = "2PL", verbose = FALSE), IRTpars = TRUE)[seq_len(k)], error = function(e) NULL)
      if (!is.null(nested)) for (q in c("a", "b"))
        sv2$value[sv2$name == q] <- pmin(pmax(sapply(nested, function(m) m[1, q]), c(a = .1, b = -6)[[q]]), c(a = 5, b = 6)[[q]])
      sv2$value[sv2$name == "c"] <- .15
    }
    safe <- function(start) tryCatch(run(start), error = function(e) { message("start discarded: ", conditionMessage(e)); NULL })
    fits <- Filter(function(f) !is.null(f) && !is.null(f$fit), list(default = safe(sv), nested = safe(sv2)))   # A5 addendum 2
    if (!length(fits)) stop("no start finished")
    lps <- sapply(fits, logpost); best <- names(which.max(lps))
    res <- fits[[best]]; res$starts <- as.list(lps); res$chosen <- best
    cf <- coef(res$fit)[seq_len(k)]
    get <- function(row, p) unname(sapply(cf, function(m) if (row %in% rownames(m)) m[row, p] else NA_real_))
    par <- setNames(lapply(qs, function(p) get("par", p)), qs); se <- setNames(lapply(qs, function(p) get("SE", p)), qs)
  }
  result <- list(id = case$id, converged = extract.mirt(res$fit, "converged"), iterations = extract.mirt(res$fit, "iterations"),
                 loglik = extract.mirt(res$fit, "logLik"), logPrior = tryCatch(extract.mirt(res$fit, "logPrior"), error = function(e) NA),
                 se_type = res$se_type, par = par, se = se, starts_logpost = res$starts, chosen_start = res$chosen)
  write_json(result, file.path(out, paste0(case$id, ".R.json")), digits = NA, auto_unbox = TRUE, na = "null")
  cat(sprintf("%s converged=%s iterations=%d logLik=%.8f se=%s seconds=%.0f\n", case$id, result$converged, result$iterations,
              result$loglik, res$se_type, as.numeric(difftime(Sys.time(), t0, units = "secs"))))
  TRUE }, error = function(e) { cat(sprintf("%s R FAILED: %s\n", case$id, conditionMessage(e))); FALSE })
}
write_json(list(R = R.version.string, R_mirt = as.character(packageVersion("mirt"))), file.path(out, "versions.json"), auto_unbox = TRUE)

# Optional native R mirt 1.48 backend (fixed, whitelisted specifications only).
# Never evaluates user-supplied R code: all inputs are validated JSON/CSV produced by core/mirt_r_backend.py.
args <- commandArgs(TRUE); stopifnot(length(args) == 1)
suppressPackageStartupMessages({library(mirt); library(jsonlite)})
stopifnot(as.character(packageVersion("mirt")) == "1.48")
root <- args[1]; set.seed(20261127)
cfg <- fromJSON(file.path(root, "request.json"), simplifyVector = TRUE)
dat <- read.csv(file.path(root, "data.csv"), check.names = FALSE)
trace <- as.matrix(read.csv(file.path(root, "trace.csv")))
J <- ncol(dat); spec <- cfg$spec
stopifnot(spec %in% c("M3PL_fixed", "M4PL_fixed", "bfactor"))
started <- proc.time()[["elapsed"]]
emit <- function(fit, values, S, nodes, extra = list()) {
  Q <- ncol(S); ip <- list(); tp <- list(); info <- list()
  for (i in seq_len(J)) {
    it <- extract.item(fit, i)
    if (!is.null(nodes)) ip[[i]] <- probtrace(it, nodes)
    tp[[i]] <- probtrace(it, trace)
    m <- iteminfo(it, trace, multidim_matrix = TRUE)
    info[[i]] <- lapply(m, function(x) as.numeric(x))
  }
  out <- c(list(engine = "R mirt", mirt_version = as.character(packageVersion("mirt")), R = R.version.string,
    spec = spec, converged = extract.mirt(fit, "converged"), iterations = extract.mirt(fit, "iterations"),
    logLik = extract.mirt(fit, "logLik"), n_parameters = extract.mirt(fit, "nest"),
    AIC = extract.mirt(fit, "AIC"), BIC = extract.mirt(fit, "BIC"), covariance = S,
    values = values[, c("item", "class", "name", "value", "lbound", "ubound", "est")],
    integration_probabilities = if (is.null(nodes)) NULL else ip, trace_probabilities = tp, trace_information = info,
    seconds = proc.time()[["elapsed"]] - started), extra)
  write_json(out, file.path(root, "result.json"), auto_unbox = TRUE, digits = NA, matrix = "rowmajor")
}
if (spec == "bfactor") {
  type <- cfg$itemtype; stopifnot(type %in% c("2PL", "graded", "gpcm"))
  fit <- bfactor(dat, as.integer(cfg$groups), itemtype = type, quadpts = as.integer(cfg$n_quadpts), SE = FALSE,
    TOL = cfg$tol, verbose = FALSE, technical = list(NCYCLES = as.integer(cfg$max_iter)))
  values <- mod2values(fit); Q <- 1 + length(unique(cfg$groups))
  emit(fit, values, diag(Q), NULL, list(integration = "R bfactor dimension reduction, quadpts per dimension",
    quadpts = as.integer(cfg$n_quadpts)))
  quit(save = "no")
}
pattern <- as.matrix(cfg$pattern); Q <- ncol(pattern); stopifnot(nrow(pattern) == J, Q >= 2)
grid <- read.csv(file.path(root, "integration.csv")); nodes <- as.matrix(grid[, seq_len(Q)])
pairs <- do.call(rbind, lapply(2:Q, function(r) cbind(r, seq_len(r - 1))))
covariance <- function(eta) {L <- diag(Q); L[pairs] <- eta; S <- tcrossprod(L); S / outer(sqrt(diag(S)), sqrt(diag(S)))}
den <- function(obj, Theta) {
  S <- covariance(obj@par); C <- chol(S); inv <- chol2inv(C)
  exp(log(grid$weight) - sum(log(diag(C))) - .5 * rowSums((Theta %*% inv - Theta) * Theta))
}
eta <- rep(0, nrow(pairs)); names(eta) <- paste0("eta", pairs[, 1], pairs[, 2])
group <- createGroup(par = eta, est = rep(TRUE, length(eta)), den = den, nfact = Q, standardize = FALSE)
syntax <- paste(vapply(seq_len(Q), function(j) paste0("F", j, " = ", paste(which(pattern[, j] != 0), collapse = ",")), ""), collapse = "\n")
family <- if (spec == "M3PL_fixed") "3PL" else "4PL"
base <- list(data = dat, model = mirt.model(syntax), itemtype = family, customGroup = group, verbose = FALSE)
sv <- do.call(mirt, c(base, list(pars = "values")))
for (i in seq_len(J)) {
  idx <- sv$item == colnames(dat)[i]; ai <- which(idx & grepl("^a[0-9]+$", sv$name)); stopifnot(length(ai) == Q)
  sv$value[ai] <- pattern[i, ] * 1.2; sv$est[ai] <- pattern[i, ] != 0
  sv$lbound[ai] <- ifelse(pattern[i, ] != 0, .1, 0); sv$ubound[ai] <- ifelse(pattern[i, ] != 0, 5, 0)
  sv$value[idx & sv$name == "d"] <- 0; sv$lbound[idx & sv$name == "d"] <- -6; sv$ubound[idx & sv$name == "d"] <- 6
  sv$value[idx & sv$name == "g"] <- cfg$c[i]; sv$est[idx & sv$name == "g"] <- FALSE
  sv$value[idx & sv$name == "u"] <- cfg$u[i]; sv$est[idx & sv$name == "u"] <- FALSE
}
sv$lbound[grepl("^eta", sv$name)] <- -3; sv$ubound[grepl("^eta", sv$name)] <- 3
free <- which(sv$est)
fit <- do.call(mirt, c(base, list(pars = sv, method = "EM", SE = FALSE, accelerate = "none", TOL = cfg$tol,
  technical = list(NCYCLES = as.integer(cfg$max_iter), customTheta = nodes, MAXQUAD = 20000),
  optimizer = "nloptr", nloptr_args = list(lb = sv$lbound[free], ub = sv$ubound[free],
    opts = list(algorithm = "NLOPT_LD_SLSQP", xtol_rel = 1e-8, maxeval = 2000)))))
values <- mod2values(fit); S <- covariance(values$value[grepl("^eta", values$name)])
emit(fit, values, S, nodes, list(integration = "common Gauss-Hermite product grid (customTheta + customGroup density)"))

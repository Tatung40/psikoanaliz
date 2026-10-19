# R references for B11 (3PL/4PL Q3, WLE/EAP, M2, two-group DIF).
#   Rscript scripts/mirt_irt_b11_reference.R 3PL     # Phase A
#   Rscript scripts/mirt_irt_b11_reference.R 4PL     # Phase B
# Run after scripts/irt_b11_inputs_for_r.py. Writes tests/fixtures/irt_b11/r_b11_<model>.csv
# and r_b11_session.txt.
#
# Matched conventions (core.irt / core.irt_mg34): P = c + (d - c) logistic(a (theta - b));
# mirt itemtype '3PL'/'4PL': P = g + (u - g) logistic(a1 theta + d_mirt) with a = a1,
# b = -d_mirt / a1, c = g, d = u. mirt estimates g, u on the logit scale without bounds or
# priors (no prior is used in core.irt either); Python uses box constraints, so estimates
# are only comparable where Python's solution is interior. mirt default quadrature (61
# nodes on [-6, 6]); core.irt_mg34 uses quadrature = "mirt" for these comparisons.
# Documented interfaces only: mirt, multipleGroup(pars, invariance, constrain), mod2values,
# logLik, vcov, fscores(method = 'WLE' / 'EAP'), residuals(type = 'Q3', Theta = ), M2.
suppressPackageStartupMessages(library(mirt))
args <- commandArgs(trailingOnly = TRUE); model <- if (length(args)) args[1] else "3PL"
dir <- "tests/fixtures/irt_b11"
out <- list()
add <- function(case, quantity, row, col, value) {
  value <- suppressWarnings(as.numeric(value)); if (!length(value)) value <- NA_real_   # NULL -> NA
  out[[length(out) + 1L]] <<- data.frame(case = case, quantity = quantity, row = as.character(row),
                                         col = as.character(col), value = value[1])
}
ctrl <- list(NCYCLES = 20000)
pn <- c("a1", "d", "g", "u")
set_python <- function(sv, it) {
  for (j in seq_len(nrow(it))) {
    sv$value[sv$item == it$Madde[j] & sv$name == "a1"] <- it$a[j]
    sv$value[sv$item == it$Madde[j] & sv$name == "d"] <- -it$a[j] * it$b[j]
    sv$value[sv$item == it$Madde[j] & sv$name == "g"] <- it$c[j]
    sv$value[sv$item == it$Madde[j] & sv$name == "u"] <- it$d[j]
  }
  sv$est <- FALSE; sv
}

single <- if (model == "3PL") c("irt3pl_ok", "irt3pl_ld", "irt3pl_missing", "irt3pl_easy") else c("irt4pl_ok", "irt4pl_ld")
for (case in single) {
  dat <- read.csv(file.path(dir, paste0("data_", case, ".csv")), na.strings = "NA")
  fit <- mirt(dat, 1, itemtype = model, TOL = 1e-10, SE = TRUE, SE.type = "Richardson",
              technical = ctrl, verbose = FALSE)
  add(case, "converged", "", "", extract.mirt(fit, "converged"))
  add(case, "loglik", "", "", logLik(fit))
  mv <- mod2values(fit)
  for (r in seq_len(nrow(mv))) if (mv$name[r] %in% pn) add(case, "par", mv$item[r], mv$name[r], mv$value[r])
  fx <- mv; fx$est <- FALSE
  add(case, "loglik_fixed", "", "", logLik(mirt(dat, 1, itemtype = model, pars = fx, TOL = NaN, verbose = FALSE)))
  if (!anyNA(dat)) {
    m2 <- tryCatch(M2(fit, type = "M2*"), error = function(e) NULL)
    if (!is.null(m2)) for (s in c("M2", "df", "p", "RMSEA", "SRMSR")) add(case, "m2", "", s, m2[[s]])
  }
  # Python parameters (fixed): logLik, WLE, EAP, Q3 with Python EAP scores.
  it <- read.csv(file.path(dir, paste0("py_items_", case, ".csv")))
  pe <- read.csv(file.path(dir, paste0("py_persons_", case, ".csv")))
  pv <- set_python(mirt(dat, 1, itemtype = model, pars = "values"), it)
  pfit <- mirt(dat, 1, itemtype = model, pars = pv, TOL = NaN, verbose = FALSE)
  add(case, "loglik_python_par", "", "", logLik(pfit))
  w <- fscores(pfit, method = "WLE", verbose = FALSE)
  e <- fscores(pfit, method = "EAP", verbose = FALSE)
  for (p in seq_len(nrow(w))) { add(case, "wle", p, "", w[p, 1]); add(case, "eap", p, "", e[p, 1]) }
  q3 <- residuals(pfit, type = "Q3", Theta = matrix(pe$eap), verbose = FALSE)
  for (i in seq_len(nrow(q3))) for (j in seq_len(ncol(q3))) if (i < j)
    add(case, "q3", rownames(q3)[i], colnames(q3)[j], q3[i, j])
}

## ---- two groups ----------------------------------------------------------------------------
# 4PL multigroup EM on flat surfaces does not reach TOL = 1e-8 in practical time; Phase B uses
# TOL = 1e-6 and at most 3000 cycles for the two-group fits (recorded; comparisons use wider tolerances).
mg_tol <- if (model == "4PL") 1e-6 else 1e-8
mg_ctrl <- if (model == "4PL") list(NCYCLES = 3000) else ctrl
mg <- function(dat, grp, invariance, constrain = NULL, SE = FALSE)
  multipleGroup(dat, 1, group = grp, itemtype = model, invariance = invariance, constrain = constrain,
                TOL = mg_tol, SE = SE, SE.type = "Richardson", technical = mg_ctrl, verbose = FALSE)
dif_cases <- if (model == "3PL") c("dif3pl", "dif3pl_none") else c("dif4pl", "dif4pl_none")
single_par <- if (model == "3PL") list(b = c("a1", "g"), c = c("a1", "d")) else
  list(b = c("a1", "g", "u"), c = c("a1", "d", "u"), d = c("a1", "d", "g"))
for (case in dif_cases) {
  d <- read.csv(file.path(dir, paste0("data_", case, ".csv")), na.strings = "NA")
  grp <- factor(d$Grup, levels = c("R", "F")); dat <- d[, -1]; items <- colnames(dat)
  lat <- c("free_means", "free_var")
  m0 <- mg(dat, grp, c(items, lat))
  add(case, "base_loglik", "", "", logLik(m0)); add(case, "base_converged", "", "", extract.mirt(m0, "converged"))
  add(case, "mg_tol", "", "", mg_tol)
  sv <- multipleGroup(dat, 1, group = grp, itemtype = model, pars = "values")
  # single-parameter LR references only for 3PL (4PL DIF is descriptive in core.irt_dif34)
  tested <- if (model == "4PL") character(0) else if (case == "dif3pl") c("M1", "M3", "M8", "M10") else c("M1", "M3")
  for (it in items) {
    m_all <- mg(dat, grp, c(setdiff(items, it), lat))
    add(case, "lr_omnibus", it, "", 2 * (logLik(m_all) - logLik(m0)))
    add(case, "omnibus_converged", it, "", extract.mirt(m_all, "converged"))
    if (it %in% tested) for (nm in names(single_par)) {
      # one parameter group-specific: the others constrained equal across groups
      cons <- lapply(single_par[[nm]], function(p) sv$parnum[sv$item == it & sv$name == p])
      m1 <- mg(dat, grp, c(setdiff(items, it), lat), constrain = cons)
      add(case, paste0("lr_", nm), it, "", 2 * (logLik(m1) - logLik(m0)))
    }
  }
  anchors <- read.csv(file.path(dir, paste0("py_dif_anchors_", case, ".csv")))$anchor
  fin <- mg(dat, grp, c(anchors, lat), SE = model == "3PL")
  add(case, "final_converged", "", "", extract.mirt(fin, "converged"))
  add(case, "final_loglik", "", "", logLik(fin))
  mv <- mod2values(fin)
  for (r in seq_len(nrow(mv))) if (mv$name[r] %in% c(pn, "MEAN_1", "COV_11")) {
    add(case, paste0("final_par_", mv$group[r]), mv$item[r], mv$name[r], mv$value[r])
    add(case, paste0("final_parnum_", mv$group[r]), mv$item[r], mv$name[r], mv$parnum[r])
  }
  V <- if (model == "3PL") tryCatch(vcov(fin), error = function(e) NULL) else NULL   # 4PL: no SE requested
  if (!is.null(V) && !is.null(colnames(V))) for (i in seq_len(ncol(V))) for (j in seq_len(ncol(V)))
    add(case, "final_vcov", colnames(V)[i], colnames(V)[j], V[i, j])
  total <- 0
  for (g in c("R", "F")) {
    sub <- dat[grp == g, ]
    s1 <- mirt(sub, 1, itemtype = model, pars = "values")
    for (it in items) for (nm in intersect(pn, unique(s1$name)))
      s1$value[s1$item == it & s1$name == nm] <- mv$value[mv$group == g & mv$item == it & mv$name == nm]
    s1$value[s1$name == "MEAN_1"] <- mv$value[mv$group == g & mv$name == "MEAN_1"]
    s1$value[s1$name == "COV_11"] <- mv$value[mv$group == g & mv$name == "COV_11"]
    s1$est <- FALSE
    total <- total + as.numeric(logLik(mirt(sub, 1, itemtype = model, pars = s1, TOL = NaN, verbose = FALSE)))
  }
  add(case, "final_loglik_fixed_sum", "", "", total)
}

write.csv(do.call(rbind, out), file.path(dir, paste0("r_b11_", model, ".csv")), row.names = FALSE)
writeLines(c(R.version.string, paste("mirt", as.character(packageVersion("mirt"))),
             "mirt source: github philchalmers/mirt a95523595be1d3d1df60b425fab42a1a02ce612d (1.48)",
             paste("generated:", format(Sys.time(), tz = "UTC", usetz = TRUE))),
           file.path(dir, "r_b11_session.txt"))
cat("done\n")

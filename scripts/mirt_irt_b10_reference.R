# R references for B10 (1PL/2PL Q3, WLE/EAP, M2, two-group DIF). Run after
# scripts/irt_b10_inputs_for_r.py from the repository root:
#   Rscript scripts/mirt_irt_b10_reference.R
# Writes tests/fixtures/irt_b10/r_b10_results.csv and r_b10_session.txt.
#
# Model conventions matched to core.irt / core.irt_mg:
#  * 2PL: mirt itemtype = '2PL', slope-intercept logistic(a * theta + d), b = -d / a.
#  * 1PL: the same '2PL' with every a1 fixed at 1 (pars = 'values', est = FALSE), theta ~ N(0, 1)
#    in the (reference) group. This is NOT itemtype = 'Rasch' (which frees the latent variance).
#  * mirt default quadrature (61 nodes on [-6, 6]); core.irt_mg uses quadrature = "mirt".
#  * Two groups: invariance = c(<anchors>, 'free_means', 'free_var'); reference N(0, 1).
# Documented interfaces only: mirt, multipleGroup(pars =, invariance =, constrain =), mod2values,
# logLik, vcov, fscores(method = 'WLE' / 'EAP'), residuals(type = 'Q3', Theta =), M2,
# DIF(scheme = 'drop').
suppressPackageStartupMessages(library(mirt))
dir <- "tests/fixtures/irt_b10"
out <- list()
add <- function(case, quantity, row, col, value)
  out[[length(out) + 1L]] <<- data.frame(case = case, quantity = quantity, row = as.character(row),
                                         col = as.character(col), value = as.numeric(value))
fix_slopes <- function(sv) { sv$value[sv$name == "a1"] <- 1; sv$est[sv$name == "a1"] <- FALSE; sv }
ctrl <- list(NCYCLES = 20000)

## ---- single group: estimation, fixed-parameter scoring, Q3, M2 ---------------------------
single <- c(irt1pl_ok = "1PL", irt2pl_ok = "2PL", irt2pl_ld = "2PL", irt2pl_missing = "2PL")
for (case in names(single)) {
  model <- single[[case]]
  dat <- read.csv(file.path(dir, paste0("data_", case, ".csv")), na.strings = "NA")
  sv <- mirt(dat, 1, itemtype = "2PL", pars = "values")
  if (model == "1PL") sv <- fix_slopes(sv)
  fit <- mirt(dat, 1, itemtype = "2PL", pars = sv, TOL = 1e-10, SE = TRUE, SE.type = "Richardson",
              technical = ctrl, verbose = FALSE)
  add(case, "converged", "", "", extract.mirt(fit, "converged"))
  add(case, "loglik", "", "", logLik(fit))
  mv <- mod2values(fit)
  for (r in seq_len(nrow(mv))) if (mv$name[r] %in% c("a1", "d"))
    add(case, "par", mv$item[r], mv$name[r], mv$value[r])
  V <- vcov(fit)
  for (i in seq_len(ncol(V))) for (j in seq_len(ncol(V))) add(case, "vcov", colnames(V)[i], colnames(V)[j], V[i, j])
  for (r in seq_len(nrow(mv))) if (mv$est[r]) add(case, "parnum", mv$item[r], mv$name[r], mv$parnum[r])
  fx <- mv; fx$est <- FALSE
  add(case, "loglik_fixed", "", "", logLik(mirt(dat, 1, itemtype = "2PL", pars = fx, TOL = NaN, verbose = FALSE)))
  if (!anyNA(dat)) {
    m2 <- M2(fit, type = "M2*")
    for (s in c("M2", "df", "p", "RMSEA", "SRMSR")) add(case, "m2", "", s, m2[[s]])
  }
  # Python parameters and WLE scores (fixed): WLE, EAP and Q3 references.
  it <- read.csv(file.path(dir, paste0("py_items_", case, ".csv")))
  pe <- read.csv(file.path(dir, paste0("py_persons_", case, ".csv")))
  pv <- mirt(dat, 1, itemtype = "2PL", pars = "values")
  for (j in seq_len(nrow(it))) {
    pv$value[pv$item == it$Madde[j] & pv$name == "a1"] <- it$a[j]
    pv$value[pv$item == it$Madde[j] & pv$name == "d"] <- -it$a[j] * it$b[j]
  }
  pv$est <- FALSE
  pfit <- mirt(dat, 1, itemtype = "2PL", pars = pv, TOL = NaN, verbose = FALSE)
  w <- fscores(pfit, method = "WLE", verbose = FALSE)
  e <- fscores(pfit, method = "EAP", verbose = FALSE)
  for (p in seq_len(nrow(w))) { add(case, "wle", p, "", w[p, 1]); add(case, "eap", p, "", e[p, 1]) }
  q3 <- residuals(pfit, type = "Q3", Theta = matrix(pe$theta), verbose = FALSE)
  for (i in seq_len(nrow(q3))) for (j in seq_len(ncol(q3))) if (i < j)
    add(case, "q3", rownames(q3)[i], colnames(q3)[j], q3[i, j])
}

## ---- two groups ----------------------------------------------------------------------------
mg_values <- function(dat, grp, model) {
  sv <- multipleGroup(dat, 1, group = grp, itemtype = "2PL", pars = "values")
  if (model == "1PL") sv <- fix_slopes(sv)
  sv
}
mg_fit <- function(dat, grp, model, invariance, constrain = NULL, SE = FALSE) {
  sv <- mg_values(dat, grp, model)
  multipleGroup(dat, 1, group = grp, itemtype = "2PL", pars = sv, invariance = invariance,
                constrain = constrain, TOL = 1e-8, SE = SE, SE.type = "Richardson",
                technical = ctrl, verbose = FALSE)
}
dif_cases <- c(dif1pl_uniform = "1PL", dif2pl_mixed = "2PL", dif2pl_none = "2PL")
for (case in names(dif_cases)) {
  model <- dif_cases[[case]]
  d <- read.csv(file.path(dir, paste0("data_", case, ".csv")), na.strings = "NA")
  grp <- factor(d$Grup, levels = c("R", "F")); dat <- d[, -1]; items <- colnames(dat)
  lat <- c("free_means", "free_var")
  # Round 0 of the procedure: all other items anchored.
  m0 <- mg_fit(dat, grp, model, c(items, lat))
  add(case, "lr_base_loglik", "", "", logLik(m0)); add(case, "lr_base_converged", "", "", extract.mirt(m0, "converged"))
  sv <- mg_values(dat, grp, model)
  for (it in items) {
    others <- setdiff(items, it)
    if (model == "2PL") {
      m2 <- mg_fit(dat, grp, model, c(others, lat))
      cons <- list(sv$parnum[sv$item == it & sv$name == "a1"])
      m1 <- mg_fit(dat, grp, model, c(others, lat), constrain = cons)
      add(case, "lr_omnibus", it, "", 2 * (logLik(m2) - logLik(m0)))
      add(case, "lr_uniform", it, "", 2 * (logLik(m1) - logLik(m0)))
      add(case, "lr_converged", it, "", extract.mirt(m1, "converged") * extract.mirt(m2, "converged"))
    } else {
      m1 <- mg_fit(dat, grp, model, c(others, lat))
      add(case, "lr_uniform", it, "", 2 * (logLik(m1) - logLik(m0)))
      add(case, "lr_converged", it, "", extract.mirt(m1, "converged"))
    }
  }
  # mirt's own DIF() routine on the same round-0 design: start from the fully constrained
  # model and free the tested parameters of one item at a time (scheme = 'drop').
  wp <- if (model == "2PL") c("a1", "d") else "d"
  dd <- DIF(m0, which.par = wp, scheme = "drop", p.adjust = "none", verbose = FALSE,
            TOL = 1e-8, technical = ctrl)
  for (it in rownames(dd)) add(case, "mirt_DIF_X2", it, "", dd[it, "X2"])
  # Final purified model (Python anchors): parameters, logLik, vcov.
  anchors <- read.csv(file.path(dir, paste0("py_dif_anchors_", case, ".csv")))$anchor
  fin <- mg_fit(dat, grp, model, c(anchors, lat), SE = TRUE)
  add(case, "final_converged", "", "", extract.mirt(fin, "converged"))
  add(case, "final_loglik", "", "", logLik(fin))
  mv <- mod2values(fin)
  for (r in seq_len(nrow(mv))) if (mv$name[r] %in% c("a1", "d", "MEAN_1", "COV_11")) {
    add(case, paste0("final_par_", mv$group[r]), mv$item[r], mv$name[r], mv$value[r])
    add(case, paste0("final_parnum_", mv$group[r]), mv$item[r], mv$name[r], mv$parnum[r])
  }
  V <- vcov(fin)
  for (i in seq_len(ncol(V))) for (j in seq_len(ncol(V))) add(case, "final_vcov", colnames(V)[i], colnames(V)[j], V[i, j])
  # Fixed-parameter logLik: sum of single-group TOL = NaN evaluations at the final parameters.
  total <- 0
  for (g in c("R", "F")) {
    sub <- dat[grp == g, ]
    s1 <- mirt(sub, 1, itemtype = "2PL", pars = "values")
    for (it in items) for (nm in c("a1", "d"))
      s1$value[s1$item == it & s1$name == nm] <- mv$value[mv$group == g & mv$item == it & mv$name == nm]
    s1$value[s1$name == "MEAN_1"] <- mv$value[mv$group == g & mv$name == "MEAN_1"]
    s1$value[s1$name == "COV_11"] <- mv$value[mv$group == g & mv$name == "COV_11"]
    s1$est <- FALSE
    total <- total + as.numeric(logLik(mirt(sub, 1, itemtype = "2PL", pars = s1, TOL = NaN, verbose = FALSE)))
  }
  add(case, "final_loglik_fixed_sum", "", "", total)
}

write.csv(do.call(rbind, out), file.path(dir, "r_b10_results.csv"), row.names = FALSE)
writeLines(c(R.version.string, paste("mirt", as.character(packageVersion("mirt"))),
             "mirt source: github philchalmers/mirt a95523595be1d3d1df60b425fab42a1a02ce612d (1.48)",
             paste("generated:", format(Sys.time(), tz = "UTC", usetz = TRUE))),
           file.path(dir, "r_b10_session.txt"))
cat("done\n")

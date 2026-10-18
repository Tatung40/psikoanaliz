# R references for B9 (Rasch DIF, Q3, M2). Run after scripts/rasch_b9_inputs_for_r.py:
#   Rscript scripts/mirt_rasch_b9_reference.R
# Writes tests/fixtures/rasch_b9/r_b9_results.csv and r_b9_session.txt.
#
# Documented interfaces only:
#  mirt: multipleGroup(itemtype='Rasch', invariance = c(<anchor names>, 'free_means',
#        'free_var'), SE = TRUE, SE.type = 'Richardson'); mod2values; vcov; logLik;
#        mirt(itemtype = 'Rasch'); residuals(type = 'Q3', Theta = ); M2(type = 'M2*').
#  eRm:  RM(); Waldtest(splitcr = ); LRtest(splitcr = )   (CML; agreement indicator only).
# mirt multipleGroup uses its default quadrature (61 nodes on [-6, 6], normal weights
# with group mean/SD); core.rasch_dif reproduces it with quadrature = "mirt".
suppressPackageStartupMessages({library(mirt); library(eRm)})
dir <- "tests/fixtures/rasch_b9"; dir8 <- "tests/fixtures/rasch"
out <- list()
add <- function(case, quantity, row, col, value)
  out[[length(out) + 1L]] <<- data.frame(case = case, quantity = quantity, row = as.character(row),
                                         col = as.character(col), value = value)

## ---- DIF: multipleGroup with the Python-purified anchor set -------------------
for (case in c("dif_known", "dif_none")) {
  d <- read.csv(file.path(dir, paste0("data_", case, ".csv")), na.strings = "NA")
  grp <- factor(d$Grup, levels = c("R", "F")); dat <- d[, -1]
  anchors <- read.csv(file.path(dir, paste0("py_dif_anchors_", case, ".csv")))$anchor
  mg <- multipleGroup(dat, 1, group = grp, itemtype = "Rasch", verbose = FALSE,
                      invariance = c(anchors, "free_means", "free_var"), TOL = 1e-9,
                      SE = TRUE, SE.type = "Richardson", technical = list(NCYCLES = 20000))
  add(case, "converged", "", "", as.numeric(extract.mirt(mg, "converged")))
  add(case, "loglik", "", "", as.numeric(logLik(mg)))
  mv <- mod2values(mg)
  for (r in seq_len(nrow(mv))) if (mv$name[r] %in% c("d", "MEAN_1", "COV_11"))
    add(case, paste0("par_", mv$group[r]), mv$item[r], mv$name[r], mv$value[r])
  for (r in seq_len(nrow(mv))) if (mv$name[r] %in% c("MEAN_1", "COV_11"))
    add(case, paste0("est_", mv$group[r]), mv$item[r], mv$name[r], as.numeric(mv$est[r]))
  V <- vcov(mg); cn <- colnames(V)
  # parameter numbers of each item's d in each group, to read Var(d_F - d_R)
  for (r in seq_len(nrow(mv))) if (mv$name[r] == "d")
    add(case, paste0("parnum_", mv$group[r]), mv$item[r], "d", mv$parnum[r])
  for (i in seq_along(cn)) for (j in seq_along(cn)) add(case, "vcov", cn[i], cn[j], V[i, j])
  # eRm (CML) agreement indicator: complete-case not required (eRm handles NA)
  rm <- RM(dat)
  wt <- Waldtest(rm, splitcr = as.numeric(grp == "F"))
  for (i in seq_len(nrow(wt$coef.table))) {
    add(case, "erm_wald", sub("^beta ", "", rownames(wt$coef.table)[i]), "z", wt$coef.table[i, 1])
    add(case, "erm_wald", sub("^beta ", "", rownames(wt$coef.table)[i]), "p", wt$coef.table[i, 2])
  }
  # Fixed-parameter check of the multigroup likelihood: the multigroup logLik is the sum
  # of single-group logLiks at each group's (mu, sigma, b). Evaluated with the B3-validated
  # single-group TOL = NaN route (multipleGroup with pars + invariance + TOL = NaN did not
  # reproduce its own logLik and is not used).
  total <- 0
  for (g in c("R", "F")) {
    sub <- dat[grp == g, ]
    sv <- mirt(sub, 1, itemtype = "Rasch", pars = "values")
    for (it in colnames(dat)) sv$value[sv$item == it & sv$name == "d"] <-
      mv$value[mv$group == g & mv$item == it & mv$name == "d"]
    sv$value[sv$name == "MEAN_1"] <- mv$value[mv$group == g & mv$name == "MEAN_1"]
    sv$value[sv$name == "COV_11"] <- mv$value[mv$group == g & mv$name == "COV_11"]
    sv$est[sv$name %in% c("MEAN_1", "COV_11")] <- FALSE
    fx <- mirt(sub, 1, itemtype = "Rasch", pars = sv, TOL = NaN, verbose = FALSE)
    add(case, "group_loglik_fixed", g, "", as.numeric(logLik(fx)))
    total <- total + as.numeric(logLik(fx))
  }
  add(case, "loglik_fixed_sum", "", "", total)
  lr <- LRtest(rm, splitcr = as.numeric(grp == "F"))
  add(case, "erm_andersen_lr", "", "LR", lr$LR); add(case, "erm_andersen_lr", "", "df", lr$df)
  add(case, "erm_andersen_lr", "", "p", lr$pvalue)
}

## ---- Q3 with Python's parameters and WLE scores -------------------------------
for (case in c("ld_pair", "rasch_missing")) {
  folder <- if (case == "ld_pair") dir else dir8
  dat <- read.csv(file.path(folder, paste0("data_", case, ".csv")), na.strings = "NA")
  it <- read.csv(file.path(dir, paste0("py_q3_items_", case, ".csv")))
  pe <- read.csv(file.path(dir, paste0("py_q3_persons_", case, ".csv")))
  sv <- mirt(dat, 1, itemtype = "Rasch", pars = "values")
  for (j in seq_len(nrow(it))) sv$value[sv$item == it$Madde[j] & sv$name == "d"] <- -it$b[j]
  sv$value[sv$name == "COV_11"] <- pe$sd[1]^2
  fixed <- mirt(dat, 1, itemtype = "Rasch", pars = sv, TOL = NaN, verbose = FALSE)
  q3 <- residuals(fixed, type = "Q3", Theta = matrix(pe$theta), verbose = FALSE)
  for (i in seq_len(nrow(q3))) for (j in seq_len(ncol(q3))) if (i < j)
    add(case, "q3", rownames(q3)[i], colnames(q3)[j], q3[i, j])
}

## ---- M2: Rasch on mirt's default grid; SD fixed (comparable) and free --------
for (case in c("ld_pair", "rasch_main")) {
  folder <- if (case == "ld_pair") dir else dir8
  dat <- read.csv(file.path(folder, paste0("data_", case, ".csv")), na.strings = "NA")
  fit <- mirt(dat, 1, itemtype = "Rasch", TOL = 1e-9, verbose = FALSE, technical = list(NCYCLES = 20000))
  mv <- mod2values(fit)
  for (r in seq_len(nrow(mv))) if (mv$name[r] %in% c("d", "COV_11"))
    add(case, "m2_par", mv$item[r], mv$name[r], mv$value[r])
  free <- M2(fit, type = "M2*")
  for (s in c("M2", "df", "p", "RMSEA", "SRMSR")) add(case, "m2_free", "", s, free[[s]])
  sv <- mv; sv$est[sv$name == "COV_11"] <- FALSE
  fixsd <- mirt(dat, 1, itemtype = "Rasch", pars = sv, TOL = NaN, verbose = FALSE)
  m2f <- M2(fixsd, type = "M2*")
  for (s in c("M2", "df", "p", "RMSEA", "SRMSR")) add(case, "m2_fixed_sd", "", s, m2f[[s]])
}

write.csv(do.call(rbind, out), file.path(dir, "r_b9_results.csv"), row.names = FALSE)
writeLines(c(R.version.string, paste("mirt", as.character(packageVersion("mirt"))),
             paste("eRm", as.character(packageVersion("eRm"))),
             paste("mirt source:", Sys.getenv("MIRT_SOURCE", "CRAN/unspecified")),
             paste("generated:", format(Sys.time(), tz = "UTC", usetz = TRUE))),
           file.path(dir, "r_b9_session.txt"))
cat("done\n")

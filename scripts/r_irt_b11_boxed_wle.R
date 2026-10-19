# Supplementary R references for B11:
#  (1) mirt fits restricted to the Python box (lbound/ubound: a1 [0.15, 4], g [0, 0.35], u [0.65, 1];
#      mirt applies the logit internally to g/u) -> comparable with core.irt_mg34 on the mirt grid;
#  (2) Warm (1989) WLE from catR::thetaEst(method = "WL", D = 1) with the Python item parameters
#      (mirt's fscores(method = 'WLE') maximizes L * sqrt(I), which equals Warm's WLE only for 1PL/2PL).
#  (3) two-group DIF models restricted to the same box (round-0 omnibus LR and the final model with
#      the fixed anchor list) -> comparable with core.irt_dif34 on the mirt grid.
#   Rscript scripts/r_irt_b11_boxed_wle.R 3PL|4PL
suppressPackageStartupMessages({library(mirt); library(catR)})
args <- commandArgs(trailingOnly = TRUE); model <- if (length(args)) args[1] else "3PL"
dir <- "tests/fixtures/irt_b11"
out <- list()
add <- function(case, quantity, row, col, value) {
  value <- suppressWarnings(as.numeric(value)); if (!length(value)) value <- NA_real_   # NULL -> NA
  out[[length(out) + 1L]] <<- data.frame(case = case, quantity = quantity, row = as.character(row),
                                         col = as.character(col), value = value[1])
}
cases <- if (model == "3PL") c("irt3pl_ok", "irt3pl_ld", "irt3pl_missing", "irt3pl_easy") else c("irt4pl_ok", "irt4pl_ld")
for (case in cases) {
  dat <- read.csv(file.path(dir, paste0("data_", case, ".csv")), na.strings = "NA")
  sv <- mirt(dat, 1, itemtype = model, pars = "values")
  sv$lbound[sv$name == "a1"] <- .15; sv$ubound[sv$name == "a1"] <- 4
  sv$lbound[sv$name == "g"] <- 0;    sv$ubound[sv$name == "g"] <- .35
  if (model == "4PL") { sv$lbound[sv$name == "u"] <- .65; sv$ubound[sv$name == "u"] <- 1 }
  fit <- mirt(dat, 1, itemtype = model, pars = sv, TOL = 1e-10, technical = list(NCYCLES = 20000), verbose = FALSE)
  add(case, "box_converged", "", "", extract.mirt(fit, "converged"))
  add(case, "box_loglik", "", "", logLik(fit))
  mv <- mod2values(fit)
  for (r in seq_len(nrow(mv))) if (mv$name[r] %in% c("a1", "d", "g", "u")) add(case, "box_par", mv$item[r], mv$name[r], mv$value[r])
  it <- read.csv(file.path(dir, paste0("py_items_", case, ".csv")))
  m <- cbind(a = it$a, b = it$b, c = it$c, d = it$d)
  x <- as.matrix(dat)
  for (p in seq_len(nrow(x))) {
    o <- !is.na(x[p, ])
    add(case, "catR_wle", p, "", thetaEst(m[o, , drop = FALSE], x[p, o], method = "WL", D = 1, range = c(-12, 12)))
  }
}
box <- function(sv) {
  sv$lbound[sv$name == "a1"] <- .15; sv$ubound[sv$name == "a1"] <- 4
  sv$lbound[sv$name == "g"] <- 0;    sv$ubound[sv$name == "g"] <- .35
  if (model == "4PL") { sv$lbound[sv$name == "u"] <- .65; sv$ubound[sv$name == "u"] <- 1 }
  sv
}
dif_cases <- if (model == "3PL") c("dif3pl", "dif3pl_none") else c("dif4pl", "dif4pl_none")
for (case in dif_cases) {
  d <- read.csv(file.path(dir, paste0("data_", case, ".csv")), na.strings = "NA")
  grp <- factor(d$Grup, levels = c("R", "F")); dat <- d[, -1]; items <- colnames(dat)
  lat <- c("free_means", "free_var")
  sv <- box(multipleGroup(dat, 1, group = grp, itemtype = model, pars = "values"))
  tol <- if (model == "4PL") 1e-6 else 1e-8; cyc <- if (model == "4PL") 3000 else 20000   # 4PL: see main script
  mgb <- function(inv, SE = FALSE) multipleGroup(dat, 1, group = grp, itemtype = model, pars = sv, invariance = inv,
                                                 TOL = tol, SE = SE, technical = list(NCYCLES = cyc), verbose = FALSE)
  m0 <- mgb(c(items, lat))
  add(case, "box_base_loglik", "", "", logLik(m0))
  for (it in items) {
    m_all <- mgb(c(setdiff(items, it), lat))
    add(case, "box_lr_omnibus", it, "", 2 * (logLik(m_all) - logLik(m0)))
  }
  anchors <- read.csv(file.path(dir, paste0("py_dif_anchors_", case, ".csv")))$anchor
  fin <- mgb(c(anchors, lat))
  add(case, "box_final_loglik", "", "", logLik(fin))
  mv <- mod2values(fin)
  for (r in seq_len(nrow(mv))) if (mv$name[r] %in% c("a1", "d", "g", "u", "MEAN_1", "COV_11"))
    add(case, paste0("box_final_par_", mv$group[r]), mv$item[r], mv$name[r], mv$value[r])
}

write.csv(do.call(rbind, out), file.path(dir, paste0("r_b11_boxed_wle_", model, ".csv")), row.names = FALSE)
writeLines(c(R.version.string, paste("mirt", as.character(packageVersion("mirt"))),
             paste("catR", as.character(packageVersion("catR")), "(github cran/catR mirror)"),
             paste("generated:", format(Sys.time(), tz = "UTC", usetz = TRUE))),
           file.path(dir, "r_b11_boxed_wle_session.txt"))
cat("done\n")

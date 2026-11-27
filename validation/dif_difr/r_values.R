suppressPackageStartupMessages({library(difR); library(mirt); library(jsonlite)})
a <- commandArgs(TRUE); py <- fromJSON(a[1]); res <- list()
for (f in names(py)) {
  d <- read.csv(sprintf("tests/fixtures/irt_b10/data_%s.csv", f)); P <- py[[f]]; x <- d[, P$items]; g <- d$Grup
  mh <- difMH(Data = x, group = g, focal.name = "F", correct = TRUE, purify = FALSE, match = "score", p.adjust.method = "BH")
  mhp <- difMH(Data = x, group = g, focal.name = "F", correct = TRUE, purify = TRUE, match = "score")
  lg <- difLogistic(Data = x, group = g, focal.name = "F", type = "both", criterion = "LRT", purify = FALSE)
  lu <- difLogistic(Data = x, group = g, focal.name = "F", type = "udif", criterion = "LRT", purify = FALSE)
  gm <- ifelse(g == "F", "focal", "reference"); A <- match(P$anchors, P$items)
  J <- match(P$sib_items, P$items)
  sib <- if (length(J) == 0) matrix(numeric(0), 0, 3, dimnames = list(NULL, c("beta", "X2", "p"))) else t(sapply(J, function(j) { s <- SIBTEST(as.matrix(x), gm, suspect_set = j, match_set = A, focal_name = "focal")
                 c(beta = s$beta[1], X2 = s$X2[1], p = s$p[1]) }))
  res[[f]] <- list(mh_chi2 = unname(mh$MH), mh_alpha = unname(mh$alphaMH), mh_delta = -2.35 * log(unname(mh$alphaMH)), mh_p = unname(mh$p.value),
                   mh_p_bh = unname(mh$adjusted.p), mh_purified_chi2 = unname(mhp$MH), lr_omnibus = unname(lg$Logistik), lr_uniform = unname(lu$Logistik),
                   lr_p = unname(lg$p.value), lr_dr2 = unname(lg$deltaR2), sib_beta = unname(sib[, "beta"]), sib_chi2 = unname(sib[, "X2"]), sib_p = unname(sib[, "p"]))
}
res$versions <- list(difR = as.character(packageVersion("difR")), mirt = as.character(packageVersion("mirt")))
write(toJSON(res, digits = NA), a[2])

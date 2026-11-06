# Rscript reference.R <irr src> <irrCAC src> <examples dir> <out.csv>
# Independent references (PROTOCOL.json): irr 0.85 @98e4535, irrCAC 1.4 @720ab22 (sourced from the CRAN GitHub
# mirrors), psych::ICC/cohen.kappa (CRAN), base R tests, and analytic formulas written separately from Python.
args <- commandArgs(trailingOnly = TRUE)
suppressPackageStartupMessages({ library(dplyr); library(magrittr) })
for (f in list.files(file.path(args[1], "R"), full.names = TRUE)) source(f)
for (f in list.files(file.path(args[2], "R"), full.names = TRUE)) source(f)
ex_dir <- args[3]; out <- list()
put <- function(id, method, ref, key, value) out[[length(out) + 1]] <<- data.frame(example = id, method = method,
  reference = ref, key = key, value = as.numeric(value))
reg <- readLines(file.path(ex_dir, "registry.json"), warn = FALSE)
# minimal registry parse without jsonlite: example id, method, file, layout (one object per example block)
blocks <- regmatches(paste(reg, collapse = "\n"), gregexpr('"id": "[^"]+",\\s*"method": "[^"]+",\\s*"file": "[^"]+"[^}]*"layout": "[a-z]+"', paste(reg, collapse = "\n")))[[1]]
field <- function(b, k) sub(paste0('.*"', k, '": "([^"]+)".*'), "\\1", b)
table2 <- function(x) { lv <- sort(unique(c(x[, 1], x[, 2]))); table(factor(x[, 1], lv), factor(x[, 2], lv)) }
for (b in blocks) {
  id <- field(b, "id"); m <- field(b, "method"); file <- field(b, "file"); layout <- sub('.*"layout": "([a-z]+)".*', "\\1", b)
  d <- read.csv(file.path(ex_dir, file), check.names = FALSE, stringsAsFactors = FALSE)
  x <- d[, -1, drop = FALSE]
  if (layout == "counts") {
    cnt <- as.matrix(x)
    raw <- t(apply(cnt, 1, function(r) rep(seq_along(r), r)))   # rater identity is irrelevant for Fleiss
    put(id, m, "irrCAC::fleiss.kappa.dist", "estimate", fleiss.kappa.dist(cnt)$coeff)
    put(id, m, "irr::kappam.fleiss", "estimate", kappam.fleiss(raw)$value)
    next
  }
  if (m %in% c("cohen_kappa", "cohen_kappa_linear", "cohen_kappa_quadratic")) {
    w <- c(cohen_kappa = "unweighted", cohen_kappa_linear = "equal", cohen_kappa_quadratic = "squared")[[m]]
    k <- kappa2(x, weight = w)
    put(id, m, "irr::kappa2", "estimate", k$value); put(id, m, "irr::kappa2", "z0", k$statistic); put(id, m, "irr::kappa2", "p", k$p.value)
    lv <- sort(unique(c(x[, 1], x[, 2]))); q <- length(lv); codes <- data.frame(a = match(x[, 1], lv), b = match(x[, 2], lv))
    W <- if (m == "cohen_kappa_linear") 1 - abs(outer(1:q, 1:q, "-")) / (q - 1) else 1 - outer(1:q, 1:q, "-")^2 / (q - 1)^2
    ck <- psych::cohen.kappa(codes, w = W, levels = 1:q)
    row <- if (m == "cohen_kappa") 1 else 2
    put(id, m, "psych::cohen.kappa", "estimate", ck$confid[row, 2])
    put(id, m, "psych::cohen.kappa", "se", sqrt(if (row == 1) ck$var.kappa else ck$var.weighted))
    put(id, m, "psych::cohen.kappa", "ci_low", ck$confid[row, 1]); put(id, m, "psych::cohen.kappa", "ci_high", ck$confid[row, 3])
  }
  if (m == "conger_kappa") { put(id, m, "irrCAC::conger.kappa.raw", "estimate", conger.kappa.raw(x)$est$coeff.val)
    put(id, m, "irr::kappam.fleiss(exact)", "estimate", kappam.fleiss(x, exact = TRUE)$value) }
  if (m == "light_kappa") put(id, m, "irr::kappam.light", "estimate", kappam.light(x)$value)
  if (m == "scott_pi") { put(id, m, "irrCAC::fleiss.kappa.raw", "estimate", fleiss.kappa.raw(x)$est$coeff.val)
    tb <- table2(x); n <- sum(tb); po <- sum(diag(tb)) / n; pe <- sum(((rowSums(tb) + colSums(tb)) / (2 * n))^2)
    put(id, m, "analytic", "estimate", (po - pe) / (1 - pe)) }
  if (m == "gwet_ac1") put(id, m, "irrCAC::gwet.ac1.raw", "estimate", gwet.ac1.raw(x)$est$coeff.val)
  if (m == "gwet_ac2") put(id, m, "irrCAC::gwet.ac1.raw(quadratic)", "estimate", gwet.ac1.raw(x, weights = "quadratic")$est$coeff.val)
  if (m == "brennan_prediger") put(id, m, "irrCAC::bp.coeff.raw", "estimate", bp.coeff.raw(x)$est$coeff.val)
  if (grepl("^kripp_", m)) {
    lev <- sub("kripp_", "", m); put(id, m, "irr::kripp.alpha", "estimate", kripp.alpha(t(as.matrix(x)), method = lev)$value)
    wt <- c(nominal = "unweighted", interval = "quadratic", ratio = "ratio")[lev]
    if (!is.na(wt)) put(id, m, "irrCAC::krippen.alpha.raw", "estimate", krippen.alpha.raw(x, weights = wt)$est$coeff.val)
  }
  if (m == "percent_agreement") { put(id, m, "irrCAC::pa.coeff.raw", "estimate", pa.coeff.raw(x)$est$coeff.val)
    put(id, m, "irr::agree", "all_agree", agree(x)$value / 100) }
  if (m == "pabak") { put(id, m, "irrCAC::bp.coeff.raw", "estimate", bp.coeff.raw(x)$est$coeff.val)
    put(id, m, "analytic", "estimate", 2 * mean(x[, 1] == x[, 2]) - 1) }
  if (m == "bangdiwala_b") { tb <- table2(x); put(id, m, "analytic", "estimate", sum(diag(tb)^2) / sum(rowSums(tb) * colSums(tb))) }
  if (grepl("^icc_", m)) {
    spec <- list(icc_1_1 = c("oneway", "agreement", "single", "ICC1"), icc_1_k = c("oneway", "agreement", "average", "ICC1k"),
                 icc_2_1 = c("twoway", "agreement", "single", "ICC2"), icc_2_k = c("twoway", "agreement", "average", "ICC2k"),
                 icc_3_1 = c("twoway", "consistency", "single", "ICC3"), icc_3_k = c("twoway", "consistency", "average", "ICC3k"))[[m]]
    r <- icc(x, model = spec[1], type = spec[2], unit = spec[3])
    for (kv in list(c("estimate", r$value), c("F", r$Fvalue), c("df1", r$df1), c("df2", r$df2), c("p", r$p.value),
                    c("ci_low", r$lbound), c("ci_high", r$ubound))) put(id, m, "irr::icc", kv[1], kv[2])
    ps <- psych::ICC(x, lmer = FALSE)$results; row <- ps[ps$type == spec[4], ]
    for (kv in list(c("estimate", row$ICC), c("F", row$F), c("df1", row$df1), c("df2", row$df2), c("p", row$p),
                    c("ci_low", row$`lower bound`), c("ci_high", row$`upper bound`))) put(id, m, "psych::ICC", kv[1], kv[2])
  }
  if (m == "kendall_w") { k <- kendall(x, correct = TRUE)
    for (kv in list(c("estimate", k$value), c("chi2", k$statistic), c("df", k$subjects - 1), c("p", k$p.value))) put(id, m, "irr::kendall", kv[1], kv[2]) }
  if (m == "spearman") { r <- cor.test(x[, 1], x[, 2], method = "spearman", exact = FALSE)
    put(id, m, "stats::cor.test", "estimate", r$estimate); put(id, m, "stats::cor.test", "p", r$p.value) }
  if (m == "pearson") { r <- cor.test(x[, 1], x[, 2], method = "pearson")
    put(id, m, "stats::cor.test", "estimate", r$estimate); put(id, m, "stats::cor.test", "p", r$p.value)
    put(id, m, "stats::cor.test", "ci_low", r$conf.int[1]); put(id, m, "stats::cor.test", "ci_high", r$conf.int[2]) }
  if (m == "mcnemar") { r <- mcnemar.test(table2(x), correct = TRUE)
    put(id, m, "stats::mcnemar.test", "chi2", r$statistic); put(id, m, "stats::mcnemar.test", "df", r$parameter); put(id, m, "stats::mcnemar.test", "p", r$p.value) }
  if (m %in% c("stuart_maxwell", "bhapkar")) {
    tb <- unclass(table2(x)); k <- nrow(tb); n <- sum(tb); dd <- (rowSums(tb) - colSums(tb))[-k]
    S <- -(tb + t(tb))[-k, -k]; diag(S) <- (rowSums(tb) + colSums(tb) - 2 * diag(tb))[-k]
    sm <- drop(t(dd) %*% solve(S, dd)); stat <- if (m == "bhapkar") sm / (1 - sm / n) else sm
    put(id, m, "analytic", "chi2", stat); put(id, m, "analytic", "df", k - 1); put(id, m, "analytic", "p", pchisq(stat, k - 1, lower.tail = FALSE))
  }
  if (m == "bland_altman") { dd <- x[, 1] - x[, 2]; n <- length(dd); b <- mean(dd); s <- sd(dd); tq <- qt(.975, n - 1)
    for (kv in list(c("bias", b), c("sd", s), c("loa_low", b - 1.96 * s), c("loa_high", b + 1.96 * s),
                    c("bias_ci_low", b - tq * s / sqrt(n)), c("bias_ci_high", b + tq * s / sqrt(n)),
                    c("loa_low_ci_low", b - 1.96 * s - tq * s * sqrt(3 / n)), c("loa_high_ci_high", b + 1.96 * s + tq * s * sqrt(3 / n))))
      put(id, m, "analytic", kv[1], kv[2]) }
}
res <- do.call(rbind, out); write.csv(res, args[4], row.names = FALSE)
cat(nrow(res), "reference values,", length(unique(res$example)), "examples\n")
writeLines(capture.output(sessionInfo()), sub("\\.csv$", "_sessionInfo.txt", args[4]))

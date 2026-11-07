# Rscript reference.R binary.csv ordinal.csv out.csv — independent observed-score DIF references (PROTOCOL.json).
suppressPackageStartupMessages(library(MASS))
a <- commandArgs(trailingOnly = TRUE); out <- list()
put <- function(kind, item, key, v) out[[length(out) + 1]] <<- data.frame(kind = kind, item = item, key = key, value = as.numeric(v))
nag <- function(ll, ll0, n) (1 - exp(2 * (ll0 - ll) / n)) / (1 - exp(2 * ll0 / n))
lr_block <- function(kind, item, y, g, m, ordinal) {
  z <- (m - mean(m)) / sqrt(mean((m - mean(m))^2))                    # numpy std (ddof = 0), same design matrix
  n <- length(y)
  if (ordinal) {
    yy <- factor(y, ordered = TRUE); fit <- function(f) as.numeric(logLik(polr(f, method = "logistic", Hess = FALSE, control = list(reltol = 1e-15, maxit = 5000))))
    ll <- c(fit(yy ~ z), fit(yy ~ z + g), fit(yy ~ z * g)); cnt <- table(y); ll0 <- sum(cnt * log(cnt / n))
  } else {
    fit <- function(f) as.numeric(logLik(glm(f, family = binomial, control = glm.control(epsilon = 1e-14, maxit = 100))))
    ll <- c(fit(y ~ z), fit(y ~ z + g), fit(y ~ z * g)); p1 <- mean(y); ll0 <- n * (p1 * log(p1) + (1 - p1) * log(1 - p1))
  }
  lr <- c(uniform = 2 * (ll[2] - ll[1]), nonuniform = 2 * (ll[3] - ll[2]), omnibus = 2 * (ll[3] - ll[1]))
  for (k in names(lr)) { put(kind, item, paste0("lr_", k), lr[[k]]); put(kind, item, paste0("p_", k), pchisq(lr[[k]], if (k == "omnibus") 2 else 1, lower.tail = FALSE)) }
  r2 <- nag(ll, ll0, n); put(kind, item, "delta_r2", r2[3] - r2[1]); put(kind, item, "delta_r2_uniform", r2[2] - r2[1])
}
b <- read.csv(a[1]); g <- as.integer(b$Grup == "F"); x <- as.matrix(b[, -1]); tot <- rowSums(x)
for (j in seq_len(ncol(x))) {
  y <- x[, j]; keep <- tot %in% as.numeric(names(which(table(tot) > 1)))
  tb <- table(factor(g[keep], c(0, 1)), factor(y[keep], c(1, 0)), tot[keep])
  mh <- mantelhaen.test(tb, correct = TRUE); it <- colnames(x)[j]
  put("binary", it, "chi2", mh$statistic); put("binary", it, "p", mh$p.value); put("binary", it, "alpha_mh", mh$estimate)
  put("binary", it, "delta", -2.35 * log(mh$estimate))
  put("binary", it, "lnOR %95 alt", log(mh$conf.int[1])); put("binary", it, "lnOR %95 üst", log(mh$conf.int[2]))
  lr_block("binary", it, y, g, tot, FALSE)
}
o <- read.csv(a[2]); g <- as.integer(o$grup == "Odak"); x <- as.matrix(o[, -1]); tot <- rowSums(x)
for (j in seq_len(ncol(x))) {
  y <- x[, j]; it <- colnames(x)[j]; Fs <- EF <- VF <- smd <- nFt <- 0
  for (s in unique(tot)) { ix <- tot == s; r <- ix & g == 0; f <- ix & g == 1; nR <- sum(r); nF <- sum(f); T <- nR + nF
    if (nR == 0 || nF == 0 || T < 2) next
    nFt <- nFt + nF; smd <- smd + nF * (mean(y[f]) - mean(y[r]))
    v <- nR * nF / (T^2 * (T - 1)) * (T * sum(y[ix]^2) - sum(y[ix])^2); if (v <= 0) next
    Fs <- Fs + sum(y[f]); EF <- EF + nF * sum(y[ix]) / T; VF <- VF + v }
  st <- (Fs - EF)^2 / VF
  put("ordinal", it, "chi2", st); put("ordinal", it, "p", pchisq(st, 1, lower.tail = FALSE)); put("ordinal", it, "smd", smd / nFt)
  lr_block("ordinal", it, y, g, tot, TRUE)
}
write.csv(do.call(rbind, out), a[3], row.names = FALSE)
writeLines(capture.output(sessionInfo()), sub("\\.csv$", "_sessionInfo.txt", a[3]))

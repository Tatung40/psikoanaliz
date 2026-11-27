# Independent R references: psych::alpha, psych::guttman (lambda 1-6 on the correlation matrix), lavaan one-factor ML CFA omega.
args <- commandArgs(TRUE); d <- args[1]
suppressMessages({library(psych); library(lavaan); library(jsonlite)})
rc <- tryCatch({ if (!requireNamespace("reliacoef", quietly = TRUE)) install.packages("reliacoef", repos = "https://cloud.r-project.org", quiet = TRUE)
                 requireNamespace("reliacoef", quietly = TRUE) }, error = function(e) FALSE)
rc_exports <- if (isTRUE(rc)) getNamespaceExports("reliacoef") else character(0)
out <- list()
for (nm in c("poly", "binary")) {
  x <- read.csv(file.path(d, paste0(nm, ".csv")))
  a <- psych::alpha(x, warnings = FALSE)$total$raw_alpha
  g <- psych::guttman(cor(x))
  z <- as.data.frame(scale(x)); v <- names(z)
  fit <- lavaan::cfa(paste("F =~", paste(v, collapse = " + ")), data = z, std.lv = TRUE, estimator = "ML")
  pe <- parameterEstimates(fit); l <- pe$est[pe$op == "=~"]; th <- pe$est[pe$op == "~~" & pe$lhs == pe$rhs & pe$lhs %in% v]
  fr <- lavaan::cfa(paste("F =~", paste(v, collapse = " + ")), data = x, std.lv = TRUE, estimator = "ML")
  pr <- parameterEstimates(fr); lr <- pr$est[pr$op == "=~"]; tr <- pr$est[pr$op == "~~" & pr$lhs == pr$rhs & pr$lhs %in% v]
  h1 <- rowSums(x[, seq(1, ncol(x), 2)]); h2 <- rowSums(x[, seq(2, ncol(x), 2)]); rr <- cor(h1, h2); tot <- h1 + h2
  k <- ncol(x); m <- mean(tot); S <- cov(x); Vt <- sum(S)
  fe <- list()
  for (f in intersect(c("feldt", "gilmer", "feldt_gilmer", "feldt_brennan"), rc_exports)) {
    fe[[f]] <- tryCatch(unname(unlist(getExportedValue("reliacoef", f)(S, print = FALSE))[1]), error = function(e) tryCatch(unname(unlist(getExportedValue("reliacoef", f)(S))[1]), error = function(e2) NA))
  }
  out[[nm]] <- list(omega_raw_ml = sum(lr)^2 / (sum(lr)^2 + sum(tr)), raw_heywood = any(tr <= 0),
                    sb_odd_even = 2 * rr / (1 + rr), guttman_l4_odd_even = 2 * (1 - (var(h1) + var(h2)) / var(tot)),
                    kr21 = k / (k - 1) * (1 - m * (k - m) / (k * var(tot))), reliacoef = fe, reliacoef_exports = rc_exports,
                    feldt_brennan_formula = Vt * (Vt - sum(diag(S))) / (Vt^2 - sum(rowSums(S)^2)),
                    alpha = a, omega_std_ml = sum(l)^2 / (sum(l)^2 + sum(th)), lavaan_converged = lavInspect(fit, "converged"),
                    heywood = any(th <= 0), lambda1 = g$lambda.1, lambda2 = g$lambda.2, lambda3 = g$lambda.3, lambda6 = g$lambda.6,
                    psych = as.character(packageVersion("psych")), lavaan = as.character(packageVersion("lavaan")))
}
write(toJSON(out, auto_unbox = TRUE, digits = NA), file.path(d, "r.json"))

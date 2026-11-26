# Independent R references: psych::alpha, psych::guttman (lambda 1-6 on the correlation matrix), lavaan one-factor ML CFA omega.
args <- commandArgs(TRUE); d <- args[1]
suppressMessages({library(psych); library(lavaan); library(jsonlite)})
out <- list()
for (nm in c("poly", "binary")) {
  x <- read.csv(file.path(d, paste0(nm, ".csv")))
  a <- psych::alpha(x, warnings = FALSE)$total$raw_alpha
  g <- psych::guttman(cor(x))
  z <- as.data.frame(scale(x)); v <- names(z)
  fit <- lavaan::cfa(paste("F =~", paste(v, collapse = " + ")), data = z, std.lv = TRUE, estimator = "ML")
  pe <- parameterEstimates(fit); l <- pe$est[pe$op == "=~"]; th <- pe$est[pe$op == "~~" & pe$lhs == pe$rhs & pe$lhs %in% v]
  out[[nm]] <- list(alpha = a, omega_std_ml = sum(l)^2 / (sum(l)^2 + sum(th)), lavaan_converged = lavInspect(fit, "converged"),
                    heywood = any(th <= 0), lambda1 = g$lambda.1, lambda2 = g$lambda.2, lambda3 = g$lambda.3, lambda6 = g$lambda.6,
                    psych = as.character(packageVersion("psych")), lavaan = as.character(packageVersion("lavaan")))
}
write(toJSON(out, auto_unbox = TRUE, digits = NA), file.path(d, "r.json"))

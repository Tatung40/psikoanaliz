# Amendment B2 R checks (4PL): R's own log-posterior at Python's and at R's solution (TOL = NaN evaluation at fixed
# parameters + R log-prior), and R EM started at Python's solution (matched initialization; dependent stationarity check).
# Usage: Rscript b2_checks.R manifest.json out_dir   (needs <id>.python.json and <id>.R.json in out_dir)
suppressPackageStartupMessages({library(mirt); library(jsonlite)})
args <- commandArgs(trailingOnly = TRUE); cases <- fromJSON(args[1], simplifyVector = FALSE); out <- args[2]
stopifnot(as.character(packageVersion("mirt")) == "1.48")
source_lines <- readLines("validation/binary_certification/reference.R")
eval(parse(text = source_lines[seq_len(grep("^settings <- list", source_lines) - 1)]))   # same P4, derivs, items (A3/A4)
lprior <- function(p) sum(dlnorm(p$a, 0, .5, log = TRUE) + dnorm(p$b, 0, 2, log = TRUE) + dbeta(p$c, 2, 8, log = TRUE) + dbeta(p$d, 8, 2, log = TRUE))
for (case in cases) {
  res <- tryCatch({
    dat <- read.csv(case$path); k <- ncol(dat)
    sv <- mirt(dat, 1, itemtype = "irt4", customItems = items["irt4"], pars = "values")
    with_par <- function(p) { s <- sv; for (q in c("a", "b", "c", "d")) s$value[s$name == q] <- unlist(p[[q]]); s }
    eval_at <- function(p) extract.mirt(mirt(dat, 1, itemtype = "irt4", customItems = items["irt4"], pars = with_par(p),
                                             quadpts = 101, theta_lim = c(-6, 6), TOL = NaN, verbose = FALSE), "logLik") + lprior(lapply(p, unlist))
    py <- fromJSON(file.path(out, paste0(case$id, ".python.json")), simplifyVector = FALSE)$par
    rr <- tryCatch(fromJSON(file.path(out, paste0(case$id, ".R.json")), simplifyVector = FALSE)$par, error = function(e) NULL)
    matched <- mirt(dat, 1, itemtype = "irt4", customItems = items["irt4"], pars = with_par(py), quadpts = 101, theta_lim = c(-6, 6),
                    TOL = 1e-8, optimizer = "NR", verbose = FALSE, SE = FALSE, technical = list(NCYCLES = 5000))
    cf <- coef(matched)[seq_len(k)]
    mp <- setNames(lapply(c("a", "b", "c", "d"), function(q) unname(sapply(cf, function(m) m["par", q]))), c("a", "b", "c", "d"))
    list(id = case$id, R_logpost_at_python = eval_at(py), R_logpost_at_R = if (is.null(rr)) NA else eval_at(rr),
         matched_converged = extract.mirt(matched, "converged"), matched_iterations = extract.mirt(matched, "iterations"), matched_par = mp)
  }, error = function(e) list(id = case$id, error = conditionMessage(e)))
  write_json(res, file.path(out, paste0(case$id, ".Rb2.json")), digits = NA, auto_unbox = TRUE, na = "null")
  cat(case$id, if (is.null(res$error)) sprintf("R logpost python=%.4f R=%.4f matched_iter=%s", res$R_logpost_at_python, res$R_logpost_at_R, res$matched_iterations) else res$error, "\n")
}

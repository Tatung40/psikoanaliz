# No fits. Run from repository root: Rscript validation/r_mirt_phase1/runtime_preflight.R OUTDIR
args <- commandArgs(TRUE)
stopifnot(length(args) == 1L, requireNamespace("mirt", quietly=TRUE))
out <- args[1]; dir.create(out, recursive=TRUE, showWarnings=FALSE)
capture.output(sessionInfo(), file=file.path(out,"sessionInfo.txt"))
stopifnot(as.character(packageVersion("mirt")) == "1.48")
source("validation/r_mirt_phase1/physical_b_constraint.R")
# Exercise the installed package's conversions, not a presumed qlogis convention.
logit <- getFromNamespace("logit", "mirt")
inv <- getFromNamespace("antilogit", "mirt")
x <- c(0, .Machine$double.xmin, .Machine$double.eps, .5,
       1-.Machine$double.eps, 1)
y <- inv(logit(x))
write.csv(data.frame(physical=x, internal=logit(x), back=y,
                     difference=y-x, identical_float=y==x),
          file.path(out,"endpoints.csv"), row.names=FALSE)
# Numerical endpoint equality is evidence to assess, not an assumed obstruction.
stopifnot(y[1] == 0, y[length(y)] == 1)
z <- c(-1000,-999,-50,-4,0,4,50,999,1000)
rows <- list()
for (g in c(0,.5)) for (u in c(.5,1)) {
  direct <- g+(u-g)*plogis(z)
  transformed <- inv(logit(g))+(inv(logit(u))-inv(logit(g)))*plogis(z)
  rows[[length(rows)+1L]] <- data.frame(g=g,u=u,z=z,direct=direct,
    transformed=transformed,difference=transformed-direct)
}
write.csv(do.call(rbind,rows),file.path(out,"boundary_probabilities.csv"),row.names=FALSE)
stopifnot(all(vapply(rows,function(r) all(r$difference==0),logical(1))))
# Parse the callback and check the runtime namespace exposes the documented API.
stopifnot("nloptr_args" %in% names(formals(mirt::mirt)))
impl <- paste(deparse(body(getFromNamespace("Mstep","mirt"))),collapse="\n")
stopifnot(grepl("eval_g_ineq",impl,fixed=TRUE),grepl("optim_args",impl,fixed=TRUE))
# pars='values' initializes specifications only; it performs no calibration.
dat <- as.data.frame(matrix(rep(c(0,1,1,0,0,1),10),ncol=3))
checks <- list()
for (model in c("3PL","4PL")) {
  values <- mirt::mirt(dat,1,itemtype=model,pars="values",verbose=FALSE)
  callback <- physical_b_constraint(values)
  for (a in c(.1,1,5)) for (b in c(-6.1,-6,0,6,6.1)) {
    full <- values$value; full[values$name=="a1"] <- a
    full[values$name=="d"] <- -a*b
    est <- values$est
    constraints <- callback(full[est],list(longpars=full,est=est))
    stopifnot(all(constraints<=0) == (abs(b)<=6))
    checks[[length(checks)+1L]] <- data.frame(model=model,a=a,b=b,
      feasible=all(constraints<=0))
  }
}
write.csv(do.call(rbind,checks),file.path(out,"coupled_bounds.csv"),row.names=FALSE)
writeLines("PREFLIGHT ONLY: no optimizer invocation or matched fits; endpoint equality does not certify gradient/likelihood/optimizer equivalence.",file.path(out,"status.txt"))

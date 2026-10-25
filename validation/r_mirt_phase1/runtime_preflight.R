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
# Regression for run 40149159575: named optimizer flags must preserve the same
# Jacobian, while a genuinely different free-coordinate layout must still fail.
source("validation/r_mirt_phase1/matched_ml.R")
test_named_optimizer_layout <- function() {
  td <- tempfile("jacobian-layout-",tmpdir=out); dir.create(td)
  on.exit(unlink(td,recursive=TRUE))
  # Configuration-only test grid, never used for a fit.
  write.csv(data.frame(theta=seq(-4,4,length.out=61),weight=rep(1/61,61)),
            file.path(td,"integration.csv"),row.names=FALSE)
  for(model in c("3PL","4PL")) {
    values <- mirt::mirt(dat,1,itemtype=model,pars="values",verbose=FALSE)
    cfg <- matched_ml(dat,model,values,td)
    flags <- cfg$pars$est
    named <- setNames(flags,cfg$pars$name)
    fn <- cfg$nloptr_args$eval_jac_g_ineq
    p <- cfg$pars$value[flags]
    stopifnot(identical(fn(p,list(est=flags)),fn(p,list(est=named))))
    changed <- named; changed[which(changed)[1]] <- FALSE
    changed[which(!named)[1]] <- TRUE
    stopifnot(inherits(try(fn(p,list(est=changed)),silent=TRUE),"try-error"))
    if(model=="4PL") {
      seed <- data.frame(item=colnames(dat),a=c(.3,1.1,5),b=c(-5.9,.25,2.2),
                         c=c(0,.2,.5),d=c(1,.629,.95))
      path <- file.path(td,"warm.csv");write.csv(seed,path,row.names=FALSE)
      warm <- matched_ml(dat,model,values,td,initialization=path)$pars
      for(i in seq_len(nrow(seed))) {
        row <- seed[i,]; expected <- c(a1=row$a,d=-row$a*row$b,g=row$c,u=row$d)
        for(n in names(expected)) stopifnot(warm$value[warm$item==row$item & warm$name==n]==expected[[n]])
      }
    }
  }
}
test_named_optimizer_layout()
# Targeted regression: installed stopping rule and history ordering at the cap.
source("validation/r_mirt_phase1/optimizer_trace.R")
test_convergence_trace_rule <- function() {
  rule <- getFromNamespace("hasConverged","mirt")
  p0 <- c(a1=0,d=0,g=-999,u=999)
  p1 <- c(a1=1e-9,d=0,g=-998,u=998)
  stopifnot(rule(p0,p1,1e-8)) # Logit moves of 1 become zero physical endpoint changes.
  at_threshold <- p0; at_threshold["a1"] <- 1e-8
  stopifnot(!rule(p0,at_threshold,1e-8)) # Strict '<', not '<='.
  completed <- convergence_trace_pair(rbind(p0,p0,p1),p1,2L)
  early_break <- convergence_trace_pair(rbind(p0,p0),p1,2L)
  stopifnot(identical(completed,early_break),
            identical(completed$previous,as.numeric(p0)),
            identical(completed$current,as.numeric(p1)))
  stopifnot(is.null(convergence_trace_pair(rbind(p0,p0,p0),p1,2L)))
}
test_convergence_trace_rule()
writeLines("PREFLIGHT ONLY: no optimizer invocation or matched fits; endpoint equality does not certify gradient/likelihood/optimizer equivalence.",file.path(out,"status.txt"))

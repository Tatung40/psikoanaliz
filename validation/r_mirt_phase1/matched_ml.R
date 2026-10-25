# Optional validation-only R mirt 1.48 configuration; no production defaults.
matched_ml <- function(dat, model, sv, out) {
  stopifnot(model %in% c("3PL","4PL"), as.character(packageVersion("mirt"))=="1.48",
            requireNamespace("nloptr",quietly=TRUE))
  source("validation/r_mirt_phase1/physical_b_constraint.R")
  grid <- read.csv(file.path(out,"integration.csv"))
  stopifnot(nrow(grid)==61, all(grid$weight>0), abs(sum(grid$weight)-1)<1e-14)
  sv$value[sv$name=="a1"] <- 1
  sv$value[sv$name=="d"] <- 0
  sv$value[sv$name=="g"] <- .2
  sv$value[sv$name=="u"] <- if(model=="4PL") .95 else 1
  sv$value[sv$name=="MEAN_1"] <- 0; sv$value[sv$name=="COV_11"] <- 1
  sv$est[sv$name %in% c("MEAN_1","COV_11")] <- FALSE
  stopifnot(all(sv$est == (sv$name %in% c("a1","d","g",if(model=="4PL") "u"))))
  boxes <- list(a1=c(.1,5),d=c(-30,30),g=c(0,.5),u=c(.5,1))
  for(n in names(boxes)) {
    sv$lbound[sv$name==n] <- boxes[[n]][1]
    sv$ubound[sv$name==n] <- boxes[[n]][2]
  }
  stopifnot(all(sv$prior.type=="none"))
  # nloptr receives the compressed, internal logit-coordinate free vector.
  lower <- sv$lbound; upper <- sv$ubound
  idx <- sv$name %in% c("g","u")
  logit <- getFromNamespace("logit","mirt")
  lower[idx] <- logit(lower[idx]); upper[idx] <- logit(upper[idx])
  free <- which(sv$est)
  aidx <- which(sv$name=="a1"); didx <- which(sv$name=="d")
  stopifnot(identical(sv$item[aidx],sv$item[didx]))
  jac <- matrix(0,2*length(aidx),length(free))
  for(i in seq_along(aidx)) {
    jac[c(i,i+length(aidx)),match(aidx[i],free)] <- -6
    jac[i,match(didx[i],free)] <- 1
    jac[i+length(aidx),match(didx[i],free)] <- -1
  }
  callback <- physical_b_constraint(sv)
  internal <- sv$value; internal[idx] <- logit(internal[idx])
  optargs <- list(longpars=internal,est=sv$est)
  p <- internal[free]; baseline <- callback(p,optargs)
  stopifnot(all(baseline<=0))
  for(j in seq_along(p)) {
    trial <- p; trial[j] <- trial[j]+1
    stopifnot(identical(as.numeric(callback(trial,optargs)-baseline),as.numeric(jac[,j])))
  }
  write.csv(sv,file.path(out,"configured_values.csv"),row.names=FALSE)
  list(data=dat,model=1,itemtype=model,pars=sv,method="EM",quadpts=61,
    TOL=1e-8,accelerate="none",SE=FALSE,verbose=FALSE,optimizer="nloptr",
    technical=list(NCYCLES=3000,customTheta=matrix(grid$theta,ncol=1),
      customPriorFun=function(Theta,Etable) grid$weight),
    nloptr_args=list(lb=lower[free],ub=upper[free],
      eval_g_ineq=callback,
      eval_jac_g_ineq=function(p,optim_args) {
        # mirt's internal est is named; names do not change coordinate order.
        stopifnot(identical(unname(which(optim_args$est)),unname(free))); jac
      },
      opts=list(algorithm="NLOPT_LD_SLSQP",xtol_rel=1e-8,maxeval=3000)))
}

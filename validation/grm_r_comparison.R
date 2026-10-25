# ONE independent R mirt GRM fit, only after the successful actual no-fit preflight.
args <- commandArgs(TRUE)
stopifnot(length(args)==3L,args[3]=="EXECUTE_GRM_ONCE",requireNamespace("mirt",quietly=TRUE),requireNamespace("nloptr",quietly=TRUE))
suppressPackageStartupMessages(library(mirt))
stopifnot(as.character(packageVersion("mirt"))=="1.48")
pre <- args[1];out <- args[2];set.seed(20261024)
checks <- read.csv(file.path(pre,"R_checks.csv"))
stopifnot(nrow(checks)>0,all(checks$checked))
data <- read.csv(file.path(pre,"data.csv"),check.names=FALSE)
stopifnot(identical(dim(data),c(500L,8L)),all(vapply(data,function(x) identical(sort(unique(x)),0:3),logical(1))))
sv <- read.csv(file.path(pre,"GRM_configured_layout.csv"),check.names=FALSE)
constraints <- read.csv(file.path(pre,"GRM_constraints.csv"),check.names=FALSE)
grid <- read.csv(file.path(pre,"integration.csv"));evaluation <- read.csv(file.path(pre,"evaluation_grid.csv"))
seed <- read.csv(file.path(pre,"GRM_physical.csv"),check.names=FALSE)
free <- which(sv$est);labels <- paste(sv$item[free],sv$name[free],sv$parnum[free],sep=":")
stopifnot(identical(as.numeric(sv$parnum),as.numeric(seq_len(nrow(sv)))),length(free)==32,
  identical(names(constraints),c(labels,"rhs")),all(sv$prior.type=="none"),
  all(!sv$est[sv$name %in% c("MEAN_1","COV_11")]),
  sv$value[sv$name=="MEAN_1"]==0,sv$value[sv$name=="COV_11"]==1,
  nrow(grid)==61,all(grid$weight>0),isTRUE(all.equal(sum(grid$weight),1)))
A <- as.matrix(constraints[,labels,drop=FALSE]);rhs <- constraints$rhs
stopifnot(all(as.numeric(A%*%sv$value[free])-rhs<=0))
# Confirm complete common conventional starts in native coordinates.
for(i in seq_len(8L)) {
  b <- as.numeric(unlist(seed[i,c("b1","b2","b3")],use.names=FALSE))
  rows <- which(sv$item==colnames(data)[i])
  stopifnot(identical(seed$item[i],colnames(data)[i]),
    identical(sv$name[rows],c("a1","d1","d2","d3")),
    isTRUE(all.equal(sv$value[rows],c(seed$a[i],-seed$a[i]*b))))
}
# Finite intercept boxes are redundant consequences of a<=5 and |b|<=6.
lower <- rep(-30,length(free));upper <- rep(30,length(free))
lower[sv$name[free]=="a1"] <- .1;upper[sv$name[free]=="a1"] <- 5
constraint <- function(p,optim_args) {
  stopifnot(identical(unname(which(optim_args$est)),free))
  as.numeric(A%*%p)-rhs
}
jacobian <- function(p,optim_args) {
  stopifnot(identical(unname(which(optim_args$est)),free));A
}
write.csv(sv,file.path(out,"R_starting_values.csv"),row.names=FALSE)
write.csv(data.frame(label=labels,lower=lower,upper=upper),file.path(out,"R_optimizer_boxes.csv"),row.names=FALSE)
if(file.exists(file.path(out,"R_fit_invoked"))) stop("Refit forbidden")
stopifnot(file.create(file.path(out,"R_fit_invoked")))
# This is the ONLY estimation call; no retry, post-hoc initialization or other family.
fit <- mirt(data,1,itemtype="graded",pars=sv,method="EM",quadpts=61,
  TOL=1e-8,accelerate="none",SE=FALSE,verbose=FALSE,optimizer="nloptr",
  technical=list(NCYCLES=3000,customTheta=matrix(grid$theta,ncol=1),
    customPriorFun=function(Theta,Etable) grid$weight,storeEMhistory=TRUE,storeEtable=TRUE),
  nloptr_args=list(lb=lower,ub=upper,eval_g_ineq=constraint,eval_jac_g_ineq=jacobian,
    opts=list(algorithm="NLOPT_LD_SLSQP",xtol_rel=1e-8,maxeval=3000,print_level=1)))
capture.output(sessionInfo(),file=file.path(out,"R_sessionInfo.txt"))
source(file.path(out,"pr63_optimizer_trace.R"));write_optimizer_trace(fit,out)
actual_theta <- as.numeric(fit@Internals$Theta);actual_weights <- as.numeric(unlist(extract.mirt(fit,"Prior")))
write.csv(data.frame(theta=actual_theta,weight=actual_weights),file.path(out,"R_actual_integration.csv"),row.names=FALSE)
stopifnot(isTRUE(all.equal(actual_theta,grid$theta)),isTRUE(all.equal(actual_weights,grid$weight)),extract.mirt(fit,"logPrior")==0)
coeff <- coef(fit,simplify=TRUE,IRTpars=FALSE)
stopifnot(as.numeric(coeff$means)==0,as.numeric(coeff$cov)==1)
raw <- coeff$items
parameters <- data.frame(item=colnames(data),a=raw[,"a1"],b1=-raw[,"d1"]/raw[,"a1"],
  b2=-raw[,"d2"]/raw[,"a1"],b3=-raw[,"d3"]/raw[,"a1"])
write.csv(parameters,file.path(out,"R_parameters.csv"),row.names=FALSE)
write.csv(mod2values(fit),file.path(out,"R_raw_parameters.csv"),row.names=FALSE)
curves <- list()
for(i in seq_len(8L)) {
  item <- extract.item(fit,i)
  P <- probtrace(item,matrix(evaluation$theta,ncol=1));I <- iteminfo(item,matrix(evaluation$theta,ncol=1))
  for(c in 0:3) curves[[length(curves)+1L]] <- data.frame(item=colnames(data)[i],node=evaluation$node,
    theta=evaluation$theta,category=c,probability=P[,c+1],information=as.numeric(I))
}
write.csv(do.call(rbind,curves),file.path(out,"R_curves.csv"),row.names=FALSE)
write.csv(data.frame(engine="R mirt",version=as.character(packageVersion("mirt")),
  R_version=R.version.string,model="GRM",converged=extract.mirt(fit,"converged"),
  iterations=extract.mirt(fit,"iterations"),loglik=extract.mirt(fit,"logLik"),
  aic=extract.mirt(fit,"AIC"),bic=extract.mirt(fit,"BIC"),mean=0,variance=1,
  criterion="all native parameter changes strictly below 1e-8; cap forces false"),
  file.path(out,"R_metadata.csv"),row.names=FALSE)

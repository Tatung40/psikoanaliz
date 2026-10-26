# Independent nominal calibration using a nonsingular public R values-table chart.
args <- commandArgs(TRUE);stopifnot(length(args)==3L,args[3]=="EXECUTE_NRM_ONCE")
pre <- args[1];out <- args[2]
stopifnot(requireNamespace("mirt",quietly=TRUE),requireNamespace("nloptr",quietly=TRUE))
suppressPackageStartupMessages(library(mirt));stopifnot(as.character(packageVersion("mirt"))=="1.48")
set.seed(20261024);capture.output(sessionInfo(),file=file.path(out,"R_sessionInfo.txt"))
x <- read.csv(file.path(out,"data.csv"),check.names=FALSE)
nominal <- read.csv(file.path(out,"nominal_data.csv"),check.names=FALSE)
labels <- c("A","B","C","D");stopifnot(identical(dim(x),c(500L,8L)))
stopifnot(all(vapply(seq_len(8L),function(i) all(match(nominal[[i]],labels)-1L==x[[i]]),logical(1))))
grid <- read.csv(file.path(pre,"integration.csv"));evaluation <- read.csv(file.path(pre,"evaluation_grid.csv"))
seed <- read.csv(file.path(out,"starting_values.csv"),check.names=FALSE)
A0 <- as.matrix(seed[,paste0("A",0:3)]);D0 <- as.matrix(seed[,paste0("D",0:3)])
sv <- mirt(x,1,itemtype="nominal",pars="values",verbose=FALSE)
original_attributes <- attributes(sv);original_classes <- vapply(sv,class,character(1))
for(i in seq_len(8L)) {
  rows <- which(sv$item==names(x)[i])
  stopifnot(identical(sv$name[rows],c("a1",paste0("ak",0:3),paste0("d",0:3))))
  sv$value[rows] <- c(1,A0[i,],D0[i,])
  sv$est[rows] <- c(FALSE,FALSE,rep(TRUE,3),FALSE,rep(TRUE,3))
}
# Reuse the proven native-table initializer rather than relying on data.frame
# subassignment to retain mirt_df attribute order/representations.
capture.output(list(before=original_attributes,after=attributes(sv)),file=file.path(out,"R_initialization_attributes.txt"))
source("validation/grm_initial_values.R")
saved_values <- sv$value;saved_est <- sv$est
sv <- restore_grm_initial_values(x,sv,itemtype="nominal")
stopifnot(identical(attributes(sv),original_attributes),identical(vapply(sv,class,character(1)),original_classes),
          identical(sv$value,as.double(saved_values)),identical(sv$est,as.logical(saved_est)))
free <- which(sv$est);stopifnot(length(free)==48,all(sv$prior.type=="none"))
stopifnot(sv$value[sv$name=="MEAN_1"]==0,sv$value[sv$name=="COV_11"]==1,!any(sv$est[sv$name %in% c("MEAN_1","COV_11")]))
lower <- ifelse(grepl("^ak",sv$name[free]),-5.,-6.);upper <- -lower
stopifnot(all(sv$value[free]>=lower & sv$value[free]<=upper))
write.csv(sv,file.path(out,"R_starting_values.csv"),row.names=FALSE)
# No-fit proof: direct chart allows unordered slopes and a zero final slope.
checks <- list()
record <- function(name,truth) {
  checks[[length(checks)+1L]] <<- data.frame(check=name,checked=isTRUE(truth))
  write.csv(do.call(rbind,checks),file.path(out,"R_NRM_preflight.csv"),row.names=FALSE)
  if(!isTRUE(truth)) stop(paste("NRM preflight:",name))
}
for(probe in list(c(0,.5,1.,1.5),c(0,.7,-.3,0),c(0,-5.,5.,-2.))) {
  ds <- c(0,-.4,.2,.1)
  obj <- methods::new("nominal",par=as.numeric(c(1,probe,ds)),nfact=1L,ncat=4L)
  tt <- c(-4,-1,0,1,4);lp <- outer(tt,probe)+matrix(ds,length(tt),4,byrow=TRUE)
  P <- exp(lp-apply(lp,1,max));P <- P/rowSums(P)
  actual <- probtrace(obj,matrix(tt,ncol=1))
  mean <- as.numeric(P%*%probe);info <- rowSums(P*(matrix(probe,length(tt),4,byrow=TRUE)-mean)^2)
  record(paste("nominal probability",length(checks)),isTRUE(all.equal(unname(actual),unname(P))))
  record(paste("nominal information",length(checks)),isTRUE(all.equal(as.numeric(iteminfo(obj,matrix(tt,ncol=1))),info)))
}
record("48 free ak/d, a1=1 and baseline fixed",length(free)==48 && all(!sv$est[sv$name %in% c("a1","ak0","d0")]) && all(sv$value[sv$name=="a1"]==1))
if(file.exists(file.path(out,"R_fit_invoked"))) stop("Refit forbidden")
stopifnot(file.create(file.path(out,"R_fit_invoked")))
fit <- mirt(x,1,itemtype="nominal",pars=sv,method="EM",quadpts=61,optimizer="nloptr",TOL=1e-8,accelerate="none",SE=FALSE,verbose=FALSE,
 technical=list(NCYCLES=3000,customTheta=matrix(grid$theta,ncol=1),customPriorFun=function(Theta,Etable) grid$weight,storeEMhistory=TRUE,storeEtable=TRUE),
 nloptr_args=list(lb=as.numeric(lower),ub=as.numeric(upper),opts=list(algorithm="NLOPT_LD_LBFGS",xtol_rel=1e-10,ftol_rel=1e-12,maxeval=3000)))
source(file.path(out,"pr63_optimizer_trace.R"));write_optimizer_trace(fit,out)
raw <- mod2values(fit);stopifnot(sum(raw$est)==48L);write.csv(raw,file.path(out,"R_raw_parameters.csv"),row.names=FALSE)
AA <- DD <- matrix(0,8,4)
for(i in seq_len(8L)) {
  rows <- which(raw$item==names(x)[i]);item <- raw[rows,]
  stopifnot(item$value[item$name=="a1"]==1,identical(item$est,c(FALSE,FALSE,rep(TRUE,3),FALSE,rep(TRUE,3))))
  AA[i,] <- item$value[match(paste0("ak",0:3),item$name)];DD[i,] <- item$value[match(paste0("d",0:3),item$name)]
}
contrast <- sum(AA*A0);scale <- sqrt(sum(AA^2)*sum(A0^2))
stopifnot(scale>0,abs(contrast)>sqrt(.Machine$double.eps)*scale)
sgn <- if(contrast>0) 1 else -1;AA <- sgn*AA
p <- data.frame(item=names(x),AA,DD,check.names=FALSE);names(p) <- c("item",paste0("A",0:3),paste0("D",0:3))
write.csv(p,file.path(out,"R_parameters.csv"),row.names=FALSE)
stopifnot(isTRUE(all.equal(as.numeric(fit@Internals$Theta),grid$theta)),isTRUE(all.equal(as.numeric(unlist(extract.mirt(fit,"Prior"))),grid$weight)),extract.mirt(fit,"logPrior")==0)
cf <- coef(fit,simplify=TRUE,IRTpars=FALSE);stopifnot(as.numeric(cf$means)==0,as.numeric(cf$cov)==1)
write.csv(data.frame(engine="R mirt",version=as.character(packageVersion("mirt")),R_version=R.version.string,model="NRM",converged=extract.mirt(fit,"converged"),iterations=extract.mirt(fit,"iterations"),n_parameters=sum(raw$est),orientation=sgn,mean=as.numeric(cf$means),variance=as.numeric(cf$cov),loglik=extract.mirt(fit,"logLik"),aic=extract.mirt(fit,"AIC"),bic=extract.mirt(fit,"BIC")),file.path(out,"R_metadata.csv"),row.names=FALSE)
curves <- gh <- list();lp <- matrix(log(grid$weight),500,61,byrow=TRUE)
for(i in seq_len(8L)) {
  item <- extract.item(fit,i);P <- probtrace(item,matrix(sgn*evaluation$theta,ncol=1));I <- iteminfo(item,matrix(sgn*evaluation$theta,ncol=1))
  PG <- probtrace(item,matrix(sgn*grid$theta,ncol=1));lp <- lp+t(log(PG[,x[[i]]+1L,drop=FALSE]))
  for(c in 0:3) {
    curves[[length(curves)+1L]] <- data.frame(item=names(x)[i],node=evaluation$node,theta=evaluation$theta,category=c,probability=P[,c+1L],information=as.numeric(I))
    gh[[length(gh)+1L]] <- data.frame(Madde=names(x)[i],node=0:60,Theta=grid$theta,category=c,Olasılık=PG[,c+1L])
  }
}
write.csv(do.call(rbind,curves),file.path(out,"R_curves.csv"),row.names=FALSE)
write.csv(do.call(rbind,gh),file.path(out,"R_GH_curves.csv"),row.names=FALSE)
if(extract.mirt(fit,"converged")) {
  post <- exp(lp-apply(lp,1,max));post <- post/rowSums(post)
  means <- as.numeric(post%*%grid$theta);dev <- matrix(grid$theta,500,61,byrow=TRUE)-means
  people <- data.frame(Birey=1:500,means,sqrt(rowSums(post*dev^2)),check.names=FALSE)
  names(people) <- c("Birey","Yetenek (θ)","SH (θ)")
  write.csv(people,file.path(out,"R_persons.csv"),row.names=FALSE)
}

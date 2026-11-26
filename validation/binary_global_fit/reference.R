# Actual mirt1.48 fixed-point comparisons and independent physical-moment calculus.
suppressPackageStartupMessages(library(mirt))
stopifnot(as.character(packageVersion("mirt"))=="1.48")
dir<-"/tmp/global-reference";dat<-read.csv(file.path(dir,"data.csv"))
par<-read.csv(file.path(dir,"parameters.csv"));quad<-read.csv(file.path(dir,"GH61.csv"))
J<-ncol(dat);X<-as.matrix(dat);obs<-c(colMeans(X),crossprod(X)[lower.tri(crossprod(X))]/nrow(X))
pairs<-t(combn(J,2));sets<-c(lapply(seq_len(J),function(j)j),lapply(seq_len(nrow(pairs)),function(k)pairs[k,]))
result<-list()
for(model in unique(par$model)){
 p<-par[par$model==model,]
 type<-if(model %in% c("Rasch","common1PL","2PL"))"2PL" else model
 sv<-mirt(dat,1,itemtype=type,pars="values",verbose=FALSE)
 for(j in seq_len(J)){
  nm<-colnames(dat)[j];sv$value[sv$item==nm & sv$name=="a1"]<-p$a[j]
  sv$value[sv$item==nm & sv$name=="d"]<--p$a[j]*p$b[j]
  sv$value[sv$item==nm & sv$name=="g"]<-p$c[j];sv$value[sv$item==nm & sv$name=="u"]<-p$d[j]
 }
 sv$est[sv$name %in% c("MEAN_1","COV_11")]<-FALSE
 sv$value[sv$name=="MEAN_1"]<-0;sv$value[sv$name=="COV_11"]<-1
 if(model=="Rasch")sv$est[sv$name=="a1"]<-FALSE
 con<-if(model=="common1PL")list(sv$parnum[sv$name=="a1"]) else NULL
 obj<-mirt(dat,1,itemtype=type,pars=sv,constrain=con,TOL=NaN,SE=FALSE,verbose=FALSE)
 items<-lapply(seq_len(J),function(j)extract.item(obj,j))
 eta<-if(model=="Rasch")p$b else if(model=="common1PL")c(p$b,p$a[1]) else as.vector(t(as.matrix(p[,c("a","b","c","d")[1:switch(model,"2PL"=2,"3PL"=3,"4PL"=4)]])))
 decode<-function(e){
  if(model=="Rasch")return(cbind(a=rep(1,J),b=e,c=0,d=1))
  if(model=="common1PL")return(cbind(a=rep(tail(e,1),J),b=head(e,-1),c=0,d=1))
  k<-switch(model,"2PL"=2,"3PL"=3,"4PL"=4);v<-matrix(e,nrow=J,byrow=TRUE)
  cbind(a=v[,1],b=v[,2],c=if(k>=3)v[,3] else 0,d=if(k==4)v[,4] else 1)
 }
 curves<-function(e){
  pp<-decode(e);P<-matrix(0,nrow(quad),J)
  for(j in seq_len(J)){
   it<-items[[j]];v<-it@par;names(v)<-it@parnames
   v["a1"]<-pp[j,"a"];v["d"]<--pp[j,"a"]*pp[j,"b"]
   v["g"]<-if(pp[j,"c"]==0)-999 else qlogis(pp[j,"c"])
   v["u"]<-if(pp[j,"d"]==1)999 else qlogis(pp[j,"d"])
   it@par<-unname(v);P[,j]<-getFromNamespace("ProbTrace","mirt")(it,matrix(quad$theta,ncol=1))[,2]
  };P
 }
 means<-function(e){P<-curves(e);vapply(sets,function(s)sum(apply(P[,s,drop=FALSE],1,prod)*quad$weight),numeric(1))}
 mu<-means(eta);P<-curves(eta)
 Xi<-matrix(0,length(sets),length(sets))
 for(i in seq_along(sets))for(j in seq_along(sets)){
  u<-union(sets[[i]],sets[[j]])
  Xi[i,j]<-sum(apply(P[,u,drop=FALSE],1,prod)*quad$weight)-mu[i]*mu[j]
 }
 jac<-function(h)sapply(seq_along(eta),function(j){v<-rep(0,length(eta));v[j]<-h;(means(eta+v)-means(eta-v))/(2*h)})
 D<-jac(1e-5);D2<-jac(5e-6)
 L<-t(chol(Xi));v<-forwardsolve(L,obs-mu)
 W<-forwardsolve(L,D);W<-sweep(W,2,sqrt(colSums(W^2)),"/")
 ss<-svd(W);rank<-sum(ss$d>ss$d[1]*1e-6)
 v<-v-ss$u[,seq_len(rank),drop=FALSE]%*%crossprod(ss$u[,seq_len(rank),drop=FALSE],v)
 independent<-nrow(X)*sum(v^2)
 # Source-verified no-fit integration injection: same nodes AND weights, no EM.
 obj@Model$Theta<-matrix(quad$theta,ncol=1)
 obj@Internals$Prior<-list(quad$weight)
 ref<-M2(obj,calcNull=TRUE,quadpts=61,use_dentype_estimate=TRUE)
 result[[model]]<-list(mirt=as.list(ref[1,]),independent_R_M2=independent,independent_R_df=length(sets)-rank,
  means=mu,Xi=Xi,Jacob=D,Jacobian_step_error=max(abs(D-D2)),rank_singular=ss$d,source_R=R.version.string,version_mirt=as.character(packageVersion("mirt")))
}
jsonlite::write_json(result,file.path(dir,"R.json"),auto_unbox=TRUE,pretty=TRUE,digits=17)
cat("R_REFERENCE_SUMMARY",jsonlite::toJSON(lapply(result,function(z)list(mirt=z$mirt,independent=z$independent_R_M2,df=z$independent_R_df,Jacobian_step_error=z$Jacobian_step_error)),auto_unbox=TRUE,digits=17),"\n")

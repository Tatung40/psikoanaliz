# R mirt 1.48 independent fixed-parameter preflight / ONE ML fit.
args <- commandArgs(TRUE);stopifnot(length(args)==2)
suppressPackageStartupMessages(library(mirt));stopifnot(as.character(packageVersion('mirt'))=='1.48')
root<-args[1];mode<-args[2];stopifnot(mode %in% c('preflight','fit'))
cfg<-jsonlite::fromJSON(file.path(root,'settings.json'));out<-file.path(root,mode)
dir.create(out,showWarnings=FALSE,recursive=TRUE);set.seed(cfg$seed)
dat<-read.csv(file.path(root,'data.csv'),check.names=FALSE)
grid<-read.csv(file.path(root,'integration.csv'));nodes<-as.matrix(grid[,seq_len(cfg$q),drop=FALSE])
trace<-as.matrix(read.csv(file.path(root,'trace.csv')))
pattern<-as.matrix(cfg$pattern);J<-ncol(dat);Q<-cfg$q
syntax<-paste(vapply(seq_len(Q),function(j)paste0('F',j,' = ',paste(which(pattern[,j]!=0),collapse=',')),''),collapse='\n')
spec<-mirt.model(syntax);type<-if(cfg$model=='GRM')'graded' else '2PL'
covariance<-function(eta=0){S<-diag(Q);if(cfg$correlated)S[1,2]<-S[2,1]<-tanh(eta);S}
group<-NULL
if(cfg$correlated){
 # Public createGroup: Gaussian correlation in Fisher coordinates. The fixed
 # GH/reference-density ratio is parameter-independent; no local optimizer.
 den<-function(obj,Theta){
  stopifnot(identical(dim(Theta),dim(nodes)),max(abs(Theta-nodes))<1e-12)
  S<-covariance(obj@par[1]);if(abs(S[1,2])>=1)return(rep(1e-300,nrow(Theta)))
  logmass<-log(grid$weight)+mvtnorm::dmvnorm(Theta,sigma=S,log=TRUE)-mvtnorm::dmvnorm(Theta,log=TRUE)
  exp(logmass)
 }
 group<-createGroup(par=c(eta12=if(mode=='preflight')atanh(.25) else 0),est=TRUE,den=den,nfact=Q,standardize=FALSE)
}
base<-list(data=dat,model=spec,itemtype=type,verbose=FALSE)
if(!is.null(group))base$customGroup<-group
sv<-do.call(mirt,c(base,list(pars='values')))
native_attributes<-attributes(sv)
capture.output(dput(native_attributes),file=file.path(out,'parameter_table_attributes.txt'))
stopifnot(is.data.frame(sv),all(sv$prior.type=='none'))
for(i in seq_len(J)){
 for(j in seq_len(Q)){
  at<-which(sv$item==colnames(dat)[i] & sv$name==paste0('a',j));stopifnot(length(at)==1)
  sv$value[at]<-pattern[i,j]*1.2;sv$est[at]<-pattern[i,j]!=0
  sv$lbound[at]<-if(pattern[i,j]).1 else 0;sv$ubound[at]<-if(pattern[i,j])5 else 0
 }
 ds<-which(sv$item==colnames(dat)[i] & grepl('^d[0-9]*$',sv$name))
 stopifnot(length(ds)==if(cfg$model=='GRM')3 else 1)
 if(cfg$model=='2PL'){sv$value[ds]<-0;sv$lbound[ds]<- -6;sv$ubound[ds]<-6}
 else {
  s<-sum(pattern[i,])*1.2;sv$value[ds]<- -s*seq(-1,1,length.out=3)
  sv$lbound[ds]<- -30*sum(pattern[i,]);sv$ubound[ds]<-30*sum(pattern[i,])
 }
}
if(!cfg$correlated){
 sv$value[grepl('^MEAN_',sv$name)]<-0
 covs<-which(grepl('^COV_',sv$name))
 sv$value[covs]<-vapply(sv$name[covs],function(n){ij<-as.integer(strsplit(sub('COV_','',n),'')[[1]]);as.numeric(ij[1]==ij[2])},0.)
 sv$est[grepl('^(MEAN|COV)_',sv$name)]<-FALSE
}
if(mode=='preflight')sv$est[]<-FALSE
# Exact GRM physical bounds b=-d/sum(a) in [-6,6] and gap >=1e-6.
A<-matrix(0,0,nrow(sv))
if(cfg$model=='GRM')for(i in seq_len(J)){
 ai<-which(sv$item==colnames(dat)[i] & grepl('^a[1-3]$',sv$name))
 ds<-which(sv$item==colnames(dat)[i] & grepl('^d[0-9]+$',sv$name))
 for(d in ds){v<-numeric(nrow(sv));v[ai]<- -6;v[d]<-1;A<-rbind(A,v);v[d]<- -1;A<-rbind(A,v)}
 for(j in seq_len(length(ds)-1)){v<-numeric(nrow(sv));v[ai]<-1e-6;v[ds[j+1]]<-1;v[ds[j]]<- -1;A<-rbind(A,v)}
}
stopifnot(!nrow(A) || max(A%*%sv$value)<=0)
free<-which(sv$est);low<-sv$lbound[free];up<-sv$ubound[free]
callback<-function(p,optim_args){full<-optim_args$longpars;full[optim_args$est]<-p;as.numeric(A%*%full)}
jac<-function(p,optim_args)A[,which(optim_args$est),drop=FALSE]
technical<-list(NCYCLES=if(mode=='preflight')1 else cfg$max_iter,customTheta=nodes,storeEMhistory=TRUE)
if(!cfg$correlated)technical$customPriorFun<-function(Theta,Etable)grid$weight
opts<-c(base,list(pars=sv,method='EM',SE=FALSE,accelerate='none',TOL=cfg$tol,technical=technical))
if(mode=='fit'){
 opts$optimizer<-'nloptr';nl<-list(lb=low,ub=up,opts=list(algorithm='NLOPT_LD_SLSQP',xtol_rel=1e-8,maxeval=1000))
 if(nrow(A)){nl$eval_g_ineq<-callback;nl$eval_jac_g_ineq<-jac};opts$nloptr_args<-nl
}
stopifnot(identical(attributes(sv),native_attributes))
write.csv(sv,file.path(out,'configured_values.csv'),row.names=FALSE)
fit<-do.call(mirt,opts);final<-mod2values(fit);write.csv(final,file.path(out,'raw_parameters.csv'),row.names=FALSE)
eta<-if(cfg$correlated)final$value[final$name=='eta12'] else 0;stopifnot(length(eta)==1)
S<-covariance(eta)
prior<-as.numeric(unlist(extract.mirt(fit,'Prior')));write.csv(data.frame(weight=prior),file.path(out,'prior.csv'),row.names=FALSE)
curves<-information<-pars<-list()
for(i in seq_len(J)){
 item<-extract.item(fit,i);p<-probtrace(item,trace)
 for(c in seq_len(ncol(p)))curves[[length(curves)+1]]<-data.frame(Node=seq_len(nrow(trace))-1,Item=i-1,Category=c-1,Probability=p[,c])
 ii<-iteminfo(item,trace,multidim_matrix=TRUE)
 for(n in seq_len(nrow(trace)))for(r in seq_len(Q))for(c in seq_len(Q))information[[length(information)+1]]<-data.frame(Node=n-1,Item=i-1,Row=r-1,Column=c-1,Information=ii[[n]][r,c])
 selected<-final[final$item==colnames(dat)[i] & grepl('^(a[1-3]|d[0-9]*)$',final$name),]
 pars[[i]]<-as.data.frame(as.list(setNames(selected$value,selected$name)))
}
write.csv(do.call(rbind,curves),file.path(out,'probabilities.csv'),row.names=FALSE)
write.csv(do.call(rbind,information),file.path(out,'information.csv'),row.names=FALSE)
write.csv(do.call(rbind,pars),file.path(out,'parameters.csv'),row.names=FALSE)
if(mode=='fit'){
 allp<-lapply(seq_len(J),function(i)probtrace(extract.item(fit,i),nodes));rows<-list()
 for(i in seq_len(J))for(c in seq_len(ncol(allp[[i]])))rows[[length(rows)+1]]<-data.frame(Node=seq_len(nrow(nodes))-1,Item=i-1,Category=c-1,Probability=allp[[i]][,c])
 write.csv(do.call(rbind,rows),file.path(out,'integration_probabilities.csv'),row.names=FALSE)
 # Independent posterior summation from R item probabilities, same scoring GH
 # measure as public Python fscores. This is not claimed to be R fscores API.
 scoring<-nodes%*%chol(S);p<-lapply(seq_len(J),function(i)probtrace(extract.item(fit,i),scoring));sc<-matrix(NA,16,2*Q)
 for(n in seq_len(16)){
  ell<-log(grid$weight)
  for(i in seq_len(J))ell<-ell+log(pmax(p[[i]][,dat[n,i]+1],1e-300))
  post<-exp(ell-max(ell));post<-post/sum(post);mean<-as.numeric(crossprod(post,scoring))
  sd<-sqrt(colSums((sweep(scoring,2,mean)^2)*post));sc[n,]<-c(mean,sd)
 }
 colnames(sc)<-c(paste0('F',seq_len(Q)),paste0('SD',seq_len(Q)));write.csv(sc,file.path(out,'eap.csv'),row.names=FALSE)
}
valid<-(!nrow(A) || max(A%*%final$value)<1e-7) && min(eigen(S,symmetric=TRUE)$values)>0 && abs(tanh(eta))<1
meta<-list(version=as.character(packageVersion('mirt')),R=R.version.string,model=cfg$model,dimensions=Q,
 execution=if(mode=='fit')'CALIBRATION' else 'FIXED PARAMETERS / NO M-STEP',converged=extract.mirt(fit,'converged'),
 logLik=extract.mirt(fit,'logLik'),AIC=extract.mirt(fit,'AIC'),BIC=extract.mirt(fit,'BIC'),iterations=extract.mirt(fit,'iterations'),
 covariance=S,constraints_valid=valid,logPrior=extract.mirt(fit,'logPrior'),correlation='Fisher-coordinate Gaussian customGroup, F1-F2 only')
jsonlite::write_json(meta,file.path(out,'metadata.json'),auto_unbox=TRUE,pretty=TRUE)
capture.output(sessionInfo(),file=file.path(out,'sessionInfo.txt'))

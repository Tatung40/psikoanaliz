# Independent native R mirt 1.48: general multidimensional dichotomous CFA.
args<-commandArgs(TRUE);stopifnot(length(args)==2)
suppressPackageStartupMessages(library(mirt));stopifnot(as.character(packageVersion('mirt'))=='1.48')
root<-args[1];mode<-args[2];stopifnot(mode %in% c('preflight','fit'));set.seed(20261027)
cfg<-jsonlite::fromJSON(file.path(root,'settings.json'));dat<-read.csv(file.path(root,'data.csv'),check.names=FALSE)
pattern<-as.matrix(cfg$pattern);Q<-ncol(pattern);J<-nrow(pattern);grid<-read.csv(file.path(root,'integration.csv'));nodes<-as.matrix(grid[,seq_len(Q)]);trace<-as.matrix(read.csv(file.path(root,'trace.csv')))
out<-file.path(root,mode);dir.create(out,recursive=TRUE,showWarnings=FALSE)
pairs<-do.call(rbind,lapply(2:Q,function(r)cbind(r,seq_len(r-1))))
covariance<-function(eta){L<-diag(Q);L[pairs]<-eta;S<-tcrossprod(L);S/outer(sqrt(diag(S)),sqrt(diag(S)))}
den<-function(obj,Theta){
 stopifnot(identical(dim(Theta),dim(nodes)),max(abs(Theta-nodes))<1e-12)
 S<-covariance(obj@par);C<-chol(S);inv<-chol2inv(C)
 ratio<- -sum(log(diag(C)))-.5*rowSums((Theta%*%inv-Theta)*Theta)
 exp(log(grid$weight)+ratio)
}
eta<-rep(0,nrow(pairs));if(mode=='preflight')eta[]<-.2
names(eta)<-paste0('eta',pairs[,1],pairs[,2])
group<-createGroup(par=eta,est=rep(TRUE,length(eta)),den=den,nfact=Q,standardize=FALSE)
syntax<-paste(vapply(seq_len(Q),function(j)paste0('F',j,' = ',paste(which(pattern[,j]!=0),collapse=',')),''),collapse='\n')
base<-list(data=dat,model=mirt.model(syntax),itemtype=cfg$family,customGroup=group,verbose=FALSE)
sv<-do.call(mirt,c(base,list(pars='values')));attrs<-attributes(sv)
for(i in seq_len(J)){
 ai<-which(sv$item==colnames(dat)[i] & grepl('^a[1-3]$',sv$name));idx<-sv$item==colnames(dat)[i]
 stopifnot(length(ai)==Q)
 sv$value[ai]<-pattern[i,]*1.2;sv$est[ai]<-pattern[i,]!=0
 sv$lbound[ai]<-ifelse(pattern[i,]!=0,.1,0);sv$ubound[ai]<-ifelse(pattern[i,]!=0,5,0)
 sv$value[idx & sv$name=='d']<-0;sv$lbound[idx & sv$name=='d']<- -6;sv$ubound[idx & sv$name=='d']<-6
 sv$value[idx & sv$name=='g']<-cfg$c;sv$est[idx & sv$name=='g']<-cfg$free_c
 sv$lbound[idx & sv$name=='g']<-0;sv$ubound[idx & sv$name=='g']<-.5
 sv$value[idx & sv$name=='u']<-cfg$u;sv$est[idx & sv$name=='u']<-cfg$free_u
 sv$lbound[idx & sv$name=='u']<-.5;sv$ubound[idx & sv$name=='u']<-1
}
sv$lbound[grepl('^eta',sv$name)]<- -3;sv$ubound[grepl('^eta',sv$name)]<-3
if(mode=='preflight')sv$est[]<-FALSE
stopifnot(setequal(names(attributes(sv)),names(attrs)),all(vapply(names(attrs),function(k)identical(attr(sv,k),attrs[[k]]),logical(1))),all(sv$prior.type=='none'))
free<-which(sv$est);lower<-sv$lbound;upper<-sv$ubound;asym<-sv$name %in% c('g','u')
logit<-getFromNamespace('logit','mirt');lower[asym]<-logit(lower[asym]);upper[asym]<-logit(upper[asym])
opts<-c(base,list(pars=sv,method='EM',SE=FALSE,accelerate='none',TOL=1e-8,
 technical=list(NCYCLES=if(mode=='preflight')1 else 3000,customTheta=nodes,MAXQUAD=20000,storeEMhistory=TRUE)))
if(mode=='fit'){
 opts$optimizer<-'nloptr';opts$nloptr_args<-list(lb=lower[free],ub=upper[free],opts=list(algorithm='NLOPT_LD_SLSQP',xtol_rel=1e-8,maxeval=2000))
}
write.csv(sv,file.path(out,'configured_values.csv'),row.names=FALSE)
fit<-do.call(mirt,opts);values<-mod2values(fit);write.csv(values,file.path(out,'raw_parameters.csv'),row.names=FALSE)
jsonlite::write_json(list(converged=extract.mirt(fit,'converged'),iterations=extract.mirt(fit,'iterations'),logLik=extract.mirt(fit,'logLik'),AIC=extract.mirt(fit,'AIC'),BIC=extract.mirt(fit,'BIC')),file.path(out,'native_fit_status.json'),auto_unbox=TRUE,pretty=TRUE,digits=NA)
actual_eta<-values$value[grepl('^eta',values$name)];S<-covariance(actual_eta)
# Normalization S_ii/(sqrt(S_ii)*sqrt(S_ii)) has one-ulp roundoff.
# Preserve exact values; only this mathematical identity uses machine epsilon.
unit_diagonal<-function(x)all(abs(x-1)<=8*.Machine$double.eps)
stopifnot(unit_diagonal(1+.Machine$double.eps),!unit_diagonal(1+1e-12))
representation<-list(diagonal=sprintf('%.17g',diag(S)),max_diagonal_error=max(abs(diag(S)-1)),machine_allowance=8*.Machine$double.eps)
jsonlite::write_json(representation,file.path(out,'covariance_representation.json'),auto_unbox=TRUE,pretty=TRUE,digits=NA)
stopifnot(length(actual_eta)==nrow(pairs),all(abs(actual_eta)<=3+1e-12),unit_diagonal(diag(S)),min(eigen(S,symmetric=TRUE)$values)>0)
pars<-list();curves<-list();information<-list();integration<-list()
for(i in seq_len(J)){
 item<-extract.item(fit,i);prob<-probtrace(item,trace);info<-iteminfo(item,trace,multidim_matrix=TRUE);ip<-probtrace(item,nodes)
 curves[[i]]<-data.frame(Node=seq_len(nrow(trace))-1,Item=i-1,Probability=prob[,2])
 integration[[i]]<-data.frame(Node=seq_len(nrow(nodes))-1,Item=i-1,Probability=ip[,2])
 for(n in seq_len(nrow(trace)))for(r in seq_len(Q))for(c in seq_len(Q))information[[length(information)+1]]<-data.frame(Node=n-1,Item=i-1,Row=r-1,Column=c-1,Information=info[[n]][r,c])
 v<-values[values$item==colnames(dat)[i],];row<-as.list(c(v$value[match(paste0('a',seq_len(Q)),v$name)],v$value[match(c('d','g','u'),v$name)]))
 names(row)<-c(paste0('a',seq_len(Q)),'delta','c','u');pars[[i]]<-as.data.frame(row)
}
write.csv(do.call(rbind,pars),file.path(out,'parameters.csv'),row.names=FALSE)
write.csv(do.call(rbind,curves),file.path(out,'probabilities.csv'),row.names=FALSE)
write.csv(do.call(rbind,information),file.path(out,'information.csv'),row.names=FALSE)
write.csv(do.call(rbind,integration),file.path(out,'integration_probabilities.csv'),row.names=FALSE)
prior<-as.numeric(unlist(extract.mirt(fit,'Prior')));write.csv(data.frame(weight=prior),file.path(out,'prior.csv'),row.names=FALSE)
meta<-list(R=R.version.string,mirt=as.character(packageVersion('mirt')),family=cfg$family,converged=extract.mirt(fit,'converged'),iterations=extract.mirt(fit,'iterations'),
 logLik=extract.mirt(fit,'logLik'),AIC=extract.mirt(fit,'AIC'),BIC=extract.mirt(fit,'BIC'),logPrior=extract.mirt(fit,'logPrior'),covariance=S,eta=actual_eta,
 mapping='P=g+(u-g)*logistic(A theta+delta), D=1; normalized fixed GH density ratio',mean=rep(0,Q),identification='Unit factor variances, positive pure anchors, named CFA axes',
 constraints_valid=all(values$value[values$name=='g']<values$value[values$name=='u']) && !any(values$value[grepl('^a',values$name)]<0))
jsonlite::write_json(meta,file.path(out,'metadata.json'),auto_unbox=TRUE,pretty=TRUE,digits=NA)
capture.output(sessionInfo(),file=file.path(out,'sessionInfo.txt'))

# Independent R mirt 1.48 fits; numDeriv delta Jacobian is independent of Python.
run_wald_lord_reference <- function(root) {
  suppressPackageStartupMessages(library(mirt))
  set.seed(20261102)
  stopifnot(as.character(packageVersion('mirt'))=='1.48')
  grid <- read.csv(file.path(root,'integration.csv'))
  pd <- function(V) {
    stopifnot(is.matrix(V),nrow(V)>0,nrow(V)==ncol(V),all(is.finite(V)))
    V <- (V+t(V))/2; ev <- eigen(V,symmetric=TRUE,only.values=TRUE)$values
    stopifnot(min(ev)>0,max(ev)/min(ev)<=1e8);V
  }
  key <- function(sv,V,item,name,group=NULL) {
    ix <- sv$item==item & sv$name==name
    if(!is.null(group))ix<-ix & sv$group==group
    stopifnot(sum(ix)==1)
    label<-paste0(name,'.',sv$parnum[ix]);pos<-match(label,colnames(V))
    stopifnot(!is.na(pos));pos
  }
  emit <- function(path,x) write.csv(x,path,row.names=FALSE)
  ctrl <- list(NCYCLES=20000)
  for(case in c('dif1pl_uniform','dif2pl_mixed','dif2pl_none')) {
    out<-file.path(root,case);cfg<-jsonlite::read_json(file.path(out,'settings.json'),simplifyVector=TRUE)
    path<-file.path('tests/fixtures/irt_b10',paste0('data_',case,'.csv'))
    stopifnot(digest::digest(file=path,algo='sha256')==cfg$data_sha256)
    frame<-read.csv(path,check.names=FALSE);frame<-frame[complete.cases(frame[,cfg$items,drop=FALSE]),]
    stopifnot(nrow(frame)==cfg$used_n)
    dat<-frame[,cfg$items,drop=FALSE];grp<-factor(frame$Grup,levels=c('R','F'))
    model<-cfg$model;items<-cfg$items;anchors<-cfg$anchors;studied<-cfg$studied;k<-length(items)
    fix_a <- function(sv) {if(model=='1PL'){sv$value[sv$name=='a1']<-1;sv$est[sv$name=='a1']<-FALSE};sv}
    getval <- function(sv,item,name,group=NULL) {
      ix<-sv$item==item & sv$name==name;if(!is.null(group))ix<-ix & sv$group==group
      stopifnot(sum(ix)==1);sv$value[ix]
    }
    # One simultaneous model: existing fixed-anchor M1/M2, mean and variance free in focal.
    cache_key<-paste(cfg$data_sha256,cfg$used_n,model,paste(anchors,collapse=','),'GH61-Lord-grid61-Wald-TOL1e-10-v1')
    cache_file<-file.path(out,'R_cache_key.txt')
    if(file.exists(cache_file))stopifnot(identical(readLines(cache_file,warn=FALSE),cache_key)) else writeLines(cache_key,cache_file)
    cache<-file.path(out,'R_joint.rds')
    if(file.exists(cache))joint<-readRDS(cache) else {
      sv<-fix_a(multipleGroup(dat,1,group=grp,itemtype='2PL',pars='values',verbose=FALSE))
      joint<-multipleGroup(dat,1,group=grp,itemtype='2PL',pars=sv,
        invariance=c(anchors,'free_means','free_var'),quadpts=61,TOL=1e-10,
        SE=TRUE,SE.type='Richardson',technical=c(ctrl,list(customTheta=matrix(seq(-6,6,length.out=61),ncol=1))),verbose=FALSE)
      saveRDS(joint,cache)
    }
    sv<-mod2values(joint);V<-pd(vcov(joint));emit(file.path(out,'r_joint_raw_parameters.csv'),sv)
    write.csv(V,file.path(out,'r_joint_raw_vcov.csv'))
    rows<-list();covrows<-list()
    for(item in studied) {
      ar<-getval(sv,item,'a1','R');af<-getval(sv,item,'a1','F')
      dr<-getval(sv,item,'d','R');df<-getval(sv,item,'d','F')
      delta<-if(model=='1PL')-df+dr else c(af-ar,-df/af+dr/ar)
      J<-matrix(0,length(delta),ncol(V))
      rr<-key(sv,V,item,'d','R');ff<-key(sv,V,item,'d','F')
      J[nrow(J),rr]<-1/ar;J[nrow(J),ff]<- -1/af
      if(model=='2PL') {
        ra<-key(sv,V,item,'a1','R');fa<-key(sv,V,item,'a1','F')
        J[1,ra]<- -1;J[1,fa]<-1
        J[2,ra]<- -dr/ar^2;J[2,fa]<-df/af^2
      }
      C<-pd(J%*%V%*%t(J));stat<-as.numeric(t(delta)%*%solve(C,delta))
      row<-data.frame(Item=item,Statistic=stat,df=length(delta),p=pchisq(stat,length(delta),lower.tail=FALSE),Delta_b=tail(delta,1))
      if(model=='2PL')row$Delta_a<-delta[1]
      rows[[length(rows)+1]]<-row
      for(j in seq_along(delta))for(l in seq_along(delta))
        covrows[[length(covrows)+1]]<-data.frame(Item=item,Row=j-1,Column=l-1,Covariance=C[j,l])
    }
    emit(file.path(out,'r_wald.csv'),do.call(rbind,rows));emit(file.path(out,'r_wald_cov.csv'),do.call(rbind,covrows))
    # Separate calibrations for Lord; shared public-library GH61 grid/prior masses.
    vectors<-list();covs<-list();converged<-c(extract.mirt(joint,'converged'))
    ja<-sv$value[sv$name=='a1'];jb<-c()
    for(gg in c('R','F'))for(ii in items)jb<-c(jb,-getval(sv,ii,'d',gg)/getval(sv,ii,'a1',gg))
    jm<-sv$value[sv$group=='F' & sv$name=='MEAN_1'];js<-sqrt(sv$value[sv$group=='F' & sv$name=='COV_11'])
    interior<-c(all(ja>.05+1e-6 & ja<8-1e-6 & jb> -8+1e-6 & jb<8-1e-6) & all(jm> -4+1e-6 & jm<4-1e-6 & js>.1 & js<10))
    for(g in c('R','F')) {
      sub<-dat[grp==g,,drop=FALSE];cache<-file.path(out,paste0('R_single_',g,'.rds'))
      if(file.exists(cache))fit<-readRDS(cache) else {
        sv1<-fix_a(mirt(sub,1,itemtype='2PL',pars='values',verbose=FALSE))
        sv1$value[sv1$name=='MEAN_1']<-0;sv1$est[sv1$name=='MEAN_1']<-FALSE
        sv1$value[sv1$name=='COV_11']<-1;sv1$est[sv1$name=='COV_11']<-FALSE
        if(model=='2PL'){sv1$lbound[sv1$name=='a1']<-.1;sv1$ubound[sv1$name=='a1']<-5}
        fit<-mirt(sub,1,itemtype='2PL',pars=sv1,method='EM',quadpts=61,TOL=1e-10,SE=TRUE,SE.type='Oakes',
          technical=c(ctrl,list(customTheta=matrix(grid$theta,ncol=1),customPriorFun=function(Theta,Etable)grid$weight)),verbose=FALSE)
        saveRDS(fit,cache)
      }
      sv1<-mod2values(fit);rawV<-pd(vcov(fit));a<-sapply(items,function(i)getval(sv1,i,'a1'));d<-sapply(items,function(i)getval(sv1,i,'d'));b<- -d/a
      z<-if(model=='1PL')b else c(a,b);J<-matrix(0,length(z),ncol(rawV))
      for(i in seq_along(items)) {
        di<-key(sv1,rawV,items[i],'d');bi<-if(model=='1PL')i else k+i
        J[bi,di]<- -1/a[i]
        if(model=='2PL'){ai<-key(sv1,rawV,items[i],'a1');J[i,ai]<-1;J[bi,ai]<-d[i]/a[i]^2}
      }
      C<-pd(J%*%rawV%*%t(J));vectors[[g]]<-as.numeric(z);covs[[g]]<-C
      emit(file.path(out,paste0('r_lord_',g,'_parameters.csv')),data.frame(value=z))
      emit(file.path(out,paste0('r_lord_',g,'_cov.csv')),as.data.frame(C))
      emit(file.path(out,paste0('r_single_',g,'_raw_parameters.csv')),sv1);write.csv(rawV,file.path(out,paste0('r_single_',g,'_raw_vcov.csv')))
      converged<-c(converged,extract.mirt(fit,'converged'))
      interior<-c(interior,all(a>.1+1e-6 & a<5-1e-6 & b> -6+1e-6 & b<6-1e-6))
    }
    # Numerical Jacobian independently propagates BOTH groups and all anchor correlations.
    link <- function(v,item) {
      n<-length(vectors$R);r<-v[seq_len(n)];f<-v[n+seq_len(n)];off<-if(model=='1PL')0 else k
      ix<-match(anchors,items)+off;ii<-match(item,items)
      A<-if(model=='1PL')1 else sd(r[ix])/sd(f[ix]);B<-mean(r[ix])-A*mean(f[ix])
      if(model=='1PL')A*f[ii]+B-r[ii] else c(f[ii]/A-r[ii],A*f[k+ii]+B-r[k+ii])
    }
    block<-as.matrix(Matrix::bdiag(covs$R,covs$F));v<-c(vectors$R,vectors$F);rows<-list();covrows<-list()
    for(item in studied) {
      delta<-link(v,item);J<-numDeriv::jacobian(function(z)link(z,item),v)
      C<-pd(J%*%block%*%t(J));stat<-as.numeric(t(delta)%*%solve(C,delta))
      row<-data.frame(Item=item,Statistic=stat,df=length(delta),p=pchisq(stat,length(delta),lower.tail=FALSE),Delta_b=tail(delta,1))
      if(model=='2PL')row$Delta_a<-delta[1]
      rows[[length(rows)+1]]<-row
      for(j in seq_along(delta))for(l in seq_along(delta))covrows[[length(covrows)+1]]<-data.frame(Item=item,Row=j-1,Column=l-1,Covariance=C[j,l])
    }
    emit(file.path(out,'r_lord.csv'),do.call(rbind,rows));emit(file.path(out,'r_lord_cov.csv'),do.call(rbind,covrows))
    meta<-list(R=R.version.string,mirt=as.character(packageVersion('mirt')),data_sha256=digest::digest(file=path,algo='sha256'),
      seed=20261102,original_n=cfg$original_n,used_n=nrow(dat),converged=as.logical(converged),bounds_interior=as.logical(interior),
      limitation='Bounds compared at interior solutions; separate and simultaneous latent assumptions differ; NOT CERTIFIED')
    jsonlite::write_json(meta,file.path(out,'r_settings.json'),pretty=TRUE,auto_unbox=TRUE)
    stopifnot(all(converged),all(interior))
  }
  writeLines(c(R.version.string,as.character(packageVersion('mirt'))),file.path(root,'R_session.txt'))
}

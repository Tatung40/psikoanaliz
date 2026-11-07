# Independently written reference: direct public lmer API, no Python adapter.
# Input scores are the identical standardized CSV used by the adapter.
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==3L, requireNamespace('lme4',quietly=TRUE))
stopifnot(packageVersion('lme4')==package_version('2.1-0'))
d <- read.csv(args[1]); for(f in intersect(c('p','i','r'),names(d))) d[[f]]<-factor(d[[f]])
formulas <- list('p x i'=score~1+(1|p)+(1|i), 'p x r'=score~1+(1|p)+(1|r),
 'p x i x r'=score~1+(1|p)+(1|i)+(1|r)+(1|p:i)+(1|p:r)+(1|i:r),
 'p x (r:i)'=score~1+(1|p)+(1|i)+(1|i:r)+(1|p:i),
 'p x (i:r)'=score~1+(1|p)+(1|r)+(1|r:i)+(1|p:r))
stopifnot(args[2] %in% names(formulas))
m <- lme4::lmer(formulas[[args[2]]],d,REML=TRUE,
 control=lme4::lmerControl(optimizer='bobyqa',optCtrl=list(maxfun=100000)))
v <- as.data.frame(lme4::VarCorr(m));v<-v[is.na(v$var2),c('grp','vcov')]
names(v)<-c('source','Variance');v$source[v$source=='Residual']<-paste(intersect(c('p','i','r'),names(d)),collapse=':')
dir.create(args[3],recursive=TRUE,showWarnings=FALSE)
write.csv(v,file.path(args[3],'variance.csv'),row.names=FALSE)
counts<-vapply(intersect(c('p','i','r'),names(d)),function(f)nlevels(d[[f]]),1L)
if(args[2]=='p x (r:i)') counts['r']<-min(tapply(as.character(d$r),d$i,function(x)length(unique(x))))
if(args[2]=='p x (i:r)') counts['i']<-min(tapply(as.character(d$i),d$r,function(x)length(unique(x))))
rows<-lapply(c(1L,2L),function(multiplier) {
 target<-counts;target[names(target)!='p']<-target[names(target)!='p']*multiplier
 U<-v$Variance[v$source=='p'];R<-A<-0
 for(k in seq_len(nrow(v))) if(v$source[k]!='p') {
  facets<-strsplit(v$source[k],':',fixed=TRUE)[[1]]
  contribution<-v$Variance[k]/prod(target[setdiff(facets,'p')])
  A<-A+contribution;if('p' %in% facets) R<-R+contribution
 }
 data.frame(multiplier=multiplier,universe=U,relative_error=R,absolute_error=A,
            rho=U/(U+R),phi=U/(U+A),relative_sem=sqrt(R),absolute_sem=sqrt(A))
})
write.csv(do.call(rbind,rows),file.path(args[3],'coefficients.csv'),row.names=FALSE)
write.csv(data.frame(logLik=as.numeric(logLik(m)),AIC=AIC(m),BIC=BIC(m),
 singular=lme4::isSingular(m),convergence_messages=length(m@optinfo$conv$lme4$messages)),
 file.path(args[3],'fit.csv'),row.names=FALSE)
capture.output(sessionInfo(),file=file.path(args[3],'sessionInfo.txt'))

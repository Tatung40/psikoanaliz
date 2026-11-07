# Independent covariance reconstruction by polarization of scalar R ANOVAs.
# This is NOT the Python matrix-MoM implementation and does not use facet.
args<-commandArgs(trailingOnly=TRUE); stopifnot(length(args)==2L)
d<-read.csv(args[1]); dims<-unique(d$Dimension);stopifnot(length(dims)==2L)
w<-reshape(d,timevar='Dimension',idvar=c('p','i'),direction='wide')
w$p<-factor(w$p);w$i<-factor(w$i);n<-nlevels(w$p);j<-nlevels(w$i)
stopifnot(nrow(w)==n*j,!anyNA(w))
vc<-function(y) {
 model<-aov(y~w$p+w$i);ms<-summary(model)[[1]][,'Mean Sq']
 c(universe=(ms[1]-ms[3])/j,facet=(ms[2]-ms[3])/n,residual=ms[3])
}
x<-as.matrix(w[paste0('score.',dims)]);v1<-vc(x[,1]);v2<-vc(x[,2]);cross<-(vc(x[,1]+x[,2])-vc(x[,1]-x[,2]))/4
matrices<-lapply(seq_len(3),function(k)matrix(c(v1[k],cross[k],cross[k],v2[k]),2,2))
names(matrices)<-names(v1);dir.create(args[2],recursive=TRUE,showWarnings=FALSE)
for(k in names(matrices)) {m<-matrices[[k]];dimnames(m)<-list(dims,dims);write.csv(m,file.path(args[2],paste0(k,'.csv')))}
rows<-list()
for(nf in c(4L,8L)) for(weight in list(c(.5,.5),c(.25,.75))) {
 q<-function(M)as.numeric(t(weight)%*%M%*%weight)
 U<-q(matrices$universe);R<-q(matrices$residual)/nf;A<-q(matrices$facet+matrices$residual)/nf
 mu<-sum(colMeans(x)*weight)
 rows[[length(rows)+1]]<-data.frame(n=nf,w1=weight[1],universe=U,relative_error=R,absolute_error=A,
 rho=U/(U+R),phi=U/(U+A),relative_sem=sqrt(R),absolute_sem=sqrt(A),
 cut0=(U+mu^2)/(U+mu^2+A),cut40=(U+(mu-40)^2)/(U+(mu-40)^2+A))
}
write.csv(do.call(rbind,rows),file.path(args[2],'coefficients.csv'),row.names=FALSE)
capture.output(sessionInfo(),file=file.path(args[2],'sessionInfo.txt'))

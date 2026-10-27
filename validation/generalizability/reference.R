# Independent R stats::aov reference. Validation only, never a production engine.
# Balanced orthogonal random-effects expected-mean-square identities.
args <- commandArgs(trailingOnly=TRUE)
d <- read.csv(args[1]); design <- args[2]; out <- args[3]
dir.create(out, recursive=TRUE, showWarnings=FALSE)
if (design == 'p x i') d <- subset(d, r == 0, select=-r)
if (design == 'p x r') d <- subset(d, i == 0, select=-i)
if (design == 'p x (r:i)') d$r <- interaction(d$i, d$r, drop=TRUE)
if (design == 'p x (i:r)') d$i <- interaction(d$r, d$i, drop=TRUE)
for (f in intersect(c('p','i','r'), names(d))) d[[f]] <- factor(d[[f]])
np <- nlevels(d$p)
nested <- grepl(':', design, fixed=TRUE)
if (nested) {
  parent <- if (design == 'p x (r:i)') 'i' else 'r'
  child <- if (parent == 'i') 'r' else 'i'
  # Canonical roles A=parent, B=child; unique B labels retained.
  d$A <- d[[parent]]; d$B <- d[[child]]
  na <- nlevels(d$A); nb <- nlevels(d$B)/na
  fit <- aov(score ~ p*A + A:B, data=d)
  a <- anova(fit); m <- setNames(a[['Mean Sq']], rownames(a))
  residual <- unname(m['Residuals'])
  v <- c((m['p']-m['p:A'])/(na*nb),
         (m['A']-m['A:B']-m['p:A']+residual)/(np*nb),
         (m['A:B']-residual)/np, (m['p:A']-residual)/nb, residual)
  names(v) <- c('p',parent,paste0(child,':',parent),paste('p x',parent),design)
  rownames(a) <- c('p',parent,paste('p x',parent),paste0(child,':',parent),design)
} else if (design == 'p x i x r') {
  ni <- nlevels(d$i); nr <- nlevels(d$r)
  a <- anova(aov(score ~ p*i*r, data=d))
  m <- setNames(a[['Mean Sq']], rownames(a)); e <- unname(m['p:i:r'])
  v <- c((m['p']-m['p:i']-m['p:r']+e)/(ni*nr),
         (m['i']-m['p:i']-m['i:r']+e)/(np*nr),
         (m['r']-m['p:r']-m['i:r']+e)/(np*ni),
         (m['p:i']-e)/nr,(m['p:r']-e)/ni,(m['i:r']-e)/np,e)
  names(v) <- c('p','i','r','p x i','p x r','i x r','p x i x r')
  rownames(a) <- c(names(v), 'ZERO_DF_RESIDUAL')[seq_len(nrow(a))]
  a <- a[rownames(a) != 'ZERO_DF_RESIDUAL',,drop=FALSE]
} else {
  other <- if (design == 'p x i') 'i' else 'r'
  d$A <- d[[other]]; na <- nlevels(d$A)
  a <- anova(aov(score ~ p + A, data=d)); m <- a[['Mean Sq']]
  v <- c((m[1]-m[3])/na,(m[2]-m[3])/np,m[3])
  names(v) <- c('p',other,design); rownames(a) <- names(v)
}
write.csv(data.frame(source=rownames(a),SS=a[['Sum Sq']],df=a[['Df']],MS=a[['Mean Sq']]),
          file.path(out,'anova.csv'),row.names=FALSE)
write.csv(data.frame(source=names(v),raw_variance=unname(v),Variance=round(unname(v),4)),
          file.path(out,'variance.csv'),row.names=FALSE)
# GeneralizIT rounds components to four decimals BEFORE coefficient calculation.
w <- pmax(round(v,4),0)
rows <- list()
for (mult in c(1,2)) {
  if (nested) {
    naD <- na*mult; nbD <- nb*mult
    relative <- w[4]/naD+w[5]/(naD*nbD)
    absolute <- relative+w[2]/naD+w[3]/(naD*nbD)
  } else if (design == 'p x i x r') {
    niD <- ni*mult; nrD <- nr*mult
    relative <- w[4]/niD+w[5]/nrD+w[7]/(niD*nrD)
    absolute <- relative+w[2]/niD+w[3]/nrD+w[6]/(niD*nrD)
  } else {
    relative <- w[3]/(na*mult); absolute <- relative+w[2]/(na*mult)
  }
  rows[[length(rows)+1]] <- data.frame(multiplier=mult,universe=unname(w[1]),
      relative_error=unname(relative),absolute_error=unname(absolute),
      rho=round(unname(w[1]/(w[1]+relative)),4),phi=round(unname(w[1]/(w[1]+absolute)),4))
}
write.csv(do.call(rbind,rows),file.path(out,'coefficients.csv'),row.names=FALSE)
writeLines(capture.output(sessionInfo()),file.path(out,'sessionInfo.txt'))

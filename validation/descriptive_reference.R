# Independent native R reference for descriptive/Pearson comparison tests.
# Validation only: not imported by the application or included in user downloads.
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==2L)

# Base R; sample variance (n-1), bias-corrected skew/excess kurtosis.
# This is a transcription of descriptive definitions, not a psychometric estimator.
numeric_frame <- function(x) as.data.frame(lapply(x, function(v) suppressWarnings(as.numeric(v))))
skew <- function(v) {
 n <- length(v); if(n<3) return(NA_real_); s <- sd(v)
 if(s==0) return(0); n/((n-1)*(n-2))*sum(((v-mean(v))/s)^3)
}
kurt <- function(v) {
 n <- length(v); if(n<4) return(NA_real_); s <- sd(v)
 if(s==0) return(0)
 n*(n+1)/((n-1)*(n-2)*(n-3))*sum(((v-mean(v))/s)^4)-3*(n-1)^2/((n-2)*(n-3))
}
describe <- function(x) {
 do.call(rbind, lapply(seq_along(x), function(i) {
  v <- x[[i]]; v <- v[!is.na(v)]; n <- length(v); missing <- nrow(x)-n
  binary <- n>0 && all(v %in% c(0,1))
  setNames(data.frame(names(x)[i], n, missing, if(nrow(x)>0) 100*missing/nrow(x) else NA_real_,
    if(n) mean(v) else NA_real_, if(n) median(v) else NA_real_, sd(v), var(v),
    if(n) min(v) else NA_real_, if(n) max(v) else NA_real_, skew(v), kurt(v), binary,
    if(binary) 100*mean(v) else NA_real_, check.names=FALSE),
    c("Madde","Geçerli N","Eksik N","Eksik %","Ortalama","Medyan","SS","Varyans", "Minimum","Maksimum",
      "Çarpıklık","Basıklık (fazlalık)","0/1","Doğru yanıt %"))
 }))
}

x <- numeric_frame(read.csv(args[1], check.names=FALSE))
write.csv(describe(x), file.path(args[2], "describe_items.csv"), row.names=FALSE, na="")
x[] <- lapply(x, function(v) { v[!is.finite(v)] <- NA_real_; v })
write.csv(cor(x, use="pairwise.complete.obs", method="pearson"), file.path(args[2], "correlation.csv"))
print(sessionInfo())

suppressPackageStartupMessages({library(difR); library(jsonlite)})
a <- commandArgs(TRUE); py <- fromJSON(a[1]); d <- read.csv("tests/fixtures/irt_b10/data_dif2pl_mixed.csv")
x <- d[, py$items]; g <- d$Grup
mh <- difMH(Data = x, group = g, focal.name = "F", correct = TRUE, purify = FALSE, match = "score")
lg <- difLogistic(Data = x, group = g, focal.name = "F", type = "both", criterion = "LRT", purify = FALSE)
write(toJSON(list(mh_chi2 = unname(mh$MH), mh_alpha = unname(mh$alphaMH), mh_p = unname(mh$p.value),
                  lr = unname(lg$Logistik), lr_p = unname(lg$p.value), difR = as.character(packageVersion("difR"))), digits = NA), a[2])

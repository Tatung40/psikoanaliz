# One regression, no estimation: reproduce mirt1.48 UpdatePrepList's class guard.
test_grm_initial_values <- function(data, native, out) {
  path <- tempfile(fileext=".csv");on.exit(unlink(path))
  write.csv(native,path,row.names=FALSE)
  broken <- read.csv(path,check.names=FALSE)
  classes <- function(x) vapply(x,class,character(1))
  # Exact condition behind run40164834396's error, without attempting a fit.
  stopifnot(!all(classes(native)==classes(broken)),!inherits(broken,"mirt_df"))
  write.csv(data.frame(column=names(native),native_class=classes(native),
    CSV_class=classes(broken),native_type=vapply(native,typeof,character(1)),
    CSV_type=vapply(broken,typeof,character(1)),
    native_NaN_count=vapply(native,function(v) if(is.numeric(v)) sum(is.nan(v)) else 0L,integer(1)),
    CSV_NaN_count=vapply(broken,function(v) if(is.numeric(v)) sum(is.nan(v)) else 0L,integer(1))),
    file.path(out,"GRM_initialization_input_types.csv"),row.names=FALSE)
  stopifnot(is.nan(native$prior_1[1]),is.nan(native$prior_2[1]),
            is.na(broken$prior_1[1]),!is.nan(as.double(broken$prior_1[1])))
  restored <- restore_grm_initial_values(data,broken)
  stopifnot(identical(classes(restored),classes(native)),
    identical(attributes(restored),attributes(native)),
    identical(restored$value,as.double(broken$value)),
    identical(restored$est,as.logical(broken$est)))
  write.csv(data.frame(column=names(native),native_class=classes(native),
    CSV_class=classes(broken),restored_class=classes(restored)),
    file.path(out,"GRM_initialization_classes.csv"),row.names=FALSE)
  stopifnot(identical(restored$prior_1,native$prior_1),identical(restored$prior_2,native$prior_2))
  finite_change <- broken;finite_change$prior_1[1] <- 0
  categorical_change <- broken;categorical_change$prior.type[1] <- "norm"
  stopifnot(inherits(try(restore_grm_initial_values(data,finite_change),silent=TRUE),"try-error"),
            inherits(try(restore_grm_initial_values(data,categorical_change),silent=TRUE),"try-error"))
  TRUE
}

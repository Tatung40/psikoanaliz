# One regression, no estimation: reproduce mirt1.48 UpdatePrepList's class guard.
test_grm_initial_values <- function(data, native, out) {
  path <- tempfile(fileext=".csv");on.exit(unlink(path))
  write.csv(native,path,row.names=FALSE)
  broken <- read.csv(path,check.names=FALSE)
  classes <- function(x) vapply(x,class,character(1))
  # Exact condition behind run40164834396's error, without attempting a fit.
  stopifnot(!all(classes(native)==classes(broken)),!inherits(broken,"mirt_df"))
  restored <- restore_grm_initial_values(data,broken)
  stopifnot(identical(classes(restored),classes(native)),
    identical(attributes(restored),attributes(native)),
    identical(restored$value,as.double(broken$value)),
    identical(restored$est,as.logical(broken$est)))
  write.csv(data.frame(column=names(native),native_class=classes(native),
    CSV_class=classes(broken),restored_class=classes(restored)),
    file.path(out,"GRM_initialization_classes.csv"),row.names=FALSE)
  TRUE
}

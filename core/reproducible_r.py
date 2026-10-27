"""Conservative native-R recipes; unsupported engines are never substituted."""
import json
import math
import numpy as np
import pandas as pd
from core.reproducible import decode, Unsupported


def literal(value):
    if value is None or value is pd.NA or (isinstance(value, float) and math.isnan(value)):
        return 'NA'
    if isinstance(value, (bool, np.bool_)):
        return 'TRUE' if value else 'FALSE'
    if isinstance(value, (int, float, np.number)):
        return str(value).replace('inf', 'Inf')
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, pd.DataFrame):
        if not all(isinstance(c, str) for c in value.columns):
            raise Unsupported('R requires explicit non-string column mapping')
        columns = ', '.join(literal(value[c].tolist()) for c in value.columns)
        return 'setNames(data.frame(' + columns + ', check.names=FALSE), ' + literal(list(value.columns)) + ')'
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        return 'c(' + ', '.join(literal(v) for v in value) + ')'
    raise Unsupported(f'R literal {type(value).__name__}')


_HELPERS = r'''
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
'''


def _r_code(manifest):
    lines, analyses = [], 0
    for call in manifest['calls']:
        name, module = call['function'], call['module']
        try:
            args = decode(call['arguments'])
        except (IndexError, KeyError, Unsupported):
            return None, f'{module}.{name}: dependent opaque engine result has no verified R mapping'
        if module == 'core.descriptives' and name == 'describe_items':
            lines += ['x <- numeric_frame(' + literal(args['frame']) + ')', 'result <- describe(x)', 'print(result)',
                      'write.csv(result, "describe_items.csv", row.names=FALSE, na="")']
            analyses += 1
        elif module == 'core.descriptives' and name == 'describe_total':
            lines += ['x <- numeric_frame(' + literal(args['frame']) + ')',
                      'complete <- x[complete.cases(x), , drop=FALSE]',
                      'scores <- rowSums(complete)',
                      'print(list(n=length(scores), excluded=nrow(x)-nrow(complete), description=describe(data.frame(Total=scores))))',
                      'if(length(scores)) print(quantile(scores, c(.25,.75), type=7))',
                      'write.csv(data.frame(score=scores), "total_scores.csv", row.names=FALSE)']
            analyses += 1
        elif module == 'core.scoring' and name == 'calculate_total_scores':
            x = args['df'][args['item_columns']]
            lines += ['x <- numeric_frame(' + literal(x) + ')',
                      'scores <- rowSums(x, na.rm=TRUE)']
            if args['missing_rule'] == 'strict':
                lines += ['scores[!complete.cases(x)] <- NA_real_']
            elif args['missing_rule'] == 'allow_partial':
                lines += [f'scores[rowSums(!is.na(x)) < ceiling(ncol(x)*{literal(args["min_valid_items_ratio"])})] <- NA_real_']
            lines += ['print(scores)']
            analyses += 1
        elif module == 'core.test_statistics' and name == 'compute_frequency_distribution':
            lines += ['scores <- ' + literal(args['scores']), 'scores <- scores[!is.na(scores)]',
                      'values <- sort(unique(scores)); f <- vapply(values, function(v) sum(scores==v), integer(1))',
                      'result <- data.frame(score=values, f=f, percent=round(100*f/length(scores),2), cumulative=cumsum(f),',
                      ' cumulative_percent=round(100*cumsum(f)/length(scores),2), PR=round(100*(cumsum(f)-f/2)/length(scores),2))',
                      'print(result)', 'write.csv(result, "frequency.csv", row.names=FALSE)']
            analyses += 1
        elif module == 'core.validation' or name in ('fingerprint', 'prepare_polytomous', 'prepare', 'data_type', 'validate'):
            # Later computation calls carry their exact prepared numerical inputs.
            continue
        else:
            return None, f'{module}.{name}: equivalent complete native-R recipe is not established; no substitute analysis'
    if not analyses:
        return None, 'No supported calculation was captured'
    return '# GENERATED / UNVERIFIED: native R was not executed for these inputs.\n# Input: exact prepared data embedded below; no recoding/filter is reapplied.\n# No seed needed for these deterministic descriptive operations.\n' + _HELPERS + '\n' + '\n'.join(lines) + '\nprint(sessionInfo())\n', None


def r_code(manifest):
    try:
        return _r_code(manifest)
    except (Unsupported, TypeError, ValueError, KeyError, IndexError) as exc:
        return None, f'UNSUPPORTED native R mapping: {exc}'

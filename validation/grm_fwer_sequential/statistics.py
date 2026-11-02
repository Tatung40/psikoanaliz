"""Registered sequential CP bounds; failures cannot manufacture inflation."""
import math
import numpy as np
from scipy.stats import beta

TAIL_ALPHA = .05 / 12
LOOKS = (500, 1000, 1500)


def cp(k, n):
    if not 0 <= k <= n or n == 0:
        raise ValueError('Invalid binomial counts')
    return [float(beta.ppf(TAIL_ALPHA, k, n-k+1)) if k else 0.,
            float(beta.ppf(1-TAIL_ALPHA, k+1, n-k)) if k < n else 1.]


def condition(rows):
    n = len(rows)
    valid = [r for r in rows if r['valid']]
    rejected = sum(r['decisions']['holm_event'] for r in valid)
    failed = n-len(valid)
    lower, upper = cp(rejected, n)[0], cp(rejected+failed, n)[1]
    operational = cp(rejected+failed, n)
    control = upper <= .075
    inflation = lower > .05
    flags = (['SCOPED CONTROL (LIBERAL .075)'] if control else []) + (['SCOPED INFLATION'] if inflation else [])
    return dict(n=n, valid=len(valid), failures=failed, observed_rejections=rejected,
        operational_events=rejected+failed, operational_fwer=(rejected+failed)/n,
        operational_mcse=math.sqrt((rejected+failed)/n*(1-(rejected+failed)/n)/n),
        operational_cp=operational, missing_outcome_robust_cp=[lower, upper],
        latent_null_fwer_empirical_sensitivity=[rejected/n, (rejected+failed)/n],
        valid_only_fwer=rejected/len(valid) if valid else None,
        bh_failure_as_event=(sum(r['decisions']['bh_event'] for r in valid)+failed)/n,
        decision=' / '.join(flags) if n in LOOKS and flags else 'INCONCLUSIVE',
        classified=bool(n in LOOKS and flags), certification='NO GENERAL CERTIFICATION')


def summary(rows):
    keys=[(r['seed'],r['impact']) for r in rows]
    if len(set(keys)) != len(keys):raise ValueError('Duplicate analysis')
    out={impact:condition([r for r in rows if r['impact']==impact])
         for impact in ('minus','zero') if any(r['impact']==impact for r in rows)}
    by={k:r for k,r in zip(keys,rows)}
    pairs=[(by[s,'zero'],by[s,'minus']) for s in sorted({r['seed'] for r in rows})
           if (s,'zero') in by and (s,'minus') in by]
    differences=[int(not a['valid'] or a['decisions']['holm_event'])-
                 int(not b['valid'] or b['decisions']['holm_event']) for a,b in pairs]
    return dict(conditions=out, attempted=len(rows),
        stop_at_look=len(out)==2 and len({v['n'] for v in out.values()})==1 and all(v['classified'] for v in out.values()),
        paired_operational_difference=dict(n=len(pairs),mean=float(np.mean(differences)) if pairs else None,
            mcse=float(np.std(differences,ddof=1)/np.sqrt(len(pairs))) if len(pairs)>1 else None,
            interpretation='Descriptive zero-minus; no sequential superiority claim'))

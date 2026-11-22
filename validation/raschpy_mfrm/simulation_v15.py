"""V15: DIF/DRF Wald tables - independent identity, empirical size under the null, power (PROTOCOL amendment_2026_11_22b).
PYTHONPATH=. python -m validation.raschpy_mfrm.simulation_v15 [--null 60] [--power 30] [--workers 1]   -> simulation_v15.json
"""
import argparse, json, warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.stats import norm

import core.raschpy_mfrm as RP
from validation.raschpy_mfrm.simulation import H, SEED, simulate

N = 500; ALPHA = .05


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); q = p[o] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]; out = np.empty(n); out[o] = np.minimum(q, 1); return out


def identity(tab):
    """max |diff| of z, p and BH-adjusted p recomputed from Difference and SE (Selected items/raters define the family?)."""
    z = tab.Difference / tab.SE; p = 2 * norm.sf(np.abs(z))
    out = dict(z=float(np.abs(z - tab.z).max()), p=float(np.abs(p - tab.p).max()))
    out["p_bh"] = float(np.abs(bh(tab.p) - tab["p (corrected)"]).max())
    return out


def one(args):
    kind, rep = args; rng = np.random.default_rng([SEED, 15, rep, kind == "power"])
    d = simulate("dense", N, rng).rename(columns={"person": "person_id", "item": "item_id", "rater": "rater_id"})
    persons = d.person_id.unique(); grp = pd.Series(rng.permutation(np.repeat(["A", "B"], len(persons) // 2)), index=persons)
    d["grup"] = d.person_id.map(grp)
    if kind == "power":                                         # +0.6 logit DIF on K3 for group B (harder): resample scores
        k = np.arange(4); tau = np.array([-1.2, 0., 1.2]); out = []
        th = pd.Series(rng.standard_normal(len(persons)), index=persons); sev = {"P1": -.8, "P2": 0., "P3": .8}
        dl = {"K1": -.5, "K2": -.1, "K3": .2, "K4": .4}
        for r in d.itertuples():
            delta = dl[r.item_id] + (.6 if (r.item_id == "K3" and r.grup == "B") else 0)
            ln = k * (th[r.person_id] - delta - sev[r.rater_id]) - np.concatenate([[0], np.cumsum(tau)])
            pr = np.exp(ln - ln.max()); out.append(int(rng.choice(4, p=pr / pr.sum())))
        d["score"] = out
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = RP.fit(d, "person_id", "item_id", "rater_id", "score", "global")
        row = dict(kind=kind, rep=rep)
        for test in ("dif",) + (("drf",) if kind == "null" else ()):
            try:
                t = RP.group_test(res, d, "person_id", "grup", test, n_boot=200)
            except Exception as e:
                row[f"{test}_error"] = str(e)[:80]; continue
            tab, om = t["tablo"], t.get("omnibus")
            row[f"{test}_rej"] = int((tab.p < ALPHA).sum()); row[f"{test}_n"] = int(len(tab))
            row[f"{test}_flag_any"] = bool(tab.Flagged.any()); row[f"{test}_om_rej"] = bool((om.p < ALPHA).any()) if om is not None else None
            row[f"{test}_identity"] = identity(tab)
            if kind == "power": row["k3_flagged"] = bool(tab[tab.Item == "K3"].Flagged.any())
    return row


def summarise(rows):
    df = pd.DataFrame(rows); out = {}
    nul = df[df.kind == "null"]
    for t in ("dif", "drf"):
        if f"{t}_rej" not in nul: continue
        ok = nul[nul[f"{t}_rej"].notna()]
        rej = float(ok[f"{t}_rej"].sum() / ok[f"{t}_n"].sum()); fw = float(ok[f"{t}_flag_any"].mean()); om = float(ok[f"{t}_om_rej"].mean())
        ident = max(max(v.values()) for v in ok[f"{t}_identity"])
        out[t] = dict(replications=int(len(ok)), errors=int(nul[f"{t}_rej"].isna().sum()), per_test_size=rej, familywise_flag_rate=fw,
                      omnibus_size=om, identity_max_diff=float(ident),
                      gate_identity=bool(ident <= 1e-8), gate_size=bool(.025 <= rej <= .075 and fw <= .10),
                      gate_omnibus=bool(.025 <= om <= .075))
    pw = df[df.kind == "power"]
    if len(pw): out["dif"]["power_k3_flagged"] = float(pw.k3_flagged.mean()); out["dif"]["power_replications"] = int(len(pw))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--null", type=int, default=60); ap.add_argument("--power", type=int, default=30)
    ap.add_argument("--workers", type=int, default=1); a = ap.parse_args()
    jobs = [("null", r) for r in range(a.null)] + [("power", r) for r in range(a.power)]
    with Pool(a.workers) as pool:
        rows = list(pool.imap_unordered(one, jobs))
    res = summarise(rows)
    (H / "simulation_v15.json").write_text(json.dumps(dict(protocol="PROTOCOL.json amendment_2026_11_22b V15", seed=SEED, N=N,
                                                            null=a.null, power=a.power, results=res), indent=1) + "\n")
    print(json.dumps(res, indent=1))

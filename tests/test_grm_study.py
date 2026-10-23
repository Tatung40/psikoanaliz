from scripts import grm_dif_study as S


def test_failed_grm_fits_are_explicit_and_resuming_does_not_refit(tmp_path):
    calls = []
    def failed(*args, **kwargs):
        calls.append(1)
        raise ValueError("Model/hessian/quadrature failed")
    first, directory, computed = S.run(tmp_path, repetitions=2, fit_fn=failed)
    assert computed == 4 and len(calls) == 4
    for condition in ("null", "both"):
        result = first["results"][condition]
        assert result["total"] == result["inconclusive"] == 2
        assert result["valid"] == 0 and result["any_bh"]["estimate"] is None
        assert result["any_bh"]["trials"] == 0
    second, same, computed = S.run(tmp_path, repetitions=2, fit_fn=failed)
    assert computed == 0 and len(calls) == 4 and same == directory and first == second


def test_family_errors_and_coverage_use_independent_replication_denominators():
    records = []
    for i in range(10):
        entries = []
        for name in ("M5", "M6"):
            estimates = [dict(parameter=p, truth=0., estimate=0., lower=-.1, upper=.1) for p in ("a", "b1", "b2")]
            entries.append(dict(item=name, p=.001 if i == 0 else .9, q=.002 if i == 0 else .9,
                                threshold_q=.02, slope_q=.8, parameters=estimates, effect=estimates[0]))
        records.append(dict(condition="null", status="valid", items=entries, mu=-.4, sd=1.2))
    records.append(dict(condition="null", status="inconclusive", items=[]))
    result = S.summarize(records)["null"]
    assert result["total"] == 11 and result["valid"] == 10 and result["inconclusive"] == 1
    assert result["any_bh"]["events"] == 1 and result["any_bh"]["trials"] == 10
    assert result["items"]["M5"]["signed_effect"]["pointwise_coverage"]["events"] == 10
    assert result["items"]["M6"]["threshold_bh"]["events"] == 10

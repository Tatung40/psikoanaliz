"""Simulation bookkeeping must never count an inconclusive fit as no DIF."""
import numpy as np
import pytest
from scripts import dif_fixed_anchor_simulation as S


def test_exact_monte_carlo_interval_includes_zero_count_uncertainty():
    zero = S.binomial(0, 10)
    assert zero["mc_ci95"] == pytest.approx([0., 1 - .025 ** .1])
    assert S.binomial(10, 10)["mc_ci95"] == pytest.approx([.025 ** .1, 1.])
    assert S.binomial(0, 0)["estimate"] is None


def test_seeded_dgp_keeps_known_anchors_and_latent_impact():
    x, groups, truth = S.generate(72, "dif", 0, 800, 500)
    repeated, _, again = S.generate(72, "dif", 0, 800, 500)
    np.testing.assert_array_equal(x, repeated)
    assert truth == again and set(np.unique(x)) == {0, 1}
    assert groups.count("R") == 800 and groups.count("F") == 500
    assert truth["M3"] == pytest.approx({"a": 0., "b": .6})
    assert truth["M7"] == pytest.approx({"a": -.9, "b": 0.})
    assert not set(S.TARGETS) & set(S.ANCHORS)
    _, _, null_truth = S.generate(72, "null", 0)
    assert all(value == 0. for item in null_truth.values() for value in item.values())


def test_failure_denominators_and_cache_reuse_are_explicit(tmp_path):
    calls = []
    def failed(*args, **kwargs):
        calls.append(1)
        raise ValueError("Synthetic fit failure")
    first, folder, computed = S.run(tmp_path, repetitions=2, fit_fn=failed)
    assert computed == 4 and len(calls) == 4
    for condition in ("null", "dif"):
        result = first["results"][condition]
        assert result["inconclusive_replicates"] == result["total_replicates"] == 2
        assert result["any_bh_rejection"]["trials"] == 0
        assert result["any_bh_rejection"]["estimate"] is None
    second, same_folder, computed = S.run(tmp_path, repetitions=2, fit_fn=failed)
    assert second == first and same_folder == folder and computed == 0 and len(calls) == 4
    assert not list(folder.glob("*.tmp"))


def test_summary_uses_replicates_for_any_rejection_and_pointwise_coverage():
    records = []
    for i in range(10):
        items = [{"item": item, "p": .001 if i == 0 else .9, "q": .002 if i == 0 else .9,
                  "a": {"estimate": 0., "truth": 0., "lower": -.1, "upper": .1},
                  "b": {"estimate": .2, "truth": 0., "lower": .1, "upper": .3}}
                 for item in S.TARGETS]
        records.append(dict(condition="null", status="valid", items=items, mu_f=-.4, sd_f=1.2))
    result = S.summarize(records)["null"]
    assert result["any_bh_rejection"]["events"] == 1 and result["any_bh_rejection"]["trials"] == 10
    effect = result["items"]["M3"]["parameter_contrasts"]
    assert effect["a"]["pointwise_ci_coverage"]["events"] == 10
    assert effect["b"]["pointwise_ci_coverage"]["events"] == 0
    assert effect["b"]["bias"] == pytest.approx(.2)
    assert effect["b"]["rmse"] == pytest.approx(.2)

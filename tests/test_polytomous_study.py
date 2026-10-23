from pathlib import Path
from scripts.polytomous_dif_study import run, summarize, known_effects
from scripts.polytomous_dif_validation import generate


def test_failed_repetitions_counted_and_cache_reused(tmp_path):
    calls = []
    def failed(*args, **kwargs):
        calls.append(1)
        raise ValueError("no valid fit")
    first = run("PCM", tmp_path, 2, fit_fn=failed)
    second = run("PCM", tmp_path, 2, fit_fn=lambda *a, **k: (_ for _ in ()).throw(AssertionError("refit")))
    assert len(calls) == 4 and first["computed"] == 4 and second["reused"] == 4
    for c in first["conditions"].values():
        assert c["total"] == c["inconclusive"] == 2 and c["valid"] == 0
        assert c["any_bh"] == {} and c["items"]["M5"]["raw_rejection"]["trials"] == 0


def test_nominal_effects_sum_zero_and_null_has_no_direction():
    for condition in ("null", "both"):
        effects = known_effects("NRM", generate("NRM", condition=condition)[2])
        assert all(abs(v.sum()) < 1e-12 for v in effects.values())
        if condition == "null": assert all(abs(v).max() < 1e-12 for v in effects.values())

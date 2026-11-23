"""V15 checkpoint recovery: deterministic jobs survive interrupted execution."""
import json
from pathlib import Path

from validation.raschpy_mfrm import simulation_v15 as v15


class InlinePool:
    def __init__(self, workers):
        self.workers = workers
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False
    def imap_unordered(self, fn, jobs):
        for job in jobs:
            yield fn(job)


def test_checkpoint_resume_does_not_repeat_completed_jobs(tmp_path, monkeypatch):
    monkeypatch.setattr(v15, "H", tmp_path)
    monkeypatch.setattr(v15, "Pool", InlinePool)
    seen = []
    def fake_one(job):
        seen.append(job)
        kind, rep = job
        row = {"kind": kind, "rep": rep}
        for test in ("dif", "drf"):
            row.update({f"{test}_rej": 1, f"{test}_n": 20,
                        f"{test}_flag_any": False, f"{test}_om_rej": False,
                        f"{test}_identity": {"z": 0., "p": 0., "p_bh": 0.}})
        row["k3_flagged"] = True
        return row
    monkeypatch.setattr(v15, "one", fake_one)
    ck = tmp_path / "checkpoint.jsonl"
    ck.write_text(json.dumps(fake_one(("null", 0))) + "\n")
    seen.clear()
    out = v15.run_checkpointed(null=2, power=1, workers=1, checkpoint=ck)
    assert seen == [("null", 1), ("power", 0)]
    assert out["dif"]["replications"] == 2
    assert (tmp_path / "simulation_v15.json").exists()
    seen.clear()
    v15.run_checkpointed(null=2, power=1, workers=1, checkpoint=ck)
    assert seen == []

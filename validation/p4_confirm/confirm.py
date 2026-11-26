import json, os, sys
os.environ.setdefault("P4_ENGINE", "adapter"); os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from multiprocessing import Pool
sys.path.insert(0, "validation")
import polytomous_mc_study as S
MODELS = ("GRM", "PCM", "GPCM", "NRM")
if __name__ == "__main__":
    shard, step, R, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    jobs = [(m, 2000, 920000000 + 100000 * i + r) for i, m in enumerate(MODELS) for r in range(R) if r % step == shard]
    with Pool(os.cpu_count()) as p, open(out, "w") as fh:
        for rec in p.imap_unordered(S.fit_job, jobs): fh.write(json.dumps(rec, default=float) + "\n")

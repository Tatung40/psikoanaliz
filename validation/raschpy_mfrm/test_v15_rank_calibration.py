"""Fast, deterministic exact rank-p calibration checks (no RaschPy fitting)."""
import unittest
import numpy as np

def rank_p(calibration, observed):
    cal=np.asarray(calibration,dtype=float)
    if not len(cal) or not np.isfinite(cal).all() or not np.isfinite(observed):
        raise ValueError("finite nonempty calibration and observed LR required")
    return (1+int(np.count_nonzero(cal>=observed)))/(len(cal)+1)

class RankPTests(unittest.TestCase):
    def test_exact_boundary_m300(self):
        cal=np.arange(300,dtype=float)
        self.assertLess(rank_p(cal,286.),.05)  # 15/301
        self.assertGreaterEqual(rank_p(cal,285.),.05) # 16/301

    def test_ties_are_conservative(self):
        self.assertEqual(rank_p([1.,2.,2.,3.],2.),4/5)

    def test_invalid_rejected(self):
        for cal,obs in [([],1.),([1.,float("nan")],2.),([1.],float("inf"))]:
            with self.assertRaises(ValueError):
                rank_p(cal,obs)

    def test_exchangeable_null_rank_size(self):
        # Exhaustively enumerate every possible observed rank under distinct exchangeable LR.
        cal_n=300
        p=np.arange(1,cal_n+2)/(cal_n+1)
        self.assertEqual(int(np.sum(p<.05)),15)
        self.assertLessEqual(np.mean(p<.05),.05)

if __name__=="__main__":
    unittest.main()

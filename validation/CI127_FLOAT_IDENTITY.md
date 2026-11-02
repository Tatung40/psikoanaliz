# PR127 CI: canonical bytes versus reconstructed floating-point identity

Failure run41371973870 / job125044169290:660 passed,2 failed,14 skipped. Only these two cached tests failed; no optimizer/MC was rerun for diagnosis.

CI reports Python3.11.17, NumPy2.4.6, SciPy1.17.1, pandas3.1.0 on ubuntu-26.04. Local Python/package versions are identical; local libc is glibc2.41. The CI log does not establish its exact libc/CPU kernel, so no claim is made that a particular SIMD implementation caused the difference. Disabling local AVX features did not reproduce the CI environment.

## Exact proof before changing tests

The archived JSON initial vectors rehashed to their ORIGINAL IDs exactly. For each of the three differing CI IDs, changing exactly ONE encoded log-gap coordinate by ONE binary64 ULP reproduced the entire reported CI SHA256. Decoding both vectors gave identical physical parameters; fixed-GH161 logLik and analytic gradient also matched exactly. This is a reconstructed transcendental-coordinate byte-identity problem, not changed response data, corrupted JSON or materially different calculation.

```json
[
  {
    "slot": "P09.Hb",
    "canonical_sha256": "653a1e118791fea826635b641cb10de9ed5450e615b726a667f63ee1c73c8d38",
    "CI_reconstructed_sha256": "f06fc46c7b014d0c84998f279f951acdcd8f9e0b7b839209754612f0f91565f5",
    "coordinate": "P01.shared.log_gap_b3",
    "canonical_hex": "0x1.42ba15b925316p-6",
    "CI_hex": "0x1.42ba15b925317p-6",
    "max_ULP": 1,
    "absolute_encoded_difference": 3.469446951953614e-18,
    "physical_max_difference": 0.0,
    "GH161_loglik_difference": 0.0,
    "GH161_gradient_max_difference": 0.0
  },
  {
    "slot": "P10.H0",
    "canonical_sha256": "d76e4ca3176c317ccdc41f73aa049e8ba65e809bc16b91fea2a6eedd12e9cbf6",
    "CI_reconstructed_sha256": "cdac1bab4f4b2bf9e0e103e63b648dc24503b00f5e7f37aa939d62df8dcae3d7",
    "coordinate": "P08.shared.log_gap_b3",
    "canonical_hex": "-0x1.9fbe6a948c7aep-4",
    "CI_hex": "-0x1.9fbe6a948c7adp-4",
    "max_ULP": 1,
    "absolute_encoded_difference": 1.3877787807814457e-17,
    "physical_max_difference": 0.0,
    "GH161_loglik_difference": 0.0,
    "GH161_gradient_max_difference": 0.0
  },
  {
    "slot": "P10.Ha",
    "canonical_sha256": "a4f6c91ff88ef5282171a2c4642c31feb323074c6f8f0bea9a6943256cb303d6",
    "CI_reconstructed_sha256": "f0206f959cdc96ad29e9c5b6ef8b94ac2c4d25bd5c19a5807ae9708ecd891592",
    "coordinate": "P09.focal.log_gap_b3",
    "canonical_hex": "0x1.d082c1723cae0p-7",
    "CI_hex": "0x1.d082c1723cae1p-7",
    "max_ULP": 1,
    "absolute_encoded_difference": 1.734723475976807e-18,
    "physical_max_difference": 0.0,
    "GH161_loglik_difference": 0.0,
    "GH161_gradient_max_difference": 0.0
  }
]
```

The second error is seed2026110101/impact−.5, P05.shared.log_gap_b2, index18: archived0x1.6ffae67c2e8dbp-4 versus CI0x1.6ffae67c2e8dap-4;1ULP=1.3877787807814457e−17. Slopes and cold latent mean/logSD remain exact.

## Minimal correction and retained safeguards

- Model IDs are validated against ORIGINAL archived role+initial-vector bytes, then bound by exact hypothesis/role semantics. Runtime hashes of recomputed log values are not treated as universal model identities. No reference ID or archived vector is updated.
- Only recomputed log coordinates receive a maximum1ULP representation comparison. Copied parameters, slopes, means and cold latent coordinates remain exact. Regression assertions reject2ULP changes and any slope drift. Scientific1e−5 gradient,1e−3 logLik, information/bounds/LR gates are untouched.
- For replay of the frozen independent auditor, a read-only canonical-encoding adapter first executes the independent frequency-count reconstruction and verifies1ULP equivalence. Only then does it return the original archived encoding for EXACT original SHA/byte checks. It patches no numerical/statistical assertions and edits no historical audit source or artifact.
- The observed maximum likelihood-rounding discrepancy is checked against the ORIGINAL auditor1e−8 bound on both platforms, instead of requiring the magnitude of floating-point error itself to be identical across platforms. Native likelihood/gate checks are still executed by the unchanged auditor.

Original core/validation files snapshotted before the fix: **722; modified:0**, including all raw archives,30-pair results/snapshots, protocols, historical code and hash references. Only the two failing tests and a new read-only comparison helper change. New diagnostics are this report.

Targeted execution: exactly the two failed tests, **2 passed in2.83s**, including full cached auditor replay. No MC, optimization, parameter estimation or full local regression. Existing mandatory CI runs on the updated PR; scientific results/status labels remain unchanged. Do not merge.

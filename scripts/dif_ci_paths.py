"""Choose relevant formal DIF PR gates; final main always runs the whole suite."""
import os
from pathlib import Path
import subprocess


def choose(paths):
    shared = {"core/rasch.py", "core/rasch_dif.py", "core/irt_dif.py", "app.py", "scripts/dif_ci_paths.py"}
    binary_prefixes = ("core/irt_dif", "core/irt_mg", "ui/irt_b10", "ui/irt_b11", "ui/rasch",
                       "tests/test_dif_", "tests/test_dif34_", "tests/test_irt_b10", "tests/test_irt_b11", "tests/fixtures/irt_b10/", "tests/fixtures/irt_b11/")
    grm_prefixes = ("core/grm_", "ui/grm_", "reports/grm_", "scripts/grm_", "tests/test_grm_", "tests/fixtures/grm_dif/")
    paths = set(paths)
    return dict(binary=any(p.startswith(binary_prefixes) or p in shared for p in paths),
                grm=any(p.startswith(grm_prefixes) or p in shared or p in
                        {"app.py", "scripts/dif_ci_paths.py", ".github/workflows/pytest.yml"} for p in paths))


def choose_polytomous(paths):
    shared = {"core/grm_dif.py", "core/rasch.py", "core/rasch_dif.py", "core/irt_dif.py",
              "app.py", "scripts/dif_ci_paths.py", ".github/workflows/pytest.yml"}
    prefixes = ("core/polytomous_dif", "ui/polytomous_dif", "reports/polytomous_dif",
                "scripts/polytomous_dif", "scripts/rsm_dif_reference", "tests/test_polytomous_", "tests/fixtures/polytomous_dif/")
    return any(p in shared or p.startswith(prefixes) for p in paths)


if __name__ == "__main__":
    # pull_request checkout is GitHub's merge commit; its first parent is the tested base.
    paths = subprocess.check_output(["git", "diff", "--name-only", "HEAD^1", "HEAD"], text=True).splitlines()
    flags = dict(**choose(paths), polytomous=choose_polytomous(paths))
    output = "".join(f"{name}={str(value).lower()}\n" for name, value in flags.items())
    if os.environ.get("GITHUB_OUTPUT"):
        with Path(os.environ["GITHUB_OUTPUT"]).open("a") as stream: stream.write(output)
    print(output, end="")

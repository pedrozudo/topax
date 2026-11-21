import csv
from pathlib import Path
import runpy


ROOT = Path(__file__).parents[2]
SIGMUND = (
    ROOT
    / "benchmarks"
    / "benchmark_scripts"
    / "topax"
    / "sigmund_2007"
    / "mbb_conv_filter.py"
)
LAZAROV = (
    ROOT
    / "benchmarks"
    / "benchmark_scripts"
    / "topax"
    / "lazarov_2011"
    / "cantilever_helmholtz_density.py"
)


def read_single_row(path):
    with path.open(newline="") as stream:
        return next(csv.DictReader(stream))


def test_sigmund_writes_history_to_requested_runtime_csv(tmp_path):
    namespace = runpy.run_path(SIGMUND)
    output = tmp_path / "runtime.csv"
    row = {
        "iteration": 1,
        "objective": 2.0,
        "physics_cg_iterations": 7,
        "iteration_time_s": 0.5,
    }

    namespace["save_history"]([row], output)

    assert read_single_row(output) == {
        "iteration": "1",
        "objective": "2.0",
        "physics_cg_iterations": "7",
        "iteration_time_s": "0.5",
    }
    args = namespace["parse_args"](
        ["--runtime-csv", str(output), "--max-iterations", "10"]
    )
    assert args.runtime_csv == output
    assert args.max_iterations == 10


def test_lazarov_writes_history_to_requested_runtime_csv(tmp_path):
    namespace = runpy.run_path(LAZAROV)
    output = tmp_path / "runtime.csv"
    row = {
        "iteration": 1,
        "comparison_compliance": 2.0,
        "physics_cg_iterations": 7,
        "iteration_time_s": 0.5,
    }

    namespace["save_case_history"](namespace["CASES"][0], [row], output)

    assert read_single_row(output) == {
        "iteration": "1",
        "comparison_compliance": "2.0",
        "physics_cg_iterations": "7",
        "iteration_time_s": "0.5",
    }
    args = namespace["parse_args"](
        [
            "--case",
            "400x100",
            "--runtime-csv",
            str(output),
            "--max-iterations",
            "10",
        ]
    )
    assert args.runtime_csv == output
    assert args.max_iterations == 10

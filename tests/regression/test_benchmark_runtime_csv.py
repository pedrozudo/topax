import csv
from pathlib import Path
import runpy

import pytest


BENCHMARKS = (
    Path(__file__).parents[2]
    / "benchmarks"
    / "benchmark_scripts"
    / "topax"
    / "traeff_2023"
    / "cantilever.py",
    Path(__file__).parents[2]
    / "benchmarks"
    / "benchmark_scripts"
    / "topax"
    / "wang_2025"
    / "cantilever.py",
)


@pytest.mark.parametrize("benchmark", BENCHMARKS)
def test_benchmark_writes_runtime_csv(benchmark, tmp_path):
    namespace = runpy.run_path(benchmark)
    fields = namespace["_RUNTIME_FIELDS"]
    row = {
        "iteration": 1,
        "compliance": 2.0,
        "mean_design_density": 0.1,
        "mean_physical_density": "",
        "volume_constraint": 0.0,
        "non_discreteness_reported": 0.25,
        "change": 0.2,
        "physics_cg_iterations": 7,
        "physics_relative_residual": 1.0e-5,
        "filter_cg_iterations": "",
        "filter_relative_residual": "",
        "iteration_time_s": 0.5,
    }
    output = tmp_path / "runtime.csv"

    namespace["_write_runtime_csv"](output, [row])

    with output.open(newline="") as stream:
        reader = csv.DictReader(stream)
        assert tuple(reader.fieldnames) == fields
        written = list(reader)
    assert len(written) == 1
    assert written[0]["iteration"] == "1"
    assert written[0]["physics_cg_iterations"] == "7"
    assert written[0]["iteration_time_s"] == "0.5"

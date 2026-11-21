import csv
from pathlib import Path
import subprocess

import pytest


COMMON = (
    Path(__file__).parents[2]
    / "benchmarks"
    / "benchmark_scripts"
    / "topax"
    / "slurm"
    / "common.sh"
)


def test_memory_csv_summarizes_host_and_device_samples(tmp_path):
    runtime = tmp_path / "runtime.csv"
    runtime.write_text("iteration,iteration_time_s\n1,0.5\n2,0.4\n")
    time_file = tmp_path / "time.txt"
    time_file.write_text("Maximum resident set size (kbytes): 2097152\n")
    gpu = tmp_path / "gpu.csv"
    gpu.write_text("timestamp_epoch_s,used_mib\n1.0,1024\n2.0,3072\n")
    output = tmp_path / "memory.csv"

    subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; topax_write_memory_csv "$2" "$3" "$4" "$5" 1024',
            "bash",
            str(COMMON),
            str(output),
            str(runtime),
            str(time_file),
            str(gpu),
        ],
        check=True,
    )

    with output.open(newline="") as stream:
        row = next(csv.DictReader(stream))
    assert int(row["design_iterations_completed"]) == 2
    assert float(row["host_peak_rss_gib"]) == pytest.approx(2.0)
    assert float(row["gpu_baseline_gib"]) == pytest.approx(1.0)
    assert float(row["gpu_peak_gib"]) == pytest.approx(3.0)
    assert float(row["gpu_peak_above_baseline_gib"]) == pytest.approx(2.0)

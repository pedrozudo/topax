from pathlib import Path
import runpy

import numpy as np


EXAMPLE = Path(__file__).parents[2] / "examples" / "compliance_3d.py"


def test_small_3d_compliance_optimization_runs_end_to_end():
    run = runpy.run_path(EXAMPLE)["run"]

    result = run(
        elements=(2, 2, 2),
        n_levels=2,
        iterations=2,
        volume_fraction=0.5,
        verbose=False,
    )

    assert float(result["compliance"]) > 0.0
    assert np.isfinite(result["compliance"])
    assert np.isfinite(result["volume_constraint"])

    density = np.asarray(result["density"])
    compliance_gradient = np.asarray(result["compliance_gradient"])
    constraint_gradient = np.asarray(result["constraint_gradient"])

    assert density.shape == (2, 2, 2)
    assert np.all((density >= 0.0) & (density <= 1.0))
    assert not np.allclose(density, 0.5)

    assert compliance_gradient.shape == (2, 2, 2)
    assert constraint_gradient.shape == (2, 2, 2)
    assert np.all(np.isfinite(compliance_gradient))
    np.testing.assert_allclose(constraint_gradient, 1.0 / 8.0)

    assert len(result["history"]) == 2
    assert all(np.isfinite(entry["compliance"]) for entry in result["history"])

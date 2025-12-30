import numpy as np


def test_MATLAB(data_dir, simulator):
    # Compare forces computed by our code with those from MATLAB implementation
    
    pts = np.loadtxt(data_dir / "init_pts.csv", delimiter=',')
    simulator.update_positions(pts)

    # Build the diagram
    diag = simulator.build()

    # Get forces
    forces = diag["forces"]

    # This is results of the MATLAB implementation for the points in 'init_pts.csv' file
    forces_matlab = np.loadtxt(data_dir / "init_forces.csv", delimiter=',')
    F_comp = np.abs(forces - forces_matlab)

    max_err = float(np.max(F_comp))
    tolerance = 1.0e-8

    assert max_err < tolerance, (
        f"Force mismatch with MATLAB implementation!\n"
        f"Max difference: {max_err:.3e} (tolerance = {tolerance})"
    )

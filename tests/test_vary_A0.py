import numpy as np


def test_MATLAB_varying_A0(data_dir, simulator):
    # Compare forces computed by our code with those from MATLAB implementation
    # Now with varying preferred areas A0
    # temporarily change delta to 0.0 for this test
    original_phys = simulator.phys
    new_phys = original_phys.replace(delta=0.0)

    assert np.max(simulator.preferred_areas - original_phys.A0) < 1.0e-12, "Preferred areas not initialized correctly."

    simulator.update_params(new_phys)

    pts = np.loadtxt(data_dir / "init_pts.csv", delimiter=',')
    A0_list = np.loadtxt(data_dir / "varying_A0.csv", delimiter=',')

    simulator.update_positions(pts, A0_list)

    # Build the diagram
    diag = simulator.build()

    # Get forces
    forces = diag["forces"]

    # This is result of the MATLAB implementation with varying A0
    forces_matlab = np.loadtxt(data_dir / "varying_A0_forces.csv", delimiter=',')
    F_comp = np.abs(forces - forces_matlab)

    max_err = float(np.max(F_comp))
    tolerance = 1.0e-8

    assert max_err < tolerance, (
        f"Force mismatch with MATLAB implementation with a varying A0!\n"
        f"Max difference: {max_err:.3e} (tolerance = {tolerance})"
    )

    # restore original delta
    simulator.update_params(original_phys)

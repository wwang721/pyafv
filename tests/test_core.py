import numpy as np
from pyafv import target_delta


def test_MATLAB(data_dir, simulator):
    # Compare forces computed by our code with those from MATLAB implementation

    # temporarily change delta to 0.0 for this test
    original_phys = simulator.phys
    new_phys = original_phys.replace(delta=0.0)
    simulator.update_params(new_phys)

    pts = np.loadtxt(data_dir / "init_pts.csv", delimiter=',')
    simulator.update_positions(pts)

    # Build the diagram
    diag = simulator.build()

    # Get forces
    forces = diag["forces"]

    # This is result of the MATLAB implementation for the points in 'init_pts.csv' file
    forces_matlab = np.loadtxt(data_dir / "init_forces.csv", delimiter=',')
    F_comp = np.abs(forces - forces_matlab)

    max_err = float(np.max(F_comp))
    tolerance = 1.0e-8

    assert max_err < tolerance, (
        f"Force mismatch with MATLAB implementation!\n"
        f"Max difference: {max_err:.3e} (tolerance = {tolerance})"
    )

    # restore original delta
    simulator.update_params(original_phys)


def test_physical_params(phys):
    l, d = phys.get_steady_state()
    params = phys.with_optimal_radius()
    params2 = phys.with_optimal_radius(digits=1, delta=0.5)

    # For the default physical params in the fixture
    l_real = 0.87
    d_real = 0.87 - 0.12
    delta_t_real = 0.16868

    target_force = 4.0
    delta_t = target_delta(params, target_force)

    assert np.abs(l - l_real) < 1.0e-3 and np.abs(d - d_real) < 1.0e-3, "Optimal radius not correct."
    assert np.abs(params.r - l) < 1.0e-6, "Optimal radius not set correctly."
    assert l >= 0 and d >= 0, "Optimal (l,d) should be non-negative."
    assert np.abs(delta_t - delta_t_real) < 1.0e-4, f"Computed delta not correct."
    assert delta_t >= 0, "Computed delta should be non-negative."
    assert params2.r == round(l, 1), "Optimal radius with digits not set correctly."
    assert params2.delta == 0.5, "Delta not set correctly in with_optimal_radius."

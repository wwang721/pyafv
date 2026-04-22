import numpy as np
import matplotlib
matplotlib.use("Agg")

import matplotlib.figure

import pyafv as afv


def test_small_clusters(simulator):
    # Test N<=3 clusters for Voronoi diagram construction
    
    pts = np.array([[0.0, 0.0]])  # N=1
    simulator.update_positions(pts)
    diag = simulator.build()
    simulator.plot_2d(show=False)
    afv.visualize_2d(pts, diag, simulator.phys.r)

    pts = np.array([[0.5, 0.5], [0.54, 0.52]]) * 25.  # N=2
    simulator.update_positions(pts)
    diag = simulator.build()
    simulator.plot_2d(show=False)
    afv.visualize_2d(pts, diag, simulator.phys.r, cell_colors=None, show_points=True)

    pts = np.array([[0.0, 0.0], [1.0, 10.0]])  # N=2 separated
    simulator.update_positions(pts)
    diag = simulator.build()
    simulator.plot_2d(show=False)
    afv.visualize_2d(pts, diag, simulator.phys.r, auto_adjust_bounds=False)

    pts = np.array([[0.0, 0.0], [1.0, 0.5], [2.0, 0.1]])  # N=3
    # pts = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])  # N=3 collinear
    simulator.update_positions(pts)
    simulator.build()


def test_geom(data_dir, simulator):
    # Test some strange geometries for Voronoi diagram construction

    pts = np.loadtxt(data_dir / 'init_seed407_pts.csv', delimiter=',')
    simulator.update_positions(pts)
    simulator.build()

    #-----------------------------
    pts = np.loadtxt(data_dir / 'init_seed46_pts.csv', delimiter=',')
    simulator.update_positions(pts)
    simulator.build()

    # -----------------------------
    pts = np.load(data_dir / 'debug_pts.npy')
    simulator.update_positions(pts)
    diag = simulator.build()

    #-----------------------------
    simulator.plot_2d()
    N = pts.shape[0]
    fig = afv.visualize_2d(pts, diag, simulator.phys.r, cell_colors=['C0'] * N)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) > 0

    # Test a right angle geometry for Voronoi diagram construction
    pts = np.array([[-1., 2.], [-1., -1.], [1., -1.]]) * 0.5
    simulator.update_positions(pts)
    simulator.build()

    # Test a square geometry for Voronoi diagram construction
    pts = np.array([[1., 1.], [-1., 1.], [-1., -1.], [1., -1.]]) * 0.5
    simulator.update_positions(pts)
    simulator.build()

    # Test a square geometry where SciPy Voronoi generate two nearly overlapped vertices at the center, which can cause issues in the geometry construction.
    # This is a regression test for a bug that was fixed by adding a "joggle" option to perturb points slightly to avoid precision issues.
    pts = np.load(data_dir / 'pts_at_error.npy')
    simulator.update_positions(pts)
    simulator.plot_2d()
    simulator.build()

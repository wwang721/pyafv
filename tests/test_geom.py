import numpy as np


def test_small_clusters(simulator):
    # Test N<=3 clusters for Voronoi diagram construction
    
    pts = np.array([[0.0, 0.0]])  # N=1
    simulator.update_positions(pts)
    simulator.build()

    pts = np.array([[0.5, 0.5], [0.54, 0.52]]) * 25.  # N=2
    simulator.update_positions(pts)
    simulator.build()

    pts = np.array([[0.0, 0.0], [1.0, 10.0]])  # N=2 separated
    simulator.update_positions(pts)
    simulator.build()

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
    simulator.build()

    #-----------------------------
    simulator.plot_2d()

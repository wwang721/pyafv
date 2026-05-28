import numpy as np
import pyafv as afv
from tqdm import tqdm
import matplotlib.pyplot as plt


def main():
    radius = 1.0
    points = np.random.default_rng(42).random((10_000, 2)) * 100.0
    phys = afv.PhysicalParams(r=radius)

    sim = afv.ParallelFiniteVoronoiSimulator(points, phys, (3, 3), n_workers=9)

    dt = 0.01
    with sim:
        for _ in tqdm(range(1000)):
            diag = sim.build()
            points += diag["forces"] * dt
            sim.update_positions(points)

        diag = sim.build(plot_mode=True)

    fig, ax = plt.subplots()
    afv.visualize_2d_parallel(points, diag, r=radius, ax=ax)
    plt.show()


if __name__ == "__main__":
    main()

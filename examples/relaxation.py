import numpy as np
import matplotlib.pyplot as plt
import tqdm

from afv.finite_voronoi import PhysicalParams, FiniteVoronoiSimulator


# Maximal radius
radius = 1.0

# Parameter set
phys = PhysicalParams(
    r=radius,
    A0=np.pi*(radius**2),
    P0=4.8*radius,
    KA=1.0,
    KP=1.0,
    lambda_tension=0.2
)

N = 100  # or 1, 2, 3, 4
np.random.seed(42)
pts = np.random.rand(N, 2)*0.3 + 0.35  # shape (N,2)
pts *= 25.

sim = FiniteVoronoiSimulator(pts, phys)

fig, ax = plt.subplots()
sim.plot_2d(ax=ax)
plt.show()


mu = 1.0
dt = 0.01

for _ in tqdm.tqdm(range(1000), desc="Relaxation"):
    diag = sim.build()
    forces = diag["forces"]
    pts += mu * forces * dt
    sim.update_positions(pts)

fig, ax = plt.subplots()
sim.plot_2d(ax=ax)
plt.show()

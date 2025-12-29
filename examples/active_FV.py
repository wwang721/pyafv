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

theta = 2. * np.pi * np.random.rand(N) - np.pi


sim = FiniteVoronoiSimulator(pts, phys)


mu = 1.0
va = 2.4
Dr = 0.3
dt = 0.01

for _ in tqdm.tqdm(range(200), desc="Relaxation"):
    diag = sim.build()
    pts += mu * diag["forces"] * dt
    sim.update_positions(pts)



for _ in tqdm.tqdm(range(1000), desc="Active dynamics"):
    diag = sim.build()
    forces = diag["forces"]
    
    active_velocity = va * np.column_stack((np.cos(theta), np.sin(theta)))

    pts += (mu * forces + active_velocity) * dt

    # Gaussian white noise
    theta += np.sqrt(2 * Dr * dt) * np.random.randn(N)

    sim.update_positions(pts)

print(pts)
fig, ax = plt.subplots()
sim.plot_2d(ax=ax)
ax.quiver(pts[:, 0], pts[:, 1], np.cos(theta), np.sin(theta), color='C4', scale=20, zorder=3)
plt.show()


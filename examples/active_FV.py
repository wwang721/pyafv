import numpy as np
import matplotlib.pyplot as plt
import tqdm
import pyafv as afv


np.random.seed(42)

N = 100         # number of cells
radius = 1.0    # maximal radius
mu = 1.0        # mobility
v0 = 2.4        # self-propulsion speed
Dr = 0.3        # rotational diffusion constant
dt = 0.01       # time step

# Parameter set
phys = afv.PhysicalParams(
    r=radius,
    A0=np.pi*(radius**2),
    P0=4.8*radius,
    KA=1.0,
    KP=1.0,
    lambda_tension=0.2
)

# Do not set delta unless you know what you are doing.
# We set it to zero here for comparison with the our primitive results.
phys = phys.replace(delta=0.0)

# Initial positions and orientations
pts = np.random.rand(N, 2)*0.3 + 0.35  # shape (N,2)
pts *= 25.
theta = 2. * np.pi * np.random.rand(N) - np.pi

# Initialize simulator
sim = afv.FiniteVoronoiSimulator(pts, phys)

# Relaxation to mechanical equilibrium
for _ in tqdm.tqdm(range(200), desc="Relaxation"):
    diag = sim.build()
    pts += mu * diag["forces"] * dt
    sim.update_positions(pts)

# Active dynamics
for _ in tqdm.tqdm(range(1000), desc="Active dynamics"):
    diag = sim.build()
    forces = diag["forces"]

    active_velocity = v0 * np.column_stack((np.cos(theta), np.sin(theta)))

    pts += (mu * forces + active_velocity) * dt

    # Gaussian white noise
    theta += np.sqrt(2 * Dr * dt) * np.random.randn(N)

    sim.update_positions(pts)


fig, ax = plt.subplots()
sim.plot_2d(ax=ax)
# Plot cell orientations
ax.quiver(pts[:, 0], pts[:, 1], np.cos(theta),
          np.sin(theta), color='C4', scale=20, zorder=3)
plt.show()

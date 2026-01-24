import numpy as np
import matplotlib.pyplot as plt
import tqdm
import pyafv as afv


np.random.seed(42)

N = 100         # number of cells
radius = 1.0    # maximal radius
mu = 1.0        # mobility
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

# Initial positions
pts = np.random.rand(N, 2)*0.3 + 0.35  # shape (N,2)
pts *= 25.

# Initialize simulator
sim = afv.FiniteVoronoiSimulator(pts, phys)

# Plot initial configuration
fig, ax = plt.subplots()
sim.plot_2d(ax=ax)
plt.show()

# Relaxation to mechanical equilibrium
for _ in tqdm.tqdm(range(1000), desc="Relaxation"):
    diag = sim.build()
    forces = diag["forces"]
    pts += mu * forces * dt
    sim.update_positions(pts)

# Plot relaxed configuration
fig, ax = plt.subplots()
sim.plot_2d(ax=ax)
plt.show()

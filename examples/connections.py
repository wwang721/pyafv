import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import pyafv as afv


np.random.seed(42)

N = 200         # number of cells
radius = 1.0    # maximal radius

# Parameter set
phys = afv.PhysicalParams(
    r=radius,
    A0=np.pi*(radius**2),
    P0=4.8*radius,
    KA=1.0,
    KP=1.0,
    lambda_tension=0.2
)

# Initial positions
pts = np.random.rand(N, 2)*0.3 + 0.35  # shape (N,2)
pts *= 70.

# Initialize simulator
sim = afv.FiniteVoronoiSimulator(pts, phys)
diag = sim.build()
connect = diag["connections"]

# Plot initial configuration
fig, ax = plt.subplots()
sim.plot_2d(ax=ax)

# Plot the connections between cells
num_connections = connect.shape[0]
if num_connections > 0:
    i_masked = connect[:, 0]
    j_masked = connect[:, 1]
    # Build list of segments (line endpoints) for visualization, shape: (num_pairs, 2, 2)
    segments = np.stack([pts[i_masked], pts[j_masked]], axis=1)
    # Create LineCollection
    lc = LineCollection(segments, colors="C7", linewidths=1.5, zorder=0)
    ax.add_collection(lc)

plt.show()

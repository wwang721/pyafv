import matplotlib.pyplot as plt
import pyafv as afv
import pyafv.calibrate as cal


phys = afv.PhysicalParams()

#===============  Auto calibrate ==================
f_detach, phys_cal = cal.auto_calibrate(phys, show=True)

print(f"Detachment force from DP model: {f_detach:.1f}")
print(f"{phys_cal=}")


#===========  Visualize DP simulation =============
"""
    The auto_calibrate process above is equivalent to
    applying a series of external forces until detachment
    and computing the delta value of finite-Voronoi model
    that matches the detachment force of DP model.
"""

sim = cal.DeformablePolygonSimulator(phys)

# Initial shape
print(f"{sim.detached=}")
fig, ax = plt.subplots()
sim.plot_2d(ax)
plt.show()
plt.close(fig)

#------------  Apply F = 2 -------------
sim.simulate(ext_force=2.0, dt=1e-3, nsteps=50_000)
print(f"{sim.detached=}")

fig, ax = plt.subplots()
sim.plot_2d(ax)
plt.show()
plt.close(fig)

#------------  Apply F = 4 -------------
sim.simulate(ext_force=4.0, dt=1e-3, nsteps=50_000)
print(f"{sim.detached=}")

fig, ax = plt.subplots()
sim.plot_2d(ax)
plt.show()
plt.close(fig)

import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass, replace


def sigmoid(x):
    # stable sigmoid that handles large |x|
    if x >= 0:
        z = np.exp(-x)
        return 1 / (1 + z)
    else:
        z = np.exp(x)
        return z / (1 + z)


@dataclass(frozen=True)
class PhysicalParams:
    # Radius (maximal) of the Voronoi cells
    r: float = 1.0
    A0: float = np.pi                    # Preferred area of the Voronoi cells
    P0: float = 4.8                      # Preferred perimeter of the Voronoi cells
    KA: float = 1.0                      # Area elasticity
    KP: float = 1.0                      # Perimeter elasticity
    lambda_tension: float = 0.2          # Tension difference
    delta: float = 0.                    # Small offset to avoid singularities

    def get_steady_state(self):
        # compute steady-state (l,d) given physical params
        params = [self.KA, self.KP, self.A0, self.P0, self.lambda_tension]
        result = self._minimize_energy(params, restarts=10)
        l, d = result[0]
        return l, d

    def with_optimal_radius(self):
        """Returns a new instance with the radius updated to steady state."""
        l, d = self.get_steady_state()
        new_params = replace(self, r=l)
        return new_params

    def with_delta(self, delta_new: float):
        """Returns a new instance with the specified delta."""
        return replace(self, delta=delta_new)

    def _energy_unconstrained(self, z, params):
        v, u = float(z[0]), float(z[1])
        l = np.exp(v)                              # l > 0
        phi = 0.5*np.pi * sigmoid(u)               # phi in (0, pi/2)
        s, c = np.sin(phi), np.cos(phi)
        theta = np.pi + 2.0*phi
        A = 0.5*l*l*(theta + np.sin(2.0*phi))
        P = 2.0*l*c + l*theta
        ln = l*theta

        KA, KP, A0, P0, Lambda = params
        return KA * (A - A0)**2 + KP * (P - P0)**2 + Lambda * ln

    def _minimize_energy(self, params, restarts=10, seed=None):
        rng = np.random.default_rng(seed)
        best = None
        for _ in range(restarts):
            z0 = rng.normal(size=2)

            res = minimize(lambda z: self._energy_unconstrained(z, params), z0, method="BFGS",
                           options={"gtol": 1e-8, "maxiter": 1e4})
            val = res.fun
            z = res.x

            if (best is None) or (val < best[0]):
                best = (val, z)

        # map back to (l,d)
        val, z = best
        v, u = float(z[0]), float(z[1])
        l = np.exp(v)
        phi = 0.5*np.pi * sigmoid(u)
        d = l*np.sin(phi)

        return [l, d], val


def target_delta(params: PhysicalParams, target_force: float) -> float:
    """
    Given physical parameters and a target detachment force, compute the corresponding delta.
    """
    KP, A0, P0, Lambda = params.KP, params.A0, params.P0, params.lambda_tension
    l = params.r

    distances = np.linspace(1e-6, 2*l-(1e-6), 10_000)
    detachment_forces = []
    for distance in distances:
        epsilon = l - (distance/2.)

        theta = 2 * np.pi - 2 * np.arctan2(np.sqrt(l**2 - (l - epsilon)**2), l - epsilon)
        A = (l - epsilon) * np.sqrt(l**2 -
                                    (l - epsilon)**2) + 0.5 * (l**2 * theta)
        P = 2 * np.sqrt(l**2 - (l - epsilon)**2) + l * theta

        f = 4. * np.sqrt((2-epsilon) * epsilon) * (A - A0 + KP * ((P - P0)/(2 - epsilon)) 
                                                   + (Lambda/2) * (1./((2-epsilon)*epsilon)))
        detachment_forces.append(f)

    # print(detachment_forces)

    detachment_forces = np.array(detachment_forces)

    idx = np.abs(detachment_forces[None, :] - target_force).argmin()
    target_distances = distances[idx]

    delta = np.sqrt(4*(l**2) - target_distances**2)

    return delta


__all__ = [
    "PhysicalParams",
    "target_delta",
]

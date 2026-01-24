from __future__ import annotations
import numpy as np
from dataclasses import dataclass, replace


def _require_float_scalar(x: object, name: str) -> float:       # pragma: no cover
    """
    Accept Python real scalars (int/float) and NumPy real scalars.
    Reject other types (including bool).
    Return a normalized Python float.
    """
    # Reject bool explicitly (since bool is a subclass of int)
    if isinstance(x, bool):
        raise TypeError(f"{name} must be a real scalar (float-like), got bool")
    elif isinstance(x, (int, float, np.integer, np.floating)):
        xf = float(x)
        if not np.isfinite(xf):
            raise ValueError(f"{name} must be finite, got {x}")
        return xf

    # Reject everything else
    raise TypeError(f"{name} must be a real scalar (float-like), got {type(x).__name__}")


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
    r"""Physical parameters for the active-finite-Voronoi (AFV) model.

    .. warning::
        * **Frozen dataclass** is used for :py:class:`PhysicalParams` to ensure immutability of instances.
        * Do not set :py:attr:`delta` unless you know what you are doing.

    Args:
        r: Radius (maximal) of the Voronoi cells, sometimes denoted as :math:`\ell`.
        A0: Preferred area of the Voronoi cells.
        P0: Preferred perimeter of the Voronoi cells.
        KA: Area elasticity constant.
        KP: Perimeter elasticity constant.
        lambda_tension: Tension difference between non-contacting edges and contacting edges.
        delta: Contact truncation threshold to avoid singularities in computations; if None, set to 0.45*r.
    """
    
    r: float = 1.0               #: Radius (maximal) of the Voronoi cells, sometimes denoted as :math:`\ell`.
    A0: float = np.pi            #: Preferred area of the Voronoi cells.
    P0: float = 4.8              #: Preferred perimeter of the Voronoi cells.
    KA: float = 1.0              #: Area elasticity constant.
    KP: float = 1.0              #: Perimeter elasticity constant.
    lambda_tension: float = 0.2  #: Tension difference between non-contacting edges and contacting edges.
    delta: float | None = None   #: Contact truncation threshold to avoid singularities in computations.

    def __post_init__(self):
        # Normalize and validate required scalar floats
        object.__setattr__(self, "r", _require_float_scalar(self.r, "r"))
        object.__setattr__(self, "A0", _require_float_scalar(self.A0, "A0"))
        object.__setattr__(self, "P0", _require_float_scalar(self.P0, "P0"))
        object.__setattr__(self, "KA", _require_float_scalar(self.KA, "KA"))
        object.__setattr__(self, "KP", _require_float_scalar(self.KP, "KP"))
        object.__setattr__(self, "lambda_tension", _require_float_scalar(self.lambda_tension, "lambda_tension"))

        if self.delta is None:
            object.__setattr__(self, "delta", 0.45 * self.r)
        else:
            try:
                object.__setattr__(self, "delta", _require_float_scalar(self.delta, "delta"))
            except TypeError:       # pragma: no cover
                raise TypeError(f"delta must be a real scalar (float-like) or None, got {type(self.delta).__name__}") from None

    def get_steady_state(self) -> tuple[float, float]:
        r"""Search for the steady-state :math:`(\ell,d)` of a cell doublet for the given physical parameters (by minimizing total energy).

        Returns:
            Steady-state (optimal) :math:`(\ell_0,d_0)` values.

        .. note::
            :math:`\ell` is the maximal cell radius, and :math:`2d` is the cell-center distance of a doublet (rather than :math:`d`).        
        """
        params = [self.KA, self.KP, self.A0, self.P0, self.lambda_tension]
        result = self._minimize_energy(params, restarts=10)
        l, d = result[0]
        return l, d

    def with_optimal_radius(self, digits: int | None = None, delta: float | None = None) -> PhysicalParams:
        r"""Returns a new instance of :py:class:`PhysicalParams` with the maximum radius :math:`\ell` (or :py:attr:`r`) updated to the steady state value :math:`\ell_0` of cell doublets.
        Other parameters (except :py:attr:`delta`) remain unchanged.
        
        Basically a wrapper around :py:meth:`get_steady_state` + creating a new instance.

        Args:
            digits: If not None, round the optimal radius :math:`\ell_0` to the specified number of decimal places.
                Only intended for randomized tests to avoid floating-point precision issues.
            delta: If not None, set the contact truncation threshold :py:attr:`delta` to this value in the returned instance.

        Returns:
            New instance with optimal radius.
        
        .. important::
            In the returned instance, the contact truncation threshold :py:attr:`delta` is set to 0.45*r by default.
        """
        l, d = self.get_steady_state()

        if digits is not None:
            l = round(l, digits)

        if delta is not None:
            delta_new = delta
        else:
            delta_new = 0.45 * l

        new_params = replace(self, r=l, delta=delta_new)
        return new_params

    def replace(self, **changes: float | None) -> PhysicalParams:
        """Returns a new instance of :py:class:`PhysicalParams` with specified fields replaced by new values.

        Args:
            **changes: Field names and their new values to be replaced.

        Returns:
            New instance with the updated fields.

        .. hint::
            This is a convenience method wrapping :py:func:`dataclasses.replace`,
            e.g., to change :py:attr:`A0` and :py:attr:`delta` of an existing instance *phys*: ``phys_new = phys.replace(A0=5.0, delta=0.3)``.
        """
        return replace(self, **changes)

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

        from scipy.optimize import minimize
        
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
    r"""
    Given the physical parameters and a target detachment force, compute the corresponding delta.

    Args:
        params: Physical parameters of the AFV model.
        target_force: Target detachment force.

    Raises:
        TypeError: If *params* is not an instance of :py:class:`PhysicalParams`.
        ValueError: If the target force is not within the achievable range.

    Returns:
        Corresponding value of the truncation threshold :math:`\delta`.

    .. note::
        We search for the cell-cell separation at which the intercellular force
        equals the target force, scanning distances from :math:`10^{-6}\ell` to
        :math:`(2-10^{-6})\ell` in steps of :math:`10^{-6}\ell`, and select the
        **largest distance** at which the match occurs.
    """

    if not isinstance(params, PhysicalParams):      # pragma: no cover
        raise TypeError("params must be an instance of PhysicalParams")

    KA, KP, A0, P0, Lambda = params.KA, params.KP, params.A0, params.P0, params.lambda_tension
    l = params.r

    distances = np.linspace(1e-6, 2.-(1e-6), 10**6) * l
    
    epsilon = l - (distances/2.)

    theta = 2 * np.pi - 2 * np.arctan2(np.sqrt(l**2 - (l - epsilon)**2), l - epsilon)
    A = (l - epsilon) * np.sqrt(l**2 -
                                (l - epsilon)**2) + 0.5 * (l**2 * theta)
    P = 2 * np.sqrt(l**2 - (l - epsilon)**2) + l * theta

    detachment_forces = 4. * np.sqrt((2 * l - epsilon) * epsilon) * (KA * (A - A0) + KP * ((P - P0)/(2 * l - epsilon)) 
                                                + (Lambda/2) * l /((2 * l - epsilon) * epsilon))
    
    # idx = np.abs(detachment_forces[None, :] - target_force).argmin()
    # target_distances = distances[idx]

    # ---------------------- Better way to search foot ----------------------------
    f = detachment_forces - target_force    # find root of f=0
    cross = (f[:-1] == 0) | (np.signbit(f[:-1]) != np.signbit(f[1:]))   # crossing points

    if np.any(cross):
        i = np.flatnonzero(cross)[-1]  # last crossing interval [i, i+1]
        # optional: linear interpolation for a better distance estimate
        x0, x1 = distances[i], distances[i+1]
        f0, f1 = f[i], f[i+1]
        target_distance = x0 if f1 == f0 else x0 + (0 - f0) * (x1 - x0) / (f1 - f0)
    else:
        raise ValueError("No valid delta found for the given target force.")
    # ------------------------------------------------------------------------------
    
    delta = np.sqrt(4*(l**2) - target_distance**2)

    return delta


__all__ = [
    "PhysicalParams",
    "target_delta",
]

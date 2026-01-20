# Enable postponed evaluation of annotations
from __future__ import annotations

# Only import typing modules when type checking, e.g., in VS Code or IDEs.
from typing import TYPE_CHECKING
if TYPE_CHECKING:                             # pragma: no cover
    import matplotlib.axes

import numpy as np
from dataclasses import replace

from ..physical_params import PhysicalParams


# ---------- geometry helpers (vectorized) ----------
def polygon_centroid(pts: np.ndarray) -> np.ndarray:
    """
    Centroid (center of mass) of a simple polygon with uniform density.

    Args:
        pts: (N,2) array of vertices in order (first need not repeat at end).
    
    Returns:
        (2,) centroid array. (If degenerate area, returns vertex mean.)
    """
    pts = np.asarray(pts, dtype=float)
    x, y = pts[:, 0], pts[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)

    cross = x * y1 - x1 * y        # per-edge cross terms
    A2 = cross.sum()               # equals 2 * signed area

    if np.isclose(A2, 0.0):        # pragma: no cover
        # degenerate: fall back to average of vertices
        return pts.mean(axis=0)

    cx = ((x + x1) * cross).sum() / (3.0 * A2)
    cy = ((y + y1) * cross).sum() / (3.0 * A2)
    return np.array([cx, cy])


def polygon_area_perimeter(pts: np.ndarray) -> tuple[float, float]:
    """
    Compute the area and perimeter for a Counter-ClockWise (CCW) polygon.

    Args:
        pts: (N,2) array of vertices in CCW order (first need not repeat at end).
    
    Returns:
        A *tuple* of (area, perimeter).
    """
    x = pts[:, 0]
    y = pts[:, 1]
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)
    # Shoelace
    A = 0.5 * np.abs(np.dot(x, y_next) - np.dot(y, x_next))
    # Perimeter
    edges = np.roll(pts, -1, axis=0) - pts
    P = np.sum(np.linalg.norm(edges, axis=1))
    return A, P


def resample_polyline(pts: np.ndarray, M: int | None = None) -> np.ndarray:
    """
    Resample an open polyline to M points with uniform arclength spacing.
    Keeps endpoints exactly. If M is None, uses original number of points.

    Args:
        pts: (N,2) array of polyline vertices.
        M: Number of output points; if None, uses N.
    
    Returns:
        (M,2) array of resampled polyline vertices.
    """
    pts = np.asarray(pts, dtype=float)
    N = pts.shape[0]
    if M is None:
        M = N

    # cumulative arclength
    seg = np.diff(pts, axis=0)
    s = np.concatenate(([0.0], np.cumsum(np.linalg.norm(seg, axis=1))))
    L = s[-1]
    if L == 0.0:                       # pragma: no cover
        return np.tile(pts[0], (M, 1))

    # target arclengths (uniform)
    t = np.linspace(0.0, L, M)

    # locate segment for each t and linearly interpolate
    idx = np.searchsorted(s, t, side='right') - 1
    idx = np.clip(idx, 0, N - 2)

    s0 = s[idx]
    s1 = s[idx + 1]
    p0 = pts[idx]
    p1 = pts[idx + 1]

    denom = (s1 - s0)
    alpha = (t - s0) / denom
    alpha = np.nan_to_num(alpha, nan=0.0)

    out = p0 + (p1 - p0) * alpha[:, None]
    out[0] = pts[0]
    out[-1] = pts[-1]
    return out


# ----------- Internal use only helpers ------------------

def _grad_area(pts):
    """
    ∂A/∂v_i for polygon vertices v_i = (x_i, y_i), CCW order.
    Using: ∂A/∂x_i = 0.5*(y_{i+1} - y_{i-1}), ∂A/∂y_i = 0.5*(x_{i-1} - x_{i+1})
    """
    x = pts[:, 0]
    y = pts[:, 1]
    y_next = np.roll(y, -1)
    y_prev = np.roll(y,  1)
    x_next = np.roll(x, -1)
    x_prev = np.roll(x,  1)
    gx = 0.5 * (y_next - y_prev)
    gy = 0.5 * (x_prev - x_next)
    return np.stack((gx, gy), axis=1)


def _grad_perimeter_and_tension(pts, lam):
    """
    Return (dP/dv_i, tension term) in one pass.
    dP/dv_i = (v_i - v_{i-1})/|e_{i-1}| + (v_i - v_{i+1})/|e_i|
    Tension term = lam_i * (v_i - v_{i+1})/|e_i| + lam_{i-1} * (v_i - v_{i-1})/|e_{i-1}|
    """
    v = pts
    v_next = np.roll(v, -1, axis=0)
    v_prev = np.roll(v,  1, axis=0)

    # directed edges
    e_fwd = v_next - v          # e_i   = v_{i+1} - v_i
    e_bwd = v - v_prev          # e_{i-1}= v_i    - v_{i-1}

    # norms (avoid divide-by-zero)
    nf = np.linalg.norm(e_fwd, axis=1)
    nb = np.linalg.norm(e_bwd, axis=1)

    # unit directions for derivative pieces
    dP_md = - e_fwd / nf[:, None]   # (v_i - v_{i+1})/|e_i|
    dP_0d = e_bwd / nb[:, None]   # (v_i - v_{i-1})/|e_{i-1}|
    dP = dP_md + dP_0d

    lam_prev = np.roll(lam, 1)
    tension = lam[:, None] * dP_md + lam_prev[:, None] * dP_0d
    return dP, tension


def _energy_gradient(pts, KA, KP, A0, P0, lam):
    """Return ∂E/∂v for one polygon with area–perimeter + edge tensions."""
    A, P = polygon_area_perimeter(pts)
    gA = _grad_area(pts)
    gP, gT = _grad_perimeter_and_tension(pts, lam)

    dE = 2.0 * KA * (A - A0) * gA + 2.0 * KP * P * gP + gT
    return dE, A, P


# ------------------------------------------------


class DeformablePolygonSimulator:
    """
    Simulator for the deformable-polygon (DP) model of cell doublets.

    Args:
        phys: An instance of PhysicalParams containing the physical parameters, while *phys.r* and *phys.delta* are ignored.
        num_vertices: Number of vertices :math:`M` to use for each cell.
    
    Raises:
        TypeError: If *phys* is not an instance of *PhysicalParams*.

    Warnings:
        If the target shape index (based on *phys.P0* and *phys.A0*) indicates a non-circular shape,
        a **UserWarning** is raised since the DP model is not valid in that regime.

    """

    def __init__(self, phys: PhysicalParams, num_vertices: int = 100):
        """
        Initialize the Deformable Polygon model for a cell doublet at steady state.
        """
        if not isinstance(phys, PhysicalParams):      # pragma: no cover
            raise TypeError("phys must be an instance of PhysicalParams")
        
        self.phys = phys
        self.num_vertices = num_vertices
        self._get_target_shape_index()                # check model validity

        P0 = phys.P0
        KP = phys.KP
        Lambda = phys.lambda_tension

        lambda_c = -P0 * 2 * KP
        lambda_n = Lambda + lambda_c
        
        l0, d0 = phys.get_steady_state()
        self.phys = replace(phys, r=l0)

        theta = np.arctan2(np.sqrt(l0**2 - (d0)**2), d0)
        # ---------- initial shapes ----------
        angles1 = np.linspace(theta, 2*np.pi-theta, num_vertices)
        angles2 = np.linspace(-np.pi+theta, np.pi-theta, num_vertices)

        pts1 = np.vstack((np.cos(angles1), np.sin(angles1))).T
        pts1 *= l0
        pts1[:, 0] -= pts1[0, 0]

        pts2 = np.vstack((np.cos(angles2), np.sin(angles2))).T
        pts2 *= l0
        pts2[:, 0] -= pts2[0, 0]

        # stitch endpoints
        pts2[0] = pts1[-1]
        pts2[-1] = pts1[0]

        # per-vertex tensions (last index uses lambda_c, others lambda_n)
        lam1 = np.full(len(pts1), lambda_n)
        lam1[-1] = lambda_c
        lam2 = np.full(len(pts2), lambda_n)
        lam2[-1] = lambda_c
        
        self.pts1 : np.ndarray = pts1  #: (N,2) array of vertices in cell 1.
        self.pts2 : np.ndarray = pts2  #: (N,2) array of vertices in cell 2.
        self.lam1 = lam1
        self.lam2 = lam2

        self.contact_length : float = np.linalg.norm(pts1[0] - pts1[-1])  #: Current contact length.
        self.detach_criterion : float = 2.*np.pi*l0/num_vertices  #: Contact length at which detachment occurs; defaults to :math:`2\pi \ell_0/M`.
        self.detached : bool = True if self.contact_length <= self.detach_criterion else False  #: Indicates whether the doublet has detached.


    def _get_target_shape_index(self) -> float:
        """Compute the target shape index
        """
        P0 = self.phys.P0
        A0 = self.phys.A0
        target_shape_index = P0 / np.sqrt(A0)
        
        # 2.0 * np.sqrt(np.pi) is for circular shape
        if target_shape_index > 2.0 * np.sqrt(np.pi):      # pragma: no cover
            # raise warning
            import warnings
            warnings.warn(
                "Target shape index indicates non-circular shape; "
                "the deformable-polygon (DP) model may not be valid in this regime. "
                "Do not use the DP model for calibration if this warning appears.",
                UserWarning,
                stacklevel=3,
            )

        return target_shape_index 


    def _step_update(self, ext_force: float, dt: float) -> None:
        r"""Single simulation step under external force.

        Args:
            ext_force: The external force applied to the cell doublet.
            dt: Time step size.
        
        .. warning::
            This is an internal method. Use with caution.
            We have implicitly assumed that the vertex mobility is :math:`\mu=1` so that :math:`\Delta x= F \Delta t`.
        """

        KA = self.phys.KA
        KP = self.phys.KP
        A0 = self.phys.A0
        P0 = self.phys.P0

        # gradients for each polygon
        g1, A1, P1 = _energy_gradient(self.pts1, KA, KP, A0, P0, self.lam1)
        g2, A2, P2 = _energy_gradient(self.pts2, KA, KP, A0, P0, self.lam2)

        # ext_force * contrib1 / P1    # uniform force in -x direction, note force is the negative gradient
        g1[:, 0] += ext_force / len(self.pts1)
        # ext_force * contrib2 / P2    # uniform force in +x direction
        g2[:, 0] -= ext_force / len(self.pts2)

        # combine gradients at the two shared vertices:
        # shared verts: pts1[-1] == pts2[0] and pts1[0] == pts2[-1]
        g1_comb = g1.copy()
        g2_comb = g2.copy()

        g1_comb[-1] = g1[-1] + g2[0]
        g1_comb[0] = g1[0] + g2[-1]
        # For pts2, we will overwrite endpoints from pts1 after stepping, so their own update is not needed.

        # update interior vertices
        self.pts1 -= dt * g1_comb

        # update interior of pts2
        self.pts2[1:-1] -= dt * g2_comb[1:-1]

        # synchronize the two shared endpoints to be identical (take the updated pts1 values)
        self.pts2[0] = self.pts1[-1]
        self.pts2[-1] = self.pts1[0]


    def simulate(self, ext_force: float, dt: float, nsteps: int, resample_every: int = 1000) -> None:
        """
        Simulate the DP model for a number of time steps under an external force.

        This is basically a wrapper around :py:meth:`_step_update` and :py:func:`resample_polyline` with some bookkeeping.

        Args:
            ext_force: The external force applied to the cell doublet.
            dt: Time step size.
            nsteps: Number of time steps to simulate.
            resample_every: How often (in steps) to resample the polygon vertices for uniform spacing.
        """
        if self.detached:
            return

        for _ in range(nsteps):
        
            self._step_update(ext_force, dt)

            if (_ + 1) % resample_every == 0:
                self.pts1 = resample_polyline(self.pts1)
                self.pts2 = resample_polyline(self.pts2)

            self.contact_length = np.linalg.norm(self.pts1[0] - self.pts1[-1])

            if self.contact_length <= self.detach_criterion:
                self.detached = True
                break


    def plot_2d(self, ax: matplotlib.axes.Axes | None = None, show: bool = False) -> matplotlib.axes.Axes:
        """
        Render a 2D snapshot of the cell doublet in DP model.

        Args:
            ax: If provided, draw into this axes; otherwise get the current axes.
            show: Whether to call ``plt.show()`` at the end.
        
        Returns:
            The matplotlib axes containing the plot.
        """

        import matplotlib.pyplot as plt

        if ax is None:             # pragma: no cover
            ax = plt.gca()

        pts1 = self.pts1
        pts2 = self.pts2

        ax.plot(pts1[:, 0], pts1[:, 1], 'o-', color=(21/255, 174/255, 22/255), ms=4, zorder=1)
        ax.plot([pts1[0, 0], pts1[-1, 0]],
                [pts1[0, 1], pts1[-1, 1]], '-', color=(222/255, 49/255, 34/255), lw=3, zorder=0)  # contact
        ax.plot(pts2[:, 0], pts2[:, 1], 'o-', color=(21/255, 174/255, 22/255), ms=4, zorder=1)


        box_size = 2.5 * self.phys.r
        ax.set_xlim(-box_size, box_size)
        ax.set_ylim(-box_size/2, box_size/2)
        ax.set_aspect('equal')
        ax.axis('off')   # <- hides ticks, labels, and spines
        
        if show:                   # pragma: no cover
            plt.show()
        return ax

"""
Python multiprocessing domain-decomposed finite Voronoi simulator.
"""

from __future__ import annotations

# Only import typing modules when type checking, e.g., in VS Code or IDEs.
from typing import TYPE_CHECKING
if TYPE_CHECKING:                             # pragma: no cover
    from concurrent.futures import ProcessPoolExecutor
    import typing

from dataclasses import dataclass
import os

import numpy as np

from .finite_voronoi import FiniteVoronoiSimulator
from .physical_params import PhysicalParams


@dataclass(frozen=True)
class DomainDecomposition:
    """Point-only data for one spatial domain plus its halo.

    This object stores the index bookkeeping needed to relate a local domain
    calculation back to the original global point array. It contains no finite
    Voronoi or AFV-specific data.

    Args:
        domain_id: Integer id of this spatial domain.
        grid_ix: Domain index in the x direction.
        grid_iy: Domain index in the y direction.
        x_range: Owned-domain x interval.
        y_range: Owned-domain y interval.
        halo_x_range: Local-domain x interval after expanding by the halo width.
        halo_y_range: Local-domain y interval after expanding by the halo width.
        local_global_ids: Global point ids included in this domain's local box,
            including halo points.
        owned_local_ids: Local indices of points uniquely owned by this domain.
        local_pts: Local point coordinates, equivalent to
            ``points[local_global_ids]``.
    """

    domain_id: int  # Integer id of this spatial domain.

    # Grid coordinates of this owned domain.
    grid_ix: int
    grid_iy: int

    # Original owned-domain box. Points inside this box belong to this domain
    # as their unique owner.
    x_range: tuple[float, float]
    y_range: tuple[float, float]

    # Expanded local box used to collect halo/ghost points.
    halo_x_range: tuple[float, float]
    halo_y_range: tuple[float, float]

    # Original/global point ids included in this domain's local box,
    # including halo points.
    local_global_ids: np.ndarray

    # Positions inside local_global_ids/local_pts for points uniquely owned
    # by this domain.
    owned_local_ids: np.ndarray

    # Point coordinates for local_global_ids:
    # local_pts == points[local_global_ids].
    local_pts: np.ndarray


def _validate_grid_shape(grid_shape: tuple[int, int]) -> tuple[int, int]:
    nx, ny = grid_shape
    nx = int(nx)
    ny = int(ny)
    if nx <= 0 or ny <= 0:                   # pragma: no cover
        raise ValueError("grid_shape entries must be positive")
    return nx, ny


def _validate_domain_bounds(
    domain_bounds: tuple[tuple[float, float], tuple[float, float]] | None,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    if domain_bounds is None:
        return None
    (xmin, xmax), (ymin, ymax) = domain_bounds
    xmin = float(xmin)
    xmax = float(xmax)
    ymin = float(ymin)
    ymax = float(ymax)
    if not (xmin < xmax and ymin < ymax):                   # pragma: no cover
        raise ValueError("domain_bounds must be ((xmin, xmax), (ymin, ymax)) with nonzero spans")
    return (xmin, xmax), (ymin, ymax)


def _domain_edges(
    points: np.ndarray,
    grid_shape: tuple[int, int],
    domain_bounds: tuple[tuple[float, float], tuple[float, float]] | None,
) -> tuple[np.ndarray, np.ndarray]:
    nx, ny = grid_shape
    if domain_bounds is None:
        xmin = float(np.min(points[:, 0]))
        xmax = float(np.max(points[:, 0]))
        ymin = float(np.min(points[:, 1]))
        ymax = float(np.max(points[:, 1]))
        if xmin == xmax or ymin == ymax:                   # pragma: no cover
            raise ValueError("points must span a nonzero range in both x and y")
    else:
        (xmin, xmax), (ymin, ymax) = domain_bounds
        if (
            np.any(points[:, 0] < xmin)
            or np.any(points[:, 0] > xmax)
            or np.any(points[:, 1] < ymin)
            or np.any(points[:, 1] > ymax)
        ):                   # pragma: no cover
            raise ValueError("points must lie inside domain_bounds")

    x_edges = np.linspace(xmin, xmax, nx + 1)
    y_edges = np.linspace(ymin, ymax, ny + 1)
    return x_edges, y_edges


def decompose_points(
    points: np.ndarray,
    grid_shape: tuple[int, int] = (2, 2),
    halo_width: float = 0.0,
    *,
    domain_bounds: tuple[tuple[float, float], tuple[float, float]] | None = None,
    method: typing.Literal["dense", "sorted_x"] = "dense",
) -> list[DomainDecomposition]:
    """Decompose points into owned grid domains plus halo/local points.

    Points on internal owned-domain boundaries are assigned to exactly one
    domain by using right-sided binning and clipping at the outermost boundary.
    Halo/local domains include points on all halo box edges.

    .. tip::
       This function is independent of the finite Voronoi model. It only computes
       global/local point-index bookkeeping.

    Args:
        points (numpy.ndarray): (N,2) array of point coordinates.
        grid_shape: Number of owned domains in the x and y directions.
        halo_width: Width added to each side of every owned domain to collect
            local halo points.
        domain_bounds: Optional domain bounds as ``((xmin, xmax), (ymin, ymax))``.
            If *None*, bounds are inferred from *points*.
        method: Method used to collect halo points. ``"dense"`` builds a
            dense domain-by-point mask and is usually faster for moderate
            systems. ``"sorted_x"`` uses less temporary memory.

    Returns:
        list[DomainDecomposition]: One list of :py:class:`DomainDecomposition` objects in row-major order,
        where each element stores the local and global point information for a single domain.

    Raises:
        ValueError: If *points* does not have shape (N,2).
        ValueError: If *points* contains non-finite values.
        ValueError: If *grid_shape*, *halo_width*, *domain_bounds*, or
            *method* is invalid.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:                   # pragma: no cover
        raise ValueError("points must have shape (N,2)")
    if not np.all(np.isfinite(points)):                   # pragma: no cover
        raise ValueError("points must be finite")
    halo_width = float(halo_width)
    if halo_width < 0.0:                   # pragma: no cover
        raise ValueError("halo_width must be non-negative")
    if method not in ("dense", "sorted_x"):                   # pragma: no cover
        raise ValueError("method must be 'dense' or 'sorted_x'")

    nx, ny = _validate_grid_shape(grid_shape)
    n_domains = nx * ny
    checked_bounds = _validate_domain_bounds(domain_bounds)

    if n_domains == 1:
        if checked_bounds is None:
            xmin = float(np.min(points[:, 0]))
            xmax = float(np.max(points[:, 0]))
            ymin = float(np.min(points[:, 1]))
            ymax = float(np.max(points[:, 1]))
        else:
            (xmin, xmax), (ymin, ymax) = checked_bounds
            if (
                np.any(points[:, 0] < xmin)
                or np.any(points[:, 0] > xmax)
                or np.any(points[:, 1] < ymin)
                or np.any(points[:, 1] > ymax)
            ):                   # pragma: no cover
                raise ValueError("points must lie inside domain_bounds")
        local_global_ids = np.arange(points.shape[0], dtype=int)
        owned_local_ids = local_global_ids.copy()
        return [
            DomainDecomposition(
                domain_id=0,
                grid_ix=0,
                grid_iy=0,
                x_range=(xmin, xmax),
                y_range=(ymin, ymax),
                halo_x_range=(
                    xmin - halo_width,
                    xmax + halo_width,
                ),
                halo_y_range=(
                    ymin - halo_width,
                    ymax + halo_width,
                ),
                local_global_ids=local_global_ids,
                owned_local_ids=owned_local_ids,
                local_pts=points[local_global_ids],
            )
        ]

    x_edges, y_edges = _domain_edges(points, (nx, ny), checked_bounds)

    ix = np.searchsorted(x_edges, points[:, 0], side="right") - 1
    iy = np.searchsorted(y_edges, points[:, 1], side="right") - 1
    ix = np.clip(ix, 0, nx - 1)
    iy = np.clip(iy, 0, ny - 1)

    domain_ids = iy * nx + ix
    owned_order = np.argsort(domain_ids, kind="stable")
    owned_domain_ids = domain_ids[owned_order]
    owned_counts = np.bincount(owned_domain_ids, minlength=n_domains)
    owned_starts = np.concatenate(([0], np.cumsum(owned_counts)))

    domain_grid_ix = np.arange(n_domains) % nx
    domain_grid_iy = np.arange(n_domains) // nx
    x0 = x_edges[domain_grid_ix]
    x1 = x_edges[domain_grid_ix + 1]
    y0 = y_edges[domain_grid_iy]
    y1 = y_edges[domain_grid_iy + 1]

    halo_x0 = x0 - halo_width
    halo_x1 = x1 + halo_width
    halo_y0 = y0 - halo_width
    halo_y1 = y1 + halo_width

    if method == "dense":
        px = points[:, 0]
        py = points[:, 1]
        local_mask = (
            (px[None, :] >= halo_x0[:, None])
            & (px[None, :] <= halo_x1[:, None])
            & (py[None, :] >= halo_y0[:, None])
            & (py[None, :] <= halo_y1[:, None])
        )
        local_domain_ids, local_global_ids_all = np.nonzero(local_mask)
        local_counts = np.bincount(local_domain_ids, minlength=n_domains)
        local_starts = np.concatenate(([0], np.cumsum(local_counts)))
    else:
        py = points[:, 1]
        x_order = np.argsort(points[:, 0], kind="stable")
        x_sorted = points[x_order, 0]
        halo_x_left = np.searchsorted(x_sorted, halo_x0, side="left")
        halo_x_right = np.searchsorted(x_sorted, halo_x1, side="right")

    domains = []
    for domain_id in range(n_domains):
        grid_iy, grid_ix = divmod(domain_id, nx)
        owned_global_ids = owned_order[
            owned_starts[domain_id] : owned_starts[domain_id + 1]
        ]
        if method == "dense":
            local_global_ids = local_global_ids_all[
                local_starts[domain_id] : local_starts[domain_id + 1]
            ]
        else:
            x_candidates = x_order[
                halo_x_left[domain_id] : halo_x_right[domain_id]
            ]
            y_mask = (
                (py[x_candidates] >= halo_y0[domain_id])
                & (py[x_candidates] <= halo_y1[domain_id])
            )
            local_global_ids = np.sort(x_candidates[y_mask])
        owned_local_ids = np.searchsorted(local_global_ids, owned_global_ids)
        valid_owned = owned_local_ids < local_global_ids.size
        if (
            not np.all(valid_owned)
            or not np.array_equal(local_global_ids[owned_local_ids], owned_global_ids)
        ):                   # pragma: no cover
            raise RuntimeError(f"Owned/local index mapping failed for domain {domain_id}")

        domains.append(
            DomainDecomposition(
                domain_id=domain_id,
                grid_ix=grid_ix,
                grid_iy=grid_iy,
                x_range=(float(x_edges[grid_ix]), float(x_edges[grid_ix + 1])),
                y_range=(float(y_edges[grid_iy]), float(y_edges[grid_iy + 1])),
                halo_x_range=(float(halo_x0[domain_id]), float(halo_x1[domain_id])),
                halo_y_range=(float(halo_y0[domain_id]), float(halo_y1[domain_id])),
                local_global_ids=local_global_ids,
                owned_local_ids=owned_local_ids,
                local_pts=points[local_global_ids],
            )
        )

    return domains


@dataclass(frozen=True)
class _DomainTask:
    domain: DomainDecomposition
    local_preferred_areas: np.ndarray
    phys: PhysicalParams
    backend: typing.Literal["cython", "python"] | None
    connect: bool
    plot_mode: bool


def _build_domain(task: _DomainTask) -> dict[str, object]:
    domain = task.domain
    pid = os.getpid()
    owned_count = int(domain.owned_local_ids.size)
    owned_global_ids = domain.local_global_ids[domain.owned_local_ids]

    empty_float = np.empty(0, dtype=float)
    empty_int = np.empty(0, dtype=int)
    empty_forces = np.empty((0, 2), dtype=float)
    empty_connections = np.empty((0, 2), dtype=int)

    if owned_count == 0:                   # pragma: no cover
        result = {
            "domain_id": domain.domain_id,
            "pid": pid,
            "owned_global_ids": empty_int,
            "forces": empty_forces,
            "areas": empty_float,
            "perimeters": empty_float,
            "arclens": empty_float,
            "coord_nums": empty_int,
            "connections": empty_connections,
        }
        if task.plot_mode:
            result["diag_plot"] = {
                "vertices": np.empty((0, 2), dtype=float),
                "regions": [],
                "edges_type": [],
            }
        return result

    sim = FiniteVoronoiSimulator(domain.local_pts, task.phys, backend=task.backend)
    sim.update_preferred_areas(task.local_preferred_areas)
    diag = sim.build(connect=task.connect)

    connections = empty_connections
    if task.connect and diag["connections"].size:
        local_connections = np.asarray(diag["connections"], dtype=int)
        owned_local_mask = np.zeros(domain.local_global_ids.size, dtype=bool)
        owned_local_mask[domain.owned_local_ids] = True
        touches_owned = (
            owned_local_mask[local_connections[:, 0]]
            | owned_local_mask[local_connections[:, 1]]
        )
        connections = np.sort(
            domain.local_global_ids[local_connections[touches_owned]],
            axis=1,
        )

    result = {
        "domain_id": domain.domain_id,
        "pid": pid,
        "owned_global_ids": owned_global_ids,
        "forces": diag["forces"][domain.owned_local_ids],
        "areas": diag["areas"][domain.owned_local_ids],
        "perimeters": diag["perimeters"][domain.owned_local_ids],
        "arclens": diag["arclens"][domain.owned_local_ids],
        "coord_nums": diag["coord_nums"][domain.owned_local_ids],
        "connections": connections,
    }
    if task.plot_mode:
        diag_plot = {
            "vertices": diag["vertices"],
            "regions": [diag["regions"][ix] for ix in domain.owned_local_ids],
            "edges_type": [diag["edges_type"][ix] for ix in domain.owned_local_ids],
        }
        result["diag_plot"] = diag_plot
    return result


class ParallelFiniteVoronoiSimulator:
    """Python multiprocessing domain-decomposed simulator for the AFV model.

    This class decomposes the full point set into rectangular domains with
    halo regions, then composes :py:class:`FiniteVoronoiSimulator` on each
    local subdomain. The returned diagnostics are merged back into global
    cell indexing for owned cells. When ``n_workers > 1``, local subdomain
    builds use Python worker processes.

    Args:
        pts (numpy.ndarray): (N,2) array of initial cell center positions.
        phys: Physical parameters used within this simulator.
        grid_shape: Number of domains in the x and y directions.
        n_workers: Number of worker processes. If *None*, use ``os.cpu_count()``.
        halo_width: Width of the halo region added to each domain. If *None*,
            use ``4.01 * phys.r``.
        backend: Optional, specify "python" to force the use of the pure Python
            fallback implementation inside each local finite Voronoi simulator.
            Otherwise, the "cython" backend is used.
        domain_bounds: Optional domain bounds as ``((xmin, xmax), (ymin, ymax))``.
            If *None*, bounds are inferred from the current point positions.
        decomposition_method: Method used to collect halo points. ``"dense"``
            is usually faster for moderate systems, while ``"sorted_x"`` uses
            less temporary memory.

    Raises:
        ValueError: If *pts* does not have shape (N,2).
        ValueError: If *grid_shape*, *halo_width*, *domain_bounds*, or
            *decomposition_method* is invalid.
        TypeError: If *phys* is not an instance of :py:class:`PhysicalParams`.

    .. note::
        For repeated calls with ``n_workers > 1``, put the loop inside the
        context manager so worker processes are created once and reused across
        build steps::

            with sim:       # sim is an instance of ParallelFiniteVoronoiSimulator
                for step in range(num_steps):
                    diag = sim.build()
    """

    def __init__(
        self,
        pts: np.ndarray,
        phys: PhysicalParams,
        grid_shape: tuple[int, int] = (2, 2),
        n_workers: int | None = None,
        *,
        halo_width: float | None = None,
        backend: typing.Literal["cython", "python"] | None = None,
        domain_bounds: tuple[tuple[float, float], tuple[float, float]] | None = None,
        decomposition_method: typing.Literal["dense", "sorted_x"] = "dense",
    ):
        pts = np.asarray(pts, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2:                   # pragma: no cover
            raise ValueError("pts must have shape (N,2)")
        if not isinstance(phys, PhysicalParams):                   # pragma: no cover
            raise TypeError("phys must be an instance of PhysicalParams")

        self.pts = pts.copy()
        self.N = pts.shape[0]
        self.phys = phys
        self.grid_shape = _validate_grid_shape(grid_shape)
        self._auto_halo_width = halo_width is None
        self.halo_width = float(4.01 * phys.r if halo_width is None else halo_width)
        if self.halo_width < 0.0:                   # pragma: no cover
            raise ValueError("halo_width must be non-negative")
        self.n_workers = int(
            n_workers if n_workers is not None else (os.cpu_count() or 1)
        )
        if self.n_workers <= 0:                   # pragma: no cover
            raise ValueError("n_workers must be positive")
        self.backend = backend
        self.domain_bounds = _validate_domain_bounds(domain_bounds)
        if decomposition_method not in ("dense", "sorted_x"):                   # pragma: no cover
            raise ValueError("decomposition_method must be 'dense' or 'sorted_x'")
        self.decomposition_method = decomposition_method
        self._preferred_areas = np.full(self.N, phys.A0, dtype=float)
        self._executor: ProcessPoolExecutor | None = None

    def __enter__(self) -> ParallelFiniteVoronoiSimulator:                   # pragma: no cover
        if self.n_workers > 1 and self._executor is None:
            from concurrent.futures import ProcessPoolExecutor

            self._executor = ProcessPoolExecutor(max_workers=self.n_workers)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:                   # pragma: no cover
        self.close()

    def close(self) -> None:                   # pragma: no cover
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def _make_domain_tasks(self, connect: bool, plot_mode: bool) -> list[_DomainTask]:
        domains = decompose_points(
            self.pts,
            self.grid_shape,
            self.halo_width,
            domain_bounds=self.domain_bounds,
            method=self.decomposition_method,
        )
        return [
            _DomainTask(
                domain=domain,
                local_preferred_areas=self._preferred_areas[domain.local_global_ids],
                phys=self.phys,
                backend=self.backend,
                connect=connect,
                plot_mode=plot_mode,
            )
            for domain in domains
        ]

    def build(self, connect: bool = False, plot_mode: bool = False) -> dict[str, object]:
        """ Build local finite Voronoi structures and merge global diagnostics.

        Do the following:
          - Decompose cell centers into owned domains plus halo points
          - Build a finite Voronoi diagram in each local subdomain
          - Extract diagnostics for owned cells
          - Merge forces and geometric quantities back into global indexing
          - Optionally collect global cell connectivity and per-domain plot data

        Args:
            connect: Whether to compute cell connectivity information.
                Setting this to ``False`` saves some computation time when
                connectivity is not needed.
                Note that the default is ``False``, unlike
                :py:meth:`FiniteVoronoiSimulator.build`, where the default is
                ``True``.
            plot_mode: Whether to include per-domain plotting diagnostics.
                If ``True``, use :py:func:`visualize_2d_parallel` for
                visualization.

        Returns:
            dict[str, object]: A dictionary containing merged forces and
            geometric properties with keys:

                - **forces**: (N,2) array of forces on cell centers
                - **areas**: (N,) array of cell areas
                - **perimeters**: (N,) array of cell perimeters
                - **arclens**: (N,) array of non-contacting edge (arc) lengths per cell
                - **coord_nums**: (N,) integer array of coordination numbers per cell
                - **connections**: (K,2) array of connected global cell index pairs
                - **pids**: (D,) integer array of process ids used for each domain
                - **plot_mode**: Whether per-domain plot diagnostics are included

            If *plot_mode* is ``True``, the dictionary also contains:

                - **owned_global_ids**: List of global cell ids owned by each domain
                - **diag_plot**: List of local plotting dictionaries, each with
                  ``vertices``, ``edges_type``, and ``regions`` for owned cells
        """
        tasks = self._make_domain_tasks(connect=connect, plot_mode=plot_mode)

        if self.n_workers == 1:
            results = [_build_domain(task) for task in tasks]
        elif self._executor is not None:                   # pragma: no cover
            results = list(self._executor.map(_build_domain, tasks))
        else:                   # pragma: no cover
            from concurrent.futures import ProcessPoolExecutor

            with ProcessPoolExecutor(max_workers=self.n_workers) as pool:
                results = list(pool.map(_build_domain, tasks))

        return self._merge_results(results, connect=connect, plot_mode=plot_mode)

    def _merge_results(
        self,
        results: list[dict[str, object]],
        connect: bool,
        plot_mode: bool,
    ) -> dict[str, object]:
        forces = np.zeros((self.N, 2), dtype=float)
        areas = np.zeros(self.N, dtype=float)
        perimeters = np.zeros(self.N, dtype=float)
        arclens = np.zeros(self.N, dtype=float)
        coord_nums = np.zeros(self.N, dtype=int)

        connection_chunks = []
        for result in results:
            owned_global_ids = np.asarray(result["owned_global_ids"], dtype=int)
            if owned_global_ids.size == 0:                   # pragma: no cover
                continue
            forces[owned_global_ids] = result["forces"]
            areas[owned_global_ids] = result["areas"]
            perimeters[owned_global_ids] = result["perimeters"]
            arclens[owned_global_ids] = result["arclens"]
            coord_nums[owned_global_ids] = result["coord_nums"]
            if connect and np.asarray(result["connections"]).size:
                connection_chunks.append(np.asarray(result["connections"], dtype=int))

        if connect and connection_chunks:
            connections = np.unique(np.concatenate(connection_chunks, axis=0), axis=0)
        else:
            connections = np.empty((0, 2), dtype=int)

        diag = {
            "forces": forces,
            "areas": areas,
            "perimeters": perimeters,
            "arclens": arclens,
            "coord_nums": coord_nums,
            "connections": connections,
            "plot_mode": plot_mode,
            "pids": np.asarray([result["pid"] for result in results], dtype=int),
        }
        if plot_mode:
            diag["owned_global_ids"] = [result["owned_global_ids"] for result in results]
            diag["diag_plot"] = [result["diag_plot"] for result in results]
        return diag

    def update_positions(self, pts: np.ndarray, A0: float | np.ndarray | None = None) -> None:      # pragma: no cover
        """
        Update cell center positions.

        .. note::
            If the number of cells changes, the preferred areas for all cells
            are reset to the default value---defined either at simulator
            instantiation or by :py:meth:`update_params`---unless *A0* is
            explicitly specified.

        Args:
            pts : New cell center positions.
            A0: Optional, set new preferred area(s).

        Raises:
            ValueError: If *pts* does not have shape (N,2).
            ValueError: If *A0* is an array and does not have shape (N,).
        """
        pts = np.asarray(pts, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError("pts must have shape (N,2)")

        N = pts.shape[0]
        self.pts = pts.copy()

        if N != self.N:
            self.N = N
            if A0 is None:
                self._preferred_areas = np.full(N, self.phys.A0, dtype=float)
            else:
                self.update_preferred_areas(A0)
        elif A0 is not None:
            self.update_preferred_areas(A0)

    def update_params(self, phys: PhysicalParams) -> None:                   # pragma: no cover
        """
        Update physical parameters.

        Args:
            phys: New :py:class:`PhysicalParams` object.

        Raises:
            TypeError: If *phys* is not an instance of :py:class:`PhysicalParams`.

        .. warning::
            This also resets all preferred cell areas to the new value of *A0*.
            If *halo_width* was not explicitly specified at instantiation, it
            also updates the halo width to ``4.01 * phys.r``.
        """
        if not isinstance(phys, PhysicalParams):
            raise TypeError("phys must be an instance of PhysicalParams")
        self.phys = phys
        if self._auto_halo_width:
            self.halo_width = 4.01 * phys.r
        self.update_preferred_areas(phys.A0)

    def update_preferred_areas(self, A0: float | np.ndarray) -> None:           # pragma: no cover
        """
        Update the preferred areas for all cells.

        Args:
            A0: New preferred area(s) for all cells, either as a scalar or
                as an array of shape (N,).

        Raises:
            ValueError: If *A0* does not match cell number.
        """
        arr = np.asarray(A0, dtype=float)
        if arr.ndim == 0:
            arr = np.full(self.N, float(arr), dtype=float)
        elif arr.shape == (1,):
            arr = np.full(self.N, float(arr[0]), dtype=float)
        elif arr.shape != (self.N,):
            raise ValueError(f"A0 must be scalar or have shape ({self.N},)")
        self._preferred_areas = arr

    @property
    def preferred_areas(self) -> np.ndarray:                # pragma: no cover
        """
        Return a copy of the preferred area array.

        Returns:
            numpy.ndarray: (N,) array of preferred cell areas.
        """
        return self._preferred_areas.copy()


__all__ = ["DomainDecomposition", "decompose_points", "ParallelFiniteVoronoiSimulator"]

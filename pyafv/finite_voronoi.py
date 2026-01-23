"""
Finite Voronoi Model Simulator in 2D
====================================
Created by Wei Wang, 2025.

Key public entry points:
------------------------
- FiniteVoronoiSimulator: configure parameters.
- build(): build the finite Voronoi diagram and compute forces.
- plot_2d(): plot the finite Voronoi diagram with matplotlib.
- update_points(): update cell center positions.
- update_params(): update physical parameters.
"""

# Enable postponed evaluation of annotations
from __future__ import annotations

# Only import typing modules when type checking, e.g., in VS Code or IDEs.
from typing import TYPE_CHECKING
if TYPE_CHECKING:                             # pragma: no cover
    from scipy.spatial import Voronoi
    import matplotlib.axes
    import typing

import numpy as np
from .physical_params import PhysicalParams


# ---- tiny helpers to avoid tiny allocations in hot loops ----
def _row_dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise dot product for 2D arrays with shape (N,2)."""
    return np.einsum("ij,ij->i", a, b)


class FiniteVoronoiSimulator:
    """Simulator for the active-finite-Voronoi (AFV) model.

    This class provides an interface to simulate the finite Voronoi model.
    It wraps around the two backend implementations, which may be
    either a Cython-accelerated version or a pure Python fallback.

    Args:
        pts (numpy.ndarray): (N,2) array of initial cell center positions.
        phys: Physical parameters used within this simulator.
        backend: Optional, specify "python" to force the use of the pure Python fallback implementation.
            Otherwise, the "cython" backend is used.

    Raises:
        ValueError: If *pts* does not have shape (N,2).
        TypeError: If *phys* is not an instance of :py:class:`PhysicalParams`.
    
    Warnings:
        If the Cython backend cannot be imported (unless *backend* is set to "python"),
        a **RuntimeWarning** is raised and the pure Python implementation is used instead.
    """

    def __init__(self, pts: np.ndarray, phys: PhysicalParams, backend: typing.Literal["cython", "python"] | None = None):
        """
        Constructor of the simulator.
        """
        from scipy.spatial import Voronoi
        self._voronoi = Voronoi
        
        pts = np.asarray(pts, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError("pts must have shape (N,2)")
        if not isinstance(phys, PhysicalParams):
            raise TypeError("phys must be an instance of PhysicalParams")
        
        self.pts = pts.copy()            # (N,2) array of initial points
        self.N = pts.shape[0]            # Number of points
        self.phys = phys
        self._preferred_areas = np.full(self.N, phys.A0, dtype=float)    # (N,) preferred areas A0
        
        if backend != "python":
            from .backend import backend_impl, _BACKEND_NAME
            self._BACKEND = _BACKEND_NAME
            self._impl = backend_impl

            if self._BACKEND not in {"cython", "numba"}:                 # pragma: no cover
                # raise warning to inform user about fallback
                import warnings
                warnings.warn(
                    "Could not import the Cython-built extension module. "
                    "Falling back to the pure Python implementation, which may be slower. "
                    "To enable the accelerated version, ensure that all dependencies are installed.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        else:  # force the use of the pure Python fallback implementation.
            self._BACKEND = "python"
            from . import cell_geom_fallback as _impl
            self._impl = _impl

    # --------------------- Voronoi construction & extension ---------------------
    def _build_voronoi_with_extensions(self) -> tuple[Voronoi, np.ndarray, list[list[int]], int, dict[tuple[int,int], int], dict[int, list[int]]]:
        """
        Build standard Voronoi structure for current points.

        For N<=2, emulate regions.
        For N>=3, extend infinite ridges, add extension vertices, and update
        regions accordingly. Return the augmented structures.
        
        .. warning::
            
            This is an internal method. Use with caution.

        Returns: 
            tuple[scipy.spatial.Voronoi, numpy.ndarray, list[list[int]], int, dict[tuple[int,int], int], dict[int, list[int]]] : A *tuple* containing:
            
                - **vor**: SciPy Voronoi object for current points with extensions.
                - **vertices_all**: (M,2) array of all Voronoi vertices including extensions.
                - **ridge_vertices_all**: List of lists of vertex indices for each ridge, including extensions.
                - **num_vertices**: Number of Voronoi vertices before adding extension.
                - **vertexpair2ridge**: *dict* mapping vertex index pairs to ridge index.
                - **vertex_points**: *dict* mapping vertex index to list of associated point indices.
        """

        Voronoi = self._voronoi

        r = self.phys.r
        pts = self.pts
        N = self.N

        # Special handling: N == 1, 2
        if N == 1:
            vor = Voronoi(np.random.rand(3, 2))
            vor.points[:N] = pts
            vor.vertices = np.array([]).reshape(-1, 2)

            vor.regions = [[]]
            vor.ridge_vertices = np.array([]).reshape(-1, 2)
            vor.point_region = np.array([0])
            vor.ridge_points = np.array([]).reshape(-1, 2)
            
            vertices_all = vor.vertices
            ridge_vertices_all = vor.ridge_vertices
            num_vertices = len(vor.vertices)
            

        if N == 2:
            vor = Voronoi(np.random.rand(3, 2))
            vor.points[:N] = pts

            p1, p2 = pts
            center = (p1 + p2) / 2.0

            t = p1 - p2
            t_norm = np.linalg.norm(t)
            t /= t_norm

            n = np.array([-t[1], t[0]]) # Perpendicular vector of t

            if t_norm >= 2 * r:
                vor.vertices = np.array([]).reshape(-1, 2)
                vor.regions = [[], []]
                vor.ridge_vertices = np.array([]).reshape(-1, 2)
                vor.ridge_points = np.array([]).reshape(-1, 2)
            else:
                v1 = center + 2 * r * n
                v2 = center - 2 * r * n
                v3 = center + 3 * r * (p1 - center) / np.linalg.norm(p2 - center)
                v4 = center + 3 * r * (p2 - center) / np.linalg.norm(p2 - center)
                vor.vertices = np.array([v1, v2, v3, v4]).reshape(-1, 2)
                vor.ridge_vertices = np.array([[0, 1], [1, 2], [0, 2], [0, 3], [1, 3]])
                vor.regions = [[0, 1, 2], [0, 1, 3]]
                vor.ridge_points = np.array([[0, 1], [-1, 0], [-1, 0], [-1, 1], [-1, 1]])

            vor.point_region = np.array([0, 1])

            vertices_all = vor.vertices
            ridge_vertices_all = vor.ridge_vertices
            num_vertices = len(vor.vertices)

        
        # N >= 3 (vectorized main path)
        if N >= 3:
            vor = Voronoi(pts)
            """
            Basic info from Voronoi object:
            -------------------------------
            vor.vertices       # (K,2) Voronoi vertices (finite)
            vor.ridge_points   # (R,2) pairs of input point indices sharing a Voronoi ridge
            vor.ridge_vertices # list of vertex-index lists (may contain -1 for infinity)
            vor.point_region   # for each input point, the region index
            vor.regions        # list of regions; each is a list of vertex indices (may include -1)
            """
            center = np.mean(pts, axis=0)

            span_x = np.ptp(vor.vertices[:, 0])  # span in x
            span_y = np.ptp(vor.vertices[:, 1])  # span in y
            pts_span_x = np.ptp(pts[:, 0])  # span in x
            pts_span_y = np.ptp(pts[:, 1])  # span in y
            span = max(span_x, span_y, pts_span_x, pts_span_y, 10. * r)  # overall span

            # Base copies
            vertices_base = vor.vertices  # (K,2)
            rv_arr = np.asarray(vor.ridge_vertices, dtype=int)  # (R,2) may contain -1
            rp_arr = np.asarray(vor.ridge_points, dtype=int)    # (R,2)
            
            # Remove -1 from regions (we will append extension ids later)
            vor.regions = [[vid for vid in region if vid >= 0] for region in vor.regions]

            # Identify ridges with an infinite endpoint
            inf_mask = (rv_arr == -1).any(axis=1)               # (R,)
            num_inf = int(inf_mask.sum())

            if num_inf > 0:
                rv_inf = rv_arr[inf_mask]                       # (M,2)
                rp_inf = rp_arr[inf_mask]                       # (M,2)

                # finite endpoint index per infinite ridge
                v_idx_finite = np.where(rv_inf[:, 0] != -1, rv_inf[:, 0], rv_inf[:, 1])  # (M,)

                # geometry for normals
                p1 = pts[rp_inf[:, 0]]                          # (M,2)
                p2 = pts[rp_inf[:, 1]]                          # (M,2)
                mid = (p1 + p2) / 2.0                           # (M,2)

                t = p1 - p2                                     # (M,2)
                t_norm = np.linalg.norm(t, axis=1, keepdims=True)  # (M,1)
                t_unit = t / t_norm

                # (M,2), perpendicular
                n = np.column_stack([-t_unit[:, 1], t_unit[:, 0]])

                # Ensure "outward" normal
                sign = np.einsum("ij,ij->i", (mid - center), n)     # (M,)
                n[sign < 0] *= -1.0

                # Build extension points
                ext = vertices_base[v_idx_finite] + (100.0 * span) * n  # (M,2), long rays (extension must be long enough!)

                # Concatenate once
                K = vertices_base.shape[0]
                vertices_all = np.vstack([vertices_base, ext])       # (K+M,2)

                # New vertex ids for extensions
                ext_ids = np.arange(K, K + num_inf, dtype=int)       # (M,)

                # Replace -1 with ext_ids in a vectorized way
                rv_new = rv_arr.copy()                                # (R,2)
                rv_sub = rv_new[inf_mask]                             # view (M,2)

                pos0 = (rv_sub[:, 0] == -1)                          # (M,)
                rv_sub[pos0, 0] = ext_ids[pos0]
                rv_sub[~pos0, 1] = ext_ids[~pos0]
                rv_new[inf_mask] = rv_sub

                ridge_vertices_all = rv_new.tolist()

                # Append extension id to both adjacent regions (list-of-lists => tiny loop)
                for m in range(num_inf):
                    i1, i2 = rp_inf[m]
                    e = ext_ids[m]
                    region_id = vor.point_region[i1]
                    vor.regions[region_id].append(e)
                    region_id = vor.point_region[i2]
                    vor.regions[region_id].append(e)
            else:
                vertices_all = vertices_base.copy()
                ridge_vertices_all = rv_arr.tolist()

            # number of native (finite) vertices
            num_vertices = len(vor.vertices)

        # Build vertexpair2ridge and vertex_points using Cython/Python backend function
        vertexpair2ridge, vertex_points = self._impl.build_vertexpair_and_vertexpoints(ridge_vertices_all, vor.ridge_points, num_vertices, N)

        return vor, vertices_all, ridge_vertices_all, num_vertices, vertexpair2ridge, vertex_points

    # --------------------- Geometry & energy contributions per cell ---------------------
    def _per_cell_geometry(self, vor: Voronoi, vertices_all: np.ndarray, ridge_vertices_all: np.ndarray, num_vertices: int, vertexpair2ridge: dict[tuple[int, int], int]) -> dict[str,object]:
        """
        Build the finite-Voronoi per-cell geometry and energy contributions.

        Iterate each cell to:
          - sort polygon/arc vertices around each cell
          - classify edges (1 = straight Voronoi edge; 0 = circular arc)
          - compute area/perimeter for each cell
          - accumulate derivatives w.r.t. vertices (dA_poly/dh, dP_poly/dh)
          - register "outer" vertices created at arc intersections and track their point pairs

        .. warning::
            This is an internal method. Use with caution.

        Args:
            vor: SciPy Voronoi object for current points with extensions.
            vertices_all: (M,2) array of all Voronoi vertices including extensions.
            ridge_vertices_all: (R,2) array of vertex indices for each ridge.
            num_vertices: Number of Voronoi vertices before adding extension.
            vertexpair2ridge: *dict* mapping vertex index pairs to ridge index.
        
        Returns:
            dict[str, object]: A diagnostics dictionary containing:

                - **vertex_in_id**: *set* of inner vertex ids.
                - **vertex_out_id**: *set* of outer vertex ids.
                - **vertices_out**: (L,2) array of outer vertex coordinates.
                - **vertex_out_points**: (L,2) array of point index pairs associated with each outer vertex.
                - **vertex_out_da_dtheta**: Array of dA/dtheta for all outer vertices.
                - **vertex_out_dl_dtheta**: Array of dL/dtheta for all outer vertices.
                - **dA_poly_dh**: Array of dA_polygon/dh for each vertex.
                - **dP_poly_dh**: Array of dP_polygon/dh for each vertex.
                - **area_list**: Array of polygon areas for each cell.
                - **perimeter_list**: Array of polygon perimeters for each cell.
                - **point_edges_type**: List of lists of edge types per cell.
                - **point_vertices_f_idx**: List of lists of vertex ids per cell.
                - **num_vertices_ext**: Number of vertices including infinite extension vertices.
        """
        N = self.N
        r = self.phys.r
        A0_list = self._preferred_areas
        P0 = self.phys.P0
        pts = self.pts

        num_vertices_ext = len(vertices_all)  # number of vertices with infinite extension points

        rv = np.asarray(ridge_vertices_all, dtype=int)           # (R,2)
        rp = np.asarray(vor.ridge_points, dtype=int)             # (R,2)
        num_ridges = rp.shape[0]

        # init outer-vertex arrays (same shapes you used)
        vertices_out = np.zeros((2 * num_ridges, 2), dtype=float)
        vertex_out_points = np.zeros((2 * num_ridges, 2), dtype=int)

        # unpack ridge endpoints and vertex indices
        p1 = rp[:, 0].copy()
        p2 = rp[:, 1].copy()
        v1 = rv[:, 0].copy()
        v2 = rv[:, 1].copy()

        valid_pts = (p1 >= 0) & (p2 >= 0)

        # coordinates
        P1 = np.zeros((num_ridges, 2), dtype=float); P2 = np.zeros((num_ridges, 2), dtype=float)
        P1[valid_pts] = self.pts[p1[valid_pts]]
        P2[valid_pts] = self.pts[p2[valid_pts]]
        V1 = vertices_all[v1]
        V2 = vertices_all[v2]

        # enforce V1->V2 clockwise around p1 (swap only p1<->p2, not v1/v2)
        P12 = P2 - P1
        V12 = V2 - V1
        swap = (P12[:, 0] * V12[:, 1] - P12[:, 1] * V12[:, 0]) > 0
        p1_sw = p1.copy(); p2_sw = p2.copy()
        p1_sw[swap] = p2[swap]; p2_sw[swap] = p1[swap]
        p1, p2 = p1_sw, p2_sw
        P1_sw = P1.copy(); P2_sw = P2.copy()
        P1[swap] = P2_sw[swap]; P2[swap] = P1_sw[swap]

        # "inner" tests relative to p1 (keep your exact choices)
        r = self.phys.r
        d1 = np.linalg.norm(V1 - P1, axis=1)
        d2 = np.linalg.norm(V2 - P1, axis=1)
        inner1 = (d1 <= r) & valid_pts
        inner2 = (d2 <= r) & valid_pts

        # segment/intersection
        dV    = V2 - V1
        segL  = np.linalg.norm(dV, axis=1)
        denom = np.where(segL > 0.0, segL, 1.0)
        dx, dy = dV[:, 0], dV[:, 1]
        x, y   = P1[:, 0], P1[:, 1]
        x1, y1 = V1[:, 0], V1[:, 1]
        t  = ((x - x1) * dx + (y - y1) * dy) / denom

        # mid-point C = (p1+p2)/2 (your exact choice)
        C = 0.5 * (P1 + P2)
        cx, cy = C[:, 0], C[:, 1]
        t1 = -t
        t2 = t1 + denom

        d  = np.linalg.norm(C - P1, axis=1)
        has_int = (d < r) & valid_pts
        tr = np.full_like(d, np.nan)
        tr[has_int] = np.sqrt(r * r - d[has_int] * d[has_int])

        cond1 = (-tr < t2) & (-tr > t1) & valid_pts
        cond2 = ( tr < t2) & ( tr > t1) & valid_pts

        invL = np.where(denom > 0.0, 1.0 / denom, 0.0)
        xr1 = cx - tr * dx * invL
        yr1 = cy - tr * dy * invL
        xr2 = cx + tr * dx * invL
        yr2 = cy + tr * dy * invL

        # fill outer-vertex arrays only where intersections happen
        idx1 = np.where(cond1)[0]
        if idx1.size:
            vertices_out[2 * idx1 + 0, 0] = xr1[idx1]
            vertices_out[2 * idx1 + 0, 1] = yr1[idx1]
            pairs1 = np.sort(np.stack([p1[idx1], p2[idx1]], axis=1), axis=1)
            vertex_out_points[2 * idx1 + 0] = pairs1

        idx2 = np.where(cond2)[0]
        if idx2.size:
            vertices_out[2 * idx2 + 1, 0] = xr2[idx2]
            vertices_out[2 * idx2 + 1, 1] = yr2[idx2]
            pairs2 = np.sort(np.stack([p1[idx2], p2[idx2]], axis=1), axis=1)
            vertex_out_points[2 * idx2 + 1] = pairs2

        # sets (same semantics as before)
        vertex_out_id = set((num_vertices_ext + np.where((np.arange(2 * num_ridges) % 2 == 0) & cond1.repeat(2))[0]).tolist())
        vertex_out_id.update((num_vertices_ext + np.where((np.arange(2 * num_ridges) % 2 == 1) & cond2.repeat(2))[0]).tolist())

        vertex_in_id = set(v1[inner1].astype(int).tolist())
        vertex_in_id.update(v2[inner2].astype(int).tolist())

        # -------- NEW: packed arrays instead of ridge_info dict --------
        # each endpoint has up to 3 entries; absent slots are -1
        p1_edges_pack  = np.full((num_ridges, 3), -1, dtype=int)
        p1_verts_pack  = np.full((num_ridges, 3), -1, dtype=int)
        p2_edges_pack  = np.full((num_ridges, 3), -1, dtype=int)
        p2_verts_pack  = np.full((num_ridges, 3), -1, dtype=int)

        out_id1 = num_vertices_ext + (2 * np.arange(num_ridges) + 0)
        out_id2 = out_id1 + 1

        # p1 order: [inner1 -> (1,v1)], [cond1 -> (1,out1)], [cond2 -> (0,out2)]
        p1_edges_pack[inner1, 0] = 1
        p1_verts_pack[inner1, 0] = v1[inner1]

        p1_edges_pack[cond1, 1] = 1
        p1_verts_pack[cond1, 1] = out_id1[cond1]

        p1_edges_pack[cond2, 2] = 0
        p1_verts_pack[cond2, 2] = out_id2[cond2]

        # p2 was "append then reverse", which yields final order: [inner2, cond2, cond1]
        p2_edges_pack[inner2, 0] = 1
        p2_verts_pack[inner2, 0] = v2[inner2]

        p2_edges_pack[cond2, 1] = 1
        p2_verts_pack[cond2, 1] = out_id2[cond2]

        p2_edges_pack[cond1, 2] = 0
        p2_verts_pack[cond1, 2] = out_id1[cond1]

        # append outer-vertex slots (unused rows stay zero like before)
        vertices_all = np.vstack([vertices_all, vertices_out])

        # --------------------------------------------------
        # Part 1 in Cython/Python backend
        # --------------------------------------------------
        vor_regions = self._impl.pad_regions(vor.regions)            # (R, Kmax) int64 with -1 padding
        point_edges_type, point_vertices_f_idx = self._impl.build_point_edges(
            vor_regions, vor.point_region.astype(np.int64),
            vertices_all.astype(np.float64), pts.astype(np.float64),
            int(num_vertices), vertexpair2ridge, 
            p1.astype(np.int64), p1_edges_pack.astype(np.int64), p1_verts_pack.astype(np.int64),
            p2.astype(np.int64), p2_edges_pack.astype(np.int64), p2_verts_pack.astype(np.int64),
        )

        # --------------------------------------------------
        # Part 2 in Cython/Python backend
        # --------------------------------------------------
        vertex_out_da_dtheta, vertex_out_dl_dtheta, dA_poly_dh, dP_poly_dh, area_list, perimeter_list = self._impl.compute_vertex_derivatives(
            point_edges_type,            # list-of-lists / arrays of edge types
            point_vertices_f_idx,        # list-of-lists / arrays of vertex ids
            vertices_all.astype(np.float64, copy=False),
            pts.astype(np.float64, copy=False),
            float(r),
            A0_list.astype(np.float64, copy=False),
            float(P0),
            int(num_vertices_ext),
            int(num_ridges),
            vertex_out_points.astype(np.int64, copy=False)
        )

        diagnostics = dict(
            vertex_in_id=set(vertex_in_id),
            vertex_out_id=set(vertex_out_id),
            vertices_out=vertices_out,
            vertex_out_points=vertex_out_points,
            vertex_out_da_dtheta=vertex_out_da_dtheta,
            vertex_out_dl_dtheta=vertex_out_dl_dtheta,
            dA_poly_dh=dA_poly_dh,
            dP_poly_dh=dP_poly_dh,
            area_list=area_list,
            perimeter_list=perimeter_list,
            point_edges_type=point_edges_type,
            point_vertices_f_idx=point_vertices_f_idx,
            num_vertices_ext=num_vertices_ext,
        )
        return diagnostics, vertices_all

    # --------------------- Force assembly ---------------------
    def _assemble_forces(self, vertices_all: np.ndarray, num_vertices_ext: int,
                         vertex_points: dict[int, list[int]], vertex_in_id: list[int], vertex_out_id: list[int],
                         vertex_out_points: list[list[int]], vertex_out_da_dtheta: np.ndarray,
                         vertex_out_dl_dtheta: np.ndarray, dA_poly_dh: np.ndarray, dP_poly_dh: np.ndarray,
                         area_list: np.ndarray, perimeter_list: np.ndarray) -> np.ndarray:
        """
        Assemble forces on cell centers from polygon and arc contributions.
        """
        N = self.N
        r = self.phys.r
        A0_list = self._preferred_areas
        P0 = self.phys.P0
        KA = self.phys.KA
        KP = self.phys.KP
        Lambda = self.phys.lambda_tension
        pts = self.pts

        dE_poly_dh = 2.0 * (KA * dA_poly_dh + KP * dP_poly_dh)

        fx = np.zeros(N)
        fy = np.zeros(N)

        # ===============================================================
        # (1) Inner vertices contributions — vectorized + bincount scatter
        # ===============================================================
        if len(vertex_in_id) > 0:
            H = np.asarray(list(vertex_in_id), dtype=int)            # (H,)
            # unpack triples (i,j,k) for each inner vertex
            I = np.empty(len(H), dtype=int)
            J = np.empty(len(H), dtype=int)
            K = np.empty(len(H), dtype=int)
            for t, h in enumerate(H):
                I[t], J[t], K[t] = vertex_points[h]

            ri = pts[I]  # (H,2)
            rj = pts[J]
            rk = pts[K]

            rj_minus_rk = rj - rk
            ri_minus_rj = ri - rj
            ri_minus_rk = ri - rk
            rj_minus_ri = -ri_minus_rj
            rk_minus_ri = -ri_minus_rk
            rk_minus_rj = rk - rj

            D0 = _row_dot(ri - rj, np.column_stack((rj_minus_rk[:,1], -rj_minus_rk[:,0])))  # cross2(ri-rj, rj-rk)
            # rewrite cross robustly:
            D0 = (ri[:,0]-rj[:,0])*(rj_minus_rk[:,1]) - (ri[:,1]-rj[:,1])*(rj_minus_rk[:,0])
            D = 2.0 * (D0 ** 2)

            # alphas
            alpha_i = _row_dot(rj_minus_rk, rj_minus_rk) * _row_dot(ri_minus_rj, ri_minus_rk) / D
            alpha_j = _row_dot(ri_minus_rk, ri_minus_rk) * _row_dot(rj_minus_ri, rj_minus_rk) / D
            alpha_k = _row_dot(ri_minus_rj, ri_minus_rj) * _row_dot(rk_minus_ri, rk_minus_rj) / D

            # d_alpha_j / d_ri and d_alpha_k / d_ri
            cross_z = np.column_stack((rj_minus_rk[:, 1], -rj_minus_rk[:, 0]))  # (H,2)
            term_j_ri = (rk_minus_rj / _row_dot(rj_minus_ri, rj_minus_rk)[:, None]) + 2.0 * (ri_minus_rk / _row_dot(ri_minus_rk, ri_minus_rk)[:, None]) - 2.0 * (cross_z / D0[:, None])
            term_k_ri = (rj_minus_rk / _row_dot(rk_minus_ri, rk_minus_rj)[:, None]) + 2.0 * (ri_minus_rj / _row_dot(ri_minus_rj, ri_minus_rj)[:, None]) - 2.0 * (cross_z / D0[:, None])
            d_alpha_j_d_ri = (alpha_j[:, None] * term_j_ri)  # (H,2)
            d_alpha_k_d_ri = (alpha_k[:, None] * term_k_ri)

            d_h_in_d_xi = alpha_i[:, None] * np.array([1.0, 0.0]) + d_alpha_j_d_ri[:, [0]] * (rj - ri) + d_alpha_k_d_ri[:, [0]] * (rk - ri)
            d_h_in_d_yi = alpha_i[:, None] * np.array([0.0, 1.0]) + d_alpha_j_d_ri[:, [1]] * (rj - ri) + d_alpha_k_d_ri[:, [1]] * (rk - ri)

            # d_alpha_i / d_rj and d_alpha_k / d_rj
            cross_z = np.column_stack((-(ri_minus_rk)[:, 1], (ri_minus_rk)[:, 0]))
            term_i_rj = (rk_minus_ri / _row_dot(ri_minus_rj, ri_minus_rk)[:, None]) + 2.0 * (rj_minus_rk / _row_dot(rj_minus_rk, rj_minus_rk)[:, None]) - 2.0 * (cross_z / D0[:, None])
            term_k_rj = (ri_minus_rk / _row_dot(rk_minus_rj, rk_minus_ri)[:, None]) + 2.0 * (rj_minus_ri / _row_dot(rj_minus_ri, rj_minus_ri)[:, None]) - 2.0 * (cross_z / D0[:, None])
            d_alpha_i_d_rj = (alpha_i[:, None] * term_i_rj)
            d_alpha_k_d_rj = (alpha_k[:, None] * term_k_rj)

            d_h_in_d_xj = d_alpha_i_d_rj[:, [0]] * (ri - rj) + alpha_j[:, None] * np.array([1.0, 0.0]) + d_alpha_k_d_rj[:, [0]] * (rk - rj)
            d_h_in_d_yj = d_alpha_i_d_rj[:, [1]] * (ri - rj) + alpha_j[:, None] * np.array([0.0, 1.0]) + d_alpha_k_d_rj[:, [1]] * (rk - rj)

            # d_alpha_i / d_rk and d_alpha_j / d_rk
            cross_z = np.column_stack(((ri_minus_rj)[:, 1], -(ri_minus_rj)[:, 0]))
            term_i_rk = (rj_minus_ri / _row_dot(ri_minus_rk, ri_minus_rj)[:, None]) + 2.0 * (rk_minus_rj / _row_dot(rk_minus_rj, rk_minus_rj)[:, None]) - 2.0 * (cross_z / D0[:, None])
            term_j_rk = (ri_minus_rj / _row_dot(rj_minus_rk, rj_minus_ri)[:, None]) + 2.0 * (rk_minus_ri / _row_dot(rk_minus_ri, rk_minus_ri)[:, None]) - 2.0 * (cross_z / D0[:, None])
            d_alpha_i_d_rk = (alpha_i[:, None] * term_i_rk)
            d_alpha_j_d_rk = (alpha_j[:, None] * term_j_rk)

            d_h_in_d_xk = d_alpha_i_d_rk[:, [0]] * (ri - rk) + d_alpha_j_d_rk[:, [0]] * (rj - rk) + alpha_k[:, None] * np.array([1.0, 0.0])
            d_h_in_d_yk = d_alpha_i_d_rk[:, [1]] * (ri - rk) + d_alpha_j_d_rk[:, [1]] * (rj - rk) + alpha_k[:, None] * np.array([0.0, 1.0])

            deh = dE_poly_dh[H]  # (H,2)
            contrib_x_i = _row_dot(deh, d_h_in_d_xi)
            contrib_x_j = _row_dot(deh, d_h_in_d_xj)
            contrib_x_k = _row_dot(deh, d_h_in_d_xk)
            contrib_y_i = _row_dot(deh, d_h_in_d_yi)
            contrib_y_j = _row_dot(deh, d_h_in_d_yj)
            contrib_y_k = _row_dot(deh, d_h_in_d_yk)

            # bincount-based scatter (faster than repeated np.add.at)
            fx += (
                np.bincount(I, weights=contrib_x_i, minlength=N)
              + np.bincount(J, weights=contrib_x_j, minlength=N)
              + np.bincount(K, weights=contrib_x_k, minlength=N)
            )
            fy += (
                np.bincount(I, weights=contrib_y_i, minlength=N)
              + np.bincount(J, weights=contrib_y_j, minlength=N)
              + np.bincount(K, weights=contrib_y_k, minlength=N)
            )

        # ===============================================================
        # (2) Outer vertices contributions — vectorized; bincount scatter
        # ===============================================================
        dA_arc_dr = np.zeros((N, 2))
        dP_arc_dr = np.zeros((N,2))
        dL_dr = np.zeros((N,2))

        if len(vertex_out_id) > 0:
            Vsel = np.asarray(vertex_out_id, dtype=int)           # absolute vertex IDs in vertices_all
            h_idx = Vsel - num_vertices_ext                       # rows into vertex_out_* arrays
            # guard against any accidental negatives / out-of-range
            valid_mask = (h_idx >= 0) & (h_idx < len(vertex_out_points))
            if np.any(valid_mask):
                Vsel = Vsel[valid_mask]
                h_idx = h_idx[valid_mask]

                # geometry slices
                h_out = vertices_all[Vsel]                        # (M,2)
                IJ = np.asarray(vertex_out_points, dtype=int)[h_idx]  # (M,2)
                I = IJ[:, 0]
                J = IJ[:, 1]

                ri = pts[I]            # (M,2)
                rj = pts[J]            # (M,2)
                rij_vec = ri - rj
                rij = np.linalg.norm(rij_vec, axis=1)             # (M,)
                root = np.sqrt(4.0 * (r ** 2) - (rij ** 2))

                # sign based on orientation: sign(cross(h_out-rj, ri-rj))
                sign = np.sign((h_out[:, 0] - rj[:, 0]) * (ri[:, 1] - rj[:, 1])
                            - (h_out[:, 1] - rj[:, 1]) * (ri[:, 0] - rj[:, 0]))

                x_unit = np.array([1.0, 0.0])[None, :]
                y_unit = np.array([0.0, 1.0])[None, :]

                cross_z = rij_vec[:, [1]] * x_unit - rij_vec[:, [0]] * y_unit  # (M,2)
                denom = (np.maximum(root[:, None], self.phys.delta) * (rij ** 3)[:, None])  # small offset to avoid singularities

                dx_terms = - (2.0 * (r ** 2) * rij_vec[:, [0]] * cross_z / denom) \
                        - (root / (2.0 * rij))[:, None] * y_unit
                dy_terms = - (2.0 * (r ** 2) * rij_vec[:, [1]] * cross_z / denom) \
                        + (root / (2.0 * rij))[:, None] * x_unit

                d_h_out_d_xi = (x_unit / 2.0) + sign[:, None] * dx_terms
                d_h_out_d_yi = (y_unit / 2.0) + sign[:, None] * dy_terms
                d_h_out_d_xj = (x_unit / 2.0) - sign[:, None] * dx_terms
                d_h_out_d_yj = (y_unit / 2.0) - sign[:, None] * dy_terms

                # polygon part on these selected outer vertices
                deh_out = dE_poly_dh[Vsel]  # (M,2)

                fx += (
                    np.bincount(I, weights=_row_dot(deh_out, d_h_out_d_xi), minlength=N)
                    + np.bincount(J, weights=_row_dot(deh_out, d_h_out_d_xj), minlength=N)
                )
                fy += (
                    np.bincount(I, weights=_row_dot(deh_out, d_h_out_d_yi), minlength=N)
                    + np.bincount(J, weights=_row_dot(deh_out, d_h_out_d_yj), minlength=N)
                )

                # ---- arc angle sensitivities (per-cell accumulators) ----
                u_i = h_out - ri
                u_j = h_out - rj
                inv_ui2 = 1.0 / _row_dot(u_i, u_i)
                inv_uj2 = 1.0 / _row_dot(u_j, u_j)
                u_perp_i = np.column_stack((-u_i[:, 1], u_i[:, 0])) * inv_ui2[:, None]
                u_perp_j = np.column_stack((-u_j[:, 1], u_j[:, 0])) * inv_uj2[:, None]

                d_theta_i_d_xj = _row_dot(d_h_out_d_xj, u_perp_i)
                d_theta_i_d_yj = _row_dot(d_h_out_d_yj, u_perp_i)
                d_theta_i_d_xi = -d_theta_i_d_xj
                d_theta_i_d_yi = -d_theta_i_d_yj

                d_theta_j_d_xi = _row_dot(d_h_out_d_xi, u_perp_j)
                d_theta_j_d_yi = _row_dot(d_h_out_d_yi, u_perp_j)
                d_theta_j_d_xj = -d_theta_j_d_xi
                d_theta_j_d_yj = -d_theta_j_d_yi

                # weights (only for the selected outer vertices h_idx)
                v_da = vertex_out_da_dtheta[h_idx]   # (M,2)
                v_dl = vertex_out_dl_dtheta[h_idx]   # (M,2)

                Ai_w_i = (area_list[I]      - A0_list[I]) * v_da[:, 0]
                Aj_w_j = (area_list[J]      - A0_list[J]) * v_da[:, 1]
                Pi_w_i = (perimeter_list[I] - P0) * v_dl[:, 0]
                Pj_w_j = (perimeter_list[J] - P0) * v_dl[:, 1]

                # accumulate with bincount
                dA_arc_dr[:, 0] += np.bincount(I, Ai_w_i * d_theta_i_d_xi + Aj_w_j * d_theta_j_d_xi, minlength=N)
                dA_arc_dr[:, 0] += np.bincount(J, Ai_w_i * d_theta_i_d_xj + Aj_w_j * d_theta_j_d_xj, minlength=N)
                dA_arc_dr[:, 1] += np.bincount(I, Ai_w_i * d_theta_i_d_yi + Aj_w_j * d_theta_j_d_yi, minlength=N)
                dA_arc_dr[:, 1] += np.bincount(J, Ai_w_i * d_theta_i_d_yj + Aj_w_j * d_theta_j_d_yj, minlength=N)

                dP_arc_dr[:, 0] += np.bincount(I, Pi_w_i * d_theta_i_d_xi + Pj_w_j * d_theta_j_d_xi, minlength=N)
                dP_arc_dr[:, 0] += np.bincount(J, Pi_w_i * d_theta_i_d_xj + Pj_w_j * d_theta_j_d_xj, minlength=N)
                dP_arc_dr[:, 1] += np.bincount(I, Pi_w_i * d_theta_i_d_yi + Pj_w_j * d_theta_j_d_yi, minlength=N)
                dP_arc_dr[:, 1] += np.bincount(J, Pi_w_i * d_theta_i_d_yj + Pj_w_j * d_theta_j_d_yj, minlength=N)

                # line-tension contributions for Lambda
                dL_arc_x = v_dl[:, 0] * d_theta_i_d_xi + v_dl[:, 1] * d_theta_j_d_xi
                dL_arc_y = v_dl[:, 0] * d_theta_i_d_yi + v_dl[:, 1] * d_theta_j_d_yi
                dL_arc_xJ = v_dl[:, 0] * d_theta_i_d_xj + v_dl[:, 1] * d_theta_j_d_xj
                dL_arc_yJ = v_dl[:, 0] * d_theta_i_d_yj + v_dl[:, 1] * d_theta_j_d_yj

                dL_dr[:, 0] += np.bincount(I, dL_arc_x,  minlength=N)
                dL_dr[:, 1] += np.bincount(I, dL_arc_y,  minlength=N)
                dL_dr[:, 0] += np.bincount(J, dL_arc_xJ, minlength=N)
                dL_dr[:, 1] += np.bincount(J, dL_arc_yJ, minlength=N)

        # combine arc terms
        dE_arc_dr = 2.0 * (KA * dA_arc_dr + KP * dP_arc_dr) + Lambda * dL_dr

        fx = -(fx + dE_arc_dr[:, 0])
        fy = -(fy + dE_arc_dr[:, 1])

        F = np.zeros((N, 2), dtype=float)
        F[:, 0] = fx
        F[:, 1] = fy
        return F

    # --------------------- One integration step ---------------------
    def build(self, connect: bool = True) -> dict[str, object]:
        """ Build the finite-Voronoi structure and compute forces, returning a dictionary of diagnostics.

        Do the following:
          - Build Voronoi (+ extensions)
          - Get cell connectivity
          - Compute per-cell quantities and derivatives
          - Assemble forces
        
        Args:
            connect: Whether to compute cell connectivity information.
                Setting this to ``False`` saves some computation time (though
                very marginal) when connectivity is not needed.
          
        Returns:
            dict[str, object]: A dictionary containing forces and geometric properties with keys:

                - **forces**: (N,2) array of forces on cell centers
                - **areas**: (N,) array of cell areas
                - **perimeters**: (N,) array of cell perimeters
                - **vertices**: (M,2) array of all Voronoi + extension vertices
                - **edges_type**: List-of-lists of edge types per cell (1=straight, 0=circular arc)
                - **regions**: List-of-lists of vertex indices per cell
                - **connections**: (K,2) array of connected cell index pairs
        """
        (vor, vertices_all, ridge_vertices_all, num_vertices,
            vertexpair2ridge, vertex_points) = self._build_voronoi_with_extensions()
        
        # Get connectivity info
        if connect:
            connections = self._get_connections(vor.ridge_points, vertices_all, ridge_vertices_all)
        else:           # pragma: no cover
            connections = np.empty((0,2), dtype=int)

        geom, vertices_all = self._per_cell_geometry(vor, vertices_all, ridge_vertices_all, num_vertices, vertexpair2ridge)

        F = self._assemble_forces(
            vertices_all=vertices_all,
            num_vertices_ext=geom["num_vertices_ext"],
            vertex_points=vertex_points,
            vertex_in_id=list(geom["vertex_in_id"]),
            vertex_out_id=list(geom["vertex_out_id"]),
            vertex_out_points=geom["vertex_out_points"],
            vertex_out_da_dtheta=geom["vertex_out_da_dtheta"],
            vertex_out_dl_dtheta=geom["vertex_out_dl_dtheta"],
            dA_poly_dh=geom["dA_poly_dh"],
            dP_poly_dh=geom["dP_poly_dh"],
            area_list=geom["area_list"],
            perimeter_list=geom["perimeter_list"],
        )

        return dict(
            forces=F,
            areas=geom["area_list"],
            perimeters=geom["perimeter_list"],
            vertices=vertices_all,
            edges_type=geom["point_edges_type"],
            regions=geom["point_vertices_f_idx"],
            connections=connections,
        )

    # --------------------- 2D plotting utilities ---------------------
    def plot_2d(self, ax: matplotlib.axes.Axes | None = None, show: bool = False) -> matplotlib.axes.Axes:
        """
        Build the finite-Voronoi structure and render a 2D snapshot.

        Basically a wrapper of :py:meth:`_build_voronoi_with_extensions` and :py:meth:`_per_cell_geometry` functions + plot.

        Args:
            ax: If provided, draw into this axes; otherwise get the current axes.
            show: Whether to call ``plt.show()`` at the end.
        
        Returns:
            The matplotlib axes containing the plot.
        """
        (vor, vertices_all, ridge_vertices_all, num_vertices,
            vertexpair2ridge, vertex_points) = self._build_voronoi_with_extensions()

        geom, vertices_all = self._per_cell_geometry(vor, vertices_all, ridge_vertices_all, num_vertices, vertexpair2ridge)

        from matplotlib import pyplot as plt

        if ax is None:
            ax = plt.gca()
        
        self._plot_routine(ax, vor, vertices_all, ridge_vertices_all,
                   geom["point_edges_type"], geom["point_vertices_f_idx"])
        
        if show:
            plt.show()
        return ax

    # --------------------- Paradigm of plotting ---------------------
    def _plot_routine(self, ax: matplotlib.axes.Axes, vor: Voronoi, vertices_all: np.ndarray, ridge_vertices_all: list[list[int]],
                      point_edges_type: list[list[int]], point_vertices_f_idx: list[list[int]]) -> None:
        """
        Low-level plot routine. Draws:
          - All Voronoi edges (solid for finite, dashed for formerly-infinite)
          - Cell centers
          - Each cell boundary (poly edges and circular arcs)
        """ 
        pts = self.pts
        r = self.phys.r
        N = self.N

        center = np.mean(pts, axis=0)
        if N > 1:
            span_x = np.ptp(pts[:, 0])  # pts span in x
            span_y = np.ptp(pts[:, 1])  # pts span in y
            L = max(span_x, span_y) + 3.0 * r
            L *= 0.8
        else:
            L = 5.0 * r

        # Draw Voronoi ridge segments
        for idx in range(len(vor.ridge_vertices)):
            x1, y1 = vertices_all[ridge_vertices_all[idx][0]]
            x2, y2 = vertices_all[ridge_vertices_all[idx][1]]
            if -1 not in vor.ridge_vertices[idx]:
                ax.plot([x1, x2], [y1, y2], 'k-', lw=0.5)
            else:
                ax.plot([x1, x2], [y1, y2], 'k--', lw=0.5)

        # Draw cell centers
        ax.plot(pts[:, 0], pts[:, 1], 'o', color='C0', markersize=2)

        # Draw each cell boundary
        for idx in range(N):
            edges_type = point_edges_type[idx]
            vertices_f_idx = point_vertices_f_idx[idx]

            x, y = pts[idx]
            if len(edges_type) < 2:
                angle = np.linspace(0, 2*np.pi, 100)
                ax.plot(x + r * np.cos(angle), y + r * np.sin(angle), color="C6", zorder=2)
                continue

            for idx_f, edge_type in enumerate(edges_type):
                v1_idx = vertices_f_idx[idx_f]
                x1, y1 = vertices_all[v1_idx]
                idx2 = idx_f + 1 if idx_f < len(edges_type)-1 else 0
                v2_idx = vertices_f_idx[idx2]
                x2, y2 = vertices_all[v2_idx]

                if edge_type == 1:
                    ax.plot([x1, x2], [y1, y2], 'b-', zorder=1)
                else:
                    angle1 = np.arctan2(y1-y, x1-x)
                    angle2 = np.arctan2(y2-y, x2-x)
                    dangle = np.linspace(0, (angle1 - angle2) % (2*np.pi), 100)

                    ax.plot(x + r * np.cos(angle2+dangle), y + r * np.sin(angle2+dangle), color="C6", zorder=2)

        ax.set_aspect("equal")
        ax.set_xlim(center[0]-L, center[0]+L)
        ax.set_ylim(center[1]-L, center[1]+L)

    # --------------------- Connections between cells ---------------------
    def _get_connections(self, ridge_points: list[list[int]], vertices_all: np.ndarray, ridge_vertices_all: list[list[int]]) -> np.ndarray:
        """
        Determine which pairs of cells are connected, i.e.,
        the distance from the cell center to its corresponding Voronoi ridge
        segment is < self.phys.r.
        """
        ridge_points_arr = np.asarray(ridge_points, dtype=int).reshape(-1, 2)        # (R, 2)
        ridge_vertices_arr = np.asarray(ridge_vertices_all, dtype=int).reshape(-1, 2)  # (R, 2)

        # take p2 for each ridge, avoid -1 points (representing space)
        p1_idx = ridge_points_arr[:, 0]        # (R,)
        p2_idx = ridge_points_arr[:, 1]        # (R,)
        p2 = self.pts[p2_idx]                  # (R, 2)

        v1 = vertices_all[ridge_vertices_arr[:, 0]]   # (R, 2)
        v2 = vertices_all[ridge_vertices_arr[:, 1]]   # (R, 2)

        # vectorized point-to-segment distance
        AB = v2 - v1                      # (R, 2)
        AP = p2 - v1                      # (R, 2)
        denom = np.einsum("ij,ij->i", AB, AB)  # (R,)

        t = np.einsum("ij,ij->i", AP, AB) / denom
        t = np.clip(t, 0.0, 1.0)[:, None]  # (R,1)

        C = v1 + t * AB                   # closest point on segment, (R,2)
        dists = np.linalg.norm(p2 - C, axis=1)  # (R,)

        mask = dists < self.phys.r

        connect = np.stack([p1_idx[mask], p2_idx[mask]], axis=1)
        if connect.size > 0:
            connect = np.sort(connect, axis=1)
        else:
            connect = np.empty((0, 2), dtype=int)
        return connect

    # --------------------- Update positions ---------------------
    def update_positions(self, pts: np.ndarray, A0: float | np.ndarray | None = None) -> None:
        """
        Update cell center positions.

        .. note::
            If the number of cells changes, the preferred areas for all cells
            are reset to the default value---defined either at simulator instantiation 
            or by :py:meth:`update_params`---unless *A0* is explicitly specified.

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
        self.pts = pts

        if N != self.N:
            self.N = N
            if A0 is None:
                self._preferred_areas = np.full(N, self.phys.A0, dtype=float)
            else:
                self.update_preferred_areas(A0)
        else:
            if A0 is not None:
                self.update_preferred_areas(A0)

    # --------------------- Update physical parameters ---------------------
    def update_params(self, phys: PhysicalParams) -> None:
        """
        Update physical parameters.

        Args:
            phys: New :py:class:`PhysicalParams` object.
        
        Raises:
            TypeError: If *phys* is not an instance of :py:class:`PhysicalParams`.

        .. warning::
            This also resets all preferred cell areas to the new value of *A0*.
        """
        if not isinstance(phys, PhysicalParams):
            raise TypeError("phys must be an instance of PhysicalParams")
        
        self.phys = phys
        self.update_preferred_areas(phys.A0)

    # --------------------- Update preferred area list ---------------------
    def update_preferred_areas(self, A0: float | np.ndarray) -> None:
        """
        Update the preferred areas for all cells.

        Args:
            A0: New preferred area(s) for all cells, either as a scalar or
                as an array of shape (N,).

        Raises:
            ValueError: If *A0* does not match cell number.
        """
        arr = np.asarray(A0, dtype=float)

        # Accept scalar (0-d) or length-1 array as "uniform"
        if arr.ndim == 0:
            arr = np.full(self.N, float(arr), dtype=float)
        elif arr.shape == (1,):
            arr = np.full(self.N, float(arr[0]), dtype=float)
        else:
            if arr.shape != (self.N,):
                raise ValueError(f"A0 must be scalar or have shape ({self.N},)")
        
        self._preferred_areas = arr

    @property
    def preferred_areas(self) -> np.ndarray:
        r"""
        Preferred areas :math:`\{A_{0,i}\}` for all cells (read-only).

        Returns:
            numpy.ndarray: A copy of the internal preferred area array.
        """
        return self._preferred_areas.copy()

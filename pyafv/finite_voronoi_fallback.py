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

from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy.spatial import Voronoi
from collections import defaultdict
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from .physical_params import PhysicalParams


# ---- tiny helpers to avoid tiny allocations in hot loops ----
def _row_dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise dot product for 2D arrays with shape (N,2)."""
    return np.einsum("ij,ij->i", a, b)


class FiniteVoronoiSimulator:
    def __init__(self, pts: np.ndarray, phys: PhysicalParams):
        self.pts = pts.copy()            # (N,2) array of initial points
        self.N = pts.shape[0]            # Number of points
        self.phys = phys

    # --------------------- Voronoi construction & extension ---------------------
    def _build_voronoi_with_extensions(self) -> Tuple[Voronoi, np.ndarray, List[List[int]], int, Dict[Tuple[int,int], int], Dict[int, List[int]]]:
        """
        Build SciPy Voronoi for current points. For N<=2, emulate regions.
        For N>=3, extend infinite ridges by adding long rays and update
        regions accordingly. Return augmented structures.
        """
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
            vor.point_region = [0]
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
                vor.ridge_vertices = [[0, 1], [1, 2], [0, 2], [0, 3], [1, 3]]
                vor.regions = [[0, 1, 2], [0, 1, 3]]
                vor.ridge_points = np.array([[0, 1], [-1, 0], [-1, 0], [-1, 1], [-1, 1]])

            vor.point_region = [0, 1]

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


        # Build ridge incidence per vertex, and a lookup for (v1,v2) -> ridge id
        vertex_incident_ridges = defaultdict(list)
        # (2) cheaper key: store both directions as tuple -> int
        vertexpair2ridge: Dict[Tuple[int,int], int] = {}

        rv_full = np.asarray(ridge_vertices_all, dtype=int)
        R = rv_full.shape[0]
        for k in range(R):
            v1, v2 = int(rv_full[k, 0]), int(rv_full[k, 1])
            vertex_incident_ridges[v1].append(k)
            vertex_incident_ridges[v2].append(k)
            vertexpair2ridge[(v1, v2)] = k
            vertexpair2ridge[(v2, v1)] = k

        # For each finite vertex, record which input points (cells) meet there
        vertex_points = {}
        if N > 2:
            for v_id in range(num_vertices):
                s = set()
                for ridge_id in vertex_incident_ridges[v_id]:
                    i, j = vor.ridge_points[ridge_id]
                    s.add(i), s.add(j)
                vertex_points[v_id] = list(s)

        return vor, vertices_all, ridge_vertices_all, num_vertices, vertexpair2ridge, vertex_points

    # --------------------- Geometry & energy contributions per cell ---------------------
    def _per_cell_geometry(self, vor: Voronoi, vertices_all: np.ndarray, ridge_vertices_all: np.ndarray, num_vertices: int, vertexpair2ridge: Dict[Tuple[int, int], int]) -> Dict:
        """
        Iterate cells to:
          - sort polygon/arc vertices around each cell
          - classify edges (1 = straight Voronoi edge; 0 = circular arc)
          - compute area/perimeter for each cell
          - accumulate derivatives w.r.t. vertices (dA_poly/dh, dP_poly/dh)
          - register 'outer' vertices created at arc intersections and track their point pairs
        """
        N = self.N
        r = self.phys.r
        A0 = self.phys.A0
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



        point_edges_type = []
        point_vertices_f_idx = []

        # --- fast vectorized per-cell processing (no inner edge loop) ---
        for idx in range(N):
            region_id = vor.point_region[idx]
            v_ids = np.asarray(vor.regions[region_id], dtype=int)

            if v_ids.size == 0:
                point_edges_type.append([])
                point_vertices_f_idx.append([])
                continue

            # sort vertices clockwise around cell center
            rel = vertices_all[v_ids] - pts[idx]
            angles = np.arctan2(rel[:, 1], rel[:, 0])
            order = np.argsort(angles)[::-1]
            v_ids = v_ids[order]

            # consecutive pairs (wrap) -> candidate edges around this cell
            v1_ids = v_ids
            v2_ids = np.roll(v_ids, -1)

            # skip ray-ray (both >= num_vertices)
            valid = ~((v1_ids >= num_vertices) & (v2_ids >= num_vertices))
            if not np.any(valid):
                point_edges_type.append([])
                point_vertices_f_idx.append([])
                continue

            v1_ids = v1_ids[valid]
            v2_ids = v2_ids[valid]

            # ---- vectorized ridge id lookup for all edges of this cell ----
            # use the dict you already built with both orientations:
            #   vertexpair2ridge[(v1, v2)] = ridge_id
            # convert all edge pairs in one go via list comprehension (still fast, no Python loop per-edge body)
            # NB: we keep it simple & reliable; if needed, switch to a sorted-structured-array map later.
            keys = list(zip(v1_ids.tolist(), v2_ids.tolist()))
            ridge_ids = np.fromiter((vertexpair2ridge[k] for k in keys), dtype=int, count=len(keys))

            # decide which endpoint pack to use (p1 vs p2) for each edge
            use_p1 = (p1[ridge_ids] == idx)
            use_p2 = ~use_p1

            # gather packs (shape (E,3)), then mask out the -1 slots
            pack_e = np.empty((len(ridge_ids), 3), dtype=int)
            pack_v = np.empty((len(ridge_ids), 3), dtype=int)

            if np.any(use_p1):
                pack_e[use_p1] = p1_edges_pack[ridge_ids[use_p1]]
                pack_v[use_p1] = p1_verts_pack[ridge_ids[use_p1]]
            if np.any(use_p2):
                pack_e[use_p2] = p2_edges_pack[ridge_ids[use_p2]]
                pack_v[use_p2] = p2_verts_pack[ridge_ids[use_p2]]

            # flatten valid entries in pack order (keeps your exact edge ordering)
            mask = (pack_e >= 0)
            edges_type = pack_e[mask].tolist()
            vertices_f_idx = pack_v[mask].tolist()

            if len(vertices_f_idx) != len(edges_type):
                raise ValueError("Vertex and edge number not equal!")

            point_edges_type.append(edges_type)
            point_vertices_f_idx.append(vertices_f_idx)




        # --- helpers ---
        def _row_cross(a, b):
            # z-component of 2D cross, row-wise
            return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]

        def _perp(u):
            # rotate 90° CW: (ux,uy) -> (uy,-ux)
            return np.column_stack((u[:, 1], -u[:, 0]))

        vertex_out_da_dtheta = np.zeros((2 * num_ridges, 2))
        vertex_out_dl_dtheta = np.zeros((2 * num_ridges, 2))

        dA_poly_dh = np.zeros((num_vertices_ext + 2 * num_ridges, 2))
        dP_poly_dh = np.zeros((num_vertices_ext + 2 * num_ridges, 2))

        area_list = np.zeros(N)
        perimeter_list = np.zeros(N)

        for idx in range(N):
            edges_type = np.asarray(point_edges_type[idx], dtype=int)
            vertices_f_idx = np.asarray(point_vertices_f_idx[idx], dtype=int)
            E = edges_type.size

            if E < 2:
                area_list[idx] = np.pi * (r ** 2)
                perimeter_list[idx] = 2.0 * np.pi * r
                continue

            # ring indices
            v1_idx = vertices_f_idx
            v2_idx = np.roll(vertices_f_idx, -1)
            v0_idx = np.roll(vertices_f_idx,  1)

            ri = pts[idx]
            V1 = vertices_all[v1_idx]
            V2 = vertices_all[v2_idx]
            V0 = vertices_all[v0_idx]
            V1mR = V1 - ri
            V2mR = V2 - ri
            V0mR = V0 - ri

            mask_str = (edges_type == 1)
            mask_arc = ~mask_str

            # ----- perimeter & area -----
            seg12 = V1 - V2
            l12 = np.linalg.norm(seg12, axis=1)
            Pi_straight = l12[mask_str].sum()
            Ai_straight = (-0.5 * _row_cross(V1mR[mask_str], V2mR[mask_str])).sum()

            if np.any(mask_arc):
                a1_full = np.arctan2(V1mR[:, 1], V1mR[:, 0])
                a2_full = np.arctan2(V2mR[:, 1], V2mR[:, 0])
                dangle_full = (a1_full - a2_full) % (2.0 * np.pi)
                dangle_arc = dangle_full[mask_arc]
                Pi_arc = (r * dangle_arc).sum()
                Ai_arc = (0.5 * (r ** 2) * dangle_arc).sum()
            else:
                Pi_arc = 0.0
                Ai_arc = 0.0

            Pi = Pi_straight + Pi_arc
            Ai = Ai_straight + Ai_arc
            perimeter_list[idx] = Pi
            area_list[idx] = Ai

            # ----- dA_poly/dh, dP_poly/dh for v1 -----
            dAi_v1 = -0.5 * _perp(V2mR) + 0.5 * _perp(V0mR)       # (E,2)

            dPi_v1 = np.zeros((E, 2))
            if np.any(mask_str):
                dPi_v1[mask_str] += seg12[mask_str] / l12[mask_str][:, None]

            mask_prev_str = np.roll(mask_str, 1)
            seg10 = V1 - V0
            l10 = np.linalg.norm(seg10, axis=1)
            if np.any(mask_prev_str):
                dPi_v1[mask_prev_str] += seg10[mask_prev_str] / l10[mask_prev_str][:, None]

            np.add.at(dA_poly_dh, v1_idx, (Ai - A0) * dAi_v1)
            np.add.at(dP_poly_dh, v1_idx, (Pi - P0) * dPi_v1)

            # ----- arc endpoint sensitivities at outer vertices -----
            if np.any(mask_arc):
                # endpoint rows in vertex_out_* are (outer_id - num_vertices_ext)
                v1_arc_idx = v1_idx[mask_arc]
                v2_arc_idx = v2_idx[mask_arc]
                k1 = v1_arc_idx - num_vertices_ext
                k2 = v2_arc_idx - num_vertices_ext
                valid1 = (k1 >= 0)
                valid2 = (k2 >= 0)

                if np.any(valid1) or np.any(valid2):
                    # da/dtheta for endpoints (sector - triangle)
                    da1_full = 0.5 * (r ** 2) * (1.0 - np.cos(dangle_full))   # v1 endpoint
                    da2_full = -da1_full                                      # v2 endpoint
                    da1_arc = da1_full[mask_arc]
                    da2_arc = da2_full[mask_arc]

                    # dl/dtheta is ±r
                    dl1 = r
                    dl2 = -r

                    vop = vertex_out_points  # rows are sorted [i,j]; column 1 is max(i,j)
                    if np.any(valid1):
                        k1v = k1[valid1]
                        # CORRECT which_point: 0 if max(i,j) > idx else 1
                        which1 = (vop[k1v, 1] <= idx).astype(int)
                        vertex_out_da_dtheta[k1v, which1] = da1_arc[valid1]
                        vertex_out_dl_dtheta[k1v, which1] = dl1

                    if np.any(valid2):
                        k2v = k2[valid2]
                        which2 = (vop[k2v, 1] <= idx).astype(int)
                        vertex_out_da_dtheta[k2v, which2] = da2_arc[valid2]
                        vertex_out_dl_dtheta[k2v, which2] = dl2


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
                         vertex_points: Dict[int, List[int]], vertex_in_id: List[int], vertex_out_id: List[int],
                         vertex_out_points: List[List[int]], vertex_out_da_dtheta: np.ndarray,
                         vertex_out_dl_dtheta: np.ndarray, dA_poly_dh: np.ndarray, dP_poly_dh: np.ndarray,
                         area_list: np.ndarray, perimeter_list: np.ndarray) -> np.ndarray:
        """
        Assemble forces on cell centers from polygon and arc contributions.
        """
        N = self.N
        r = self.phys.r
        A0 = self.phys.A0
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

                Ai_w_i = (area_list[I]      - A0) * v_da[:, 0]
                Aj_w_j = (area_list[J]      - A0) * v_da[:, 1]
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
    def build(self) -> Dict:
        """
        Do the following:
          - Build Voronoi (+ extensions)
          - Get cell connectivity
          - Compute per-cell quantities and derivatives
          - Assemble forces
        Returns a dictionary of diagnostics.
        """
        (vor, vertices_all, ridge_vertices_all, num_vertices,
            vertexpair2ridge, vertex_points) = self._build_voronoi_with_extensions()
        
        connections = self._get_connections(vor.ridge_points, vertices_all, ridge_vertices_all)

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
    def plot_2d(self, ax: Optional[Axes] = None, show: bool = False) -> Axes:
        """
        Build the Voronoi(+extensions) and render a 2D snapshot.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or None
            If provided, draw into this axes; otherwise get the current axes.
        show : bool
            Whether to call plt.show() at the end.

        Returns
        -------
        ax : matplotlib.axes.Axes
        """
        (vor, vertices_all, ridge_vertices_all, num_vertices,
            vertexpair2ridge, vertex_points) = self._build_voronoi_with_extensions()

        geom, vertices_all = self._per_cell_geometry(vor, vertices_all, ridge_vertices_all, num_vertices, vertexpair2ridge)

        if ax is None:
            ax = plt.gca()
        
        self._plot_routine(ax, vor, vertices_all, ridge_vertices_all,
                   geom["point_edges_type"], geom["point_vertices_f_idx"])
        
        if show:
            plt.show()
        return ax


    # --------------------- Paradigm of plotting ---------------------
    def _plot_routine(self, ax: Axes, vor: Voronoi, vertices_all: np.ndarray, ridge_vertices_all: List[List[int]],
                      point_edges_type: List[List[int]], point_vertices_f_idx: List[List[int]]) -> None:
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
    def _get_connections(self, ridge_points: List[List[int]], vertices_all: np.ndarray, ridge_vertices_all: List[List[int]]) -> np.ndarray:
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
    def update_positions(self, pts: np.ndarray) -> None:
        """
        Update cell center positions.
        """
        self.N, dim = pts.shape
        if dim != 2:
            raise ValueError("Positions must have shape (N,2)")
        
        self.pts = pts


    # --------------------- Update physical parameters ---------------------
    def update_params(self, phys: PhysicalParams) -> None:
        """
        Update physical parameters.
        """
        self.phys = phys

"""
These are pure-Python fallback implementations of key functions for finite Voronoi
cell geometry and derivative calculations, used when Cython extensions are not
available.
"""

import numpy as np
from collections import defaultdict


def build_vertexpair_and_vertexpoints(ridge_vertices_all, ridge_points, num_vertices, N):
    # Build ridge incidence per vertex, and a lookup for (v1,v2) -> ridge id
    vertex_incident_ridges = defaultdict(list)
    # Cheaper key: store both directions as tuple -> int
    vertexpair2ridge: dict[tuple[int, int], int] = {}

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
                i, j = ridge_points[ridge_id]
                s.add(i), s.add(j)
            vertex_points[v_id] = list(s)

    return vertexpair2ridge, vertex_points


def pad_regions(regions):
    # fake function to match cython backend interface
    return regions


def build_point_edges(vor_regions, point_region,
                      vertices_all, pts,
                      num_vertices, vertexpair2ridge,
                      p1, p1_edges_pack, p1_verts_pack,
                      p2, p2_edges_pack, p2_verts_pack):

    N = pts.shape[0]

    point_edges_type = []
    point_vertices_f_idx = []

    # --- fast vectorized per-cell processing (no inner edge loop) ---
    for idx in range(N):
        region_id = point_region[idx]
        v_ids = np.asarray(vor_regions[region_id], dtype=int)

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
        if not np.any(valid):                # pragma: no cover
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
        ridge_ids = np.fromiter(
            (vertexpair2ridge[k] for k in keys), dtype=int, count=len(keys))

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

    return point_edges_type, point_vertices_f_idx


def compute_vertex_derivatives(
            point_edges_type,            # list-of-lists / arrays of edge types
            point_vertices_f_idx,        # list-of-lists / arrays of vertex ids
            vertices_all,
            pts, r, A0_list, P0, num_vertices_ext, num_ridges,
            vertex_out_points):

    N = pts.shape[0]

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

        np.add.at(dA_poly_dh, v1_idx, (Ai - A0_list[idx]) * dAi_v1)
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

    return vertex_out_da_dtheta, vertex_out_dl_dtheta, dA_poly_dh, dP_poly_dh, area_list, perimeter_list

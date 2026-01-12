# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False
# cython: freethreading_compatible = True
# distutils: language = c++

import numpy as np
cimport numpy as cnp
from libc.math cimport atan2, cos
cimport cython

# Routine used in _build_voronoi_with_extensions()
def build_vertexpair_and_vertexpoints(object ridge_vertices_all,
                                      object ridge_points,
                                      long num_vertices,
                                      long N):
    # --- accept any int dtype and make sure it's contiguous int64 ---
    cdef cnp.ndarray[cnp.int64_t, ndim=2] rv = np.ascontiguousarray(ridge_vertices_all, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=2] rp = np.ascontiguousarray(ridge_points,      dtype=np.int64)
    # ----------------------------------------------------------------

    cdef Py_ssize_t R = rv.shape[0]
    cdef Py_ssize_t k, v1, v2, v_id, ridge_id
    cdef Py_ssize_t i_pt, j_pt

    cdef dict vertex_incident_ridges = {}
    cdef dict vertexpair2ridge = {}
    cdef dict vertex_points = {}
    cdef list ridges_for_v
    cdef set s

    for k in range(R):
        v1 = <Py_ssize_t> rv[k, 0]
        v2 = <Py_ssize_t> rv[k, 1]

        if v1 not in vertex_incident_ridges:
            vertex_incident_ridges[v1] = []
        if v2 not in vertex_incident_ridges:
            vertex_incident_ridges[v2] = []

        (<list>vertex_incident_ridges[v1]).append(k)
        (<list>vertex_incident_ridges[v2]).append(k)

        vertexpair2ridge[(<int>v1, <int>v2)] = <int>k
        vertexpair2ridge[(<int>v2, <int>v1)] = <int>k

    if N > 2:
        for v_id in range(num_vertices):
            if v_id not in vertex_incident_ridges:
                continue
            ridges_for_v = <list>vertex_incident_ridges[v_id]
            s = set()
            for ridge_id in ridges_for_v:
                i_pt = <int> rp[ridge_id, 0]
                j_pt = <int> rp[ridge_id, 1]
                s.add(i_pt); s.add(j_pt)
            vertex_points[v_id] = list(s)

    return vertexpair2ridge, vertex_points


# ---------------------------------------------------------------------------------------
# Part 1
# ---------------------------------------------------------------------------------------

def pad_regions(object regions):
    cdef Py_ssize_t n = len(regions)
    cdef Py_ssize_t i, j, m, kmax = 0
    cdef list r

    for i in range(n):
        m = len(<list>regions[i])
        if m > kmax:
            kmax = m

    cdef cnp.ndarray[cnp.int64_t, ndim=2] out = \
        np.empty((n, kmax), dtype=np.int64)

    # fill with -1
    for i in range(n):
        for j in range(kmax):
            out[i, j] = -1

    # copy rows
    for i in range(n):
        r = <list>regions[i]
        m = len(r)
        for j in range(m):
            out[i, j] = <long long>r[j]

    return out

@cython.cfunc
cdef inline long long pack_pair_ll(long a, long b) nogil:
    return ((<long long>a) << 32) | (<unsigned long long>b)

@cython.boundscheck(False)
@cython.wraparound(False)
def build_point_edges(
    cnp.int64_t[:, :] vor_regions,
    cnp.int64_t[:] point_region,
    cnp.float64_t[:, :] vertices_all,
    cnp.float64_t[:, :] pts,
    long num_vertices,
    dict vertexpair2ridge_py,
    cnp.int64_t[:] p1,
    cnp.int64_t[:, :] p1_edges_pack,
    cnp.int64_t[:, :] p1_verts_pack,
    cnp.int64_t[:] p2,
    cnp.int64_t[:, :] p2_edges_pack,
    cnp.int64_t[:, :] p2_verts_pack
):
    cdef Py_ssize_t N = pts.shape[0]
    cdef list point_edges_type = [None] * N
    cdef list point_vertices_f_idx = [None] * N

    # packed (v1,v2) -> ridge_id
    cdef dict pair2ridge = {}
    cdef object k, rid_val_py
    for k, rid_val_py in vertexpair2ridge_py.items():
        pair2ridge[ ((<long long>(<tuple>k)[0])<<32) | (<unsigned long long>(<tuple>k)[1]) ] = <int>rid_val_py

    # declarations
    cdef Py_ssize_t idx, j, m, ecount, n_valid
    cdef int rid
    cdef long region_id, v1, v2
    cdef cnp.int64_t[:] region_row
    cdef cnp.float64_t cx, cy, vx, vy
    cdef int Rmax = vor_regions.shape[1]
    cdef bint use_p1_flag
    cdef long val
    cdef list edges_type
    cdef list vertices_f_idx

    # scratch arrays as Python objects + memoryviews
    cdef object angles_np = np.empty(Rmax, dtype=np.float64)
    cdef cnp.float64_t[:] angles = angles_np
    cdef object valid_np  = np.empty(Rmax, dtype=np.uint8)
    cdef cnp.uint8_t[:] valid = valid_np

    cdef object v_ids_np
    cdef object v2_ids_np
    cdef cnp.int64_t[:] v_ids
    cdef cnp.int64_t[:] v2_ids

    cdef cnp.int64_t[:, :] pack_e
    cdef cnp.int64_t[:, :] pack_v

    for idx in range(N):
        region_id = <long>point_region[idx]
        if region_id < 0:
            point_edges_type[idx] = []
            point_vertices_f_idx[idx] = []
            continue

        # 1) collect region vertex ids (stop at -1)
        region_row = vor_regions[region_id]
        m = 0
        while m < Rmax and region_row[m] != -1:
            m += 1
        if m == 0:
            point_edges_type[idx] = []
            point_vertices_f_idx[idx] = []
            continue

        v_ids_np = np.empty(m, dtype=np.int64)
        v_ids = v_ids_np
        for j in range(m):
            v_ids[j] = region_row[j]

        # 2) angles around cell center
        cx = pts[idx, 0]; cy = pts[idx, 1]
        for j in range(m):
            vx = vertices_all[v_ids[j], 0] - cx
            vy = vertices_all[v_ids[j], 1] - cy
            angles[j] = atan2(vy, vx)

        # 3) sort clockwise (descending angle)
        order = np.argsort(np.asarray(angles_np)[:m])  # Python object
        v_ids_np = np.asarray(v_ids_np)[order[::-1]]   # new array
        v_ids = v_ids_np                                # rebind memoryview to the new array

        # 4) consecutive pairs with wrap
        v2_ids_np = np.empty(m, dtype=np.int64)
        v2_ids = v2_ids_np
        for j in range(m-1):
            v2_ids[j] = v_ids[j+1]
        v2_ids[m-1] = v_ids[0]

        # 5) mask: skip ray-ray
        n_valid = 0
        for j in range(m):
            valid[j] = not ((v_ids[j] >= num_vertices) and (v2_ids[j] >= num_vertices))
            if valid[j]:
                n_valid += 1
        if n_valid == 0:
            point_edges_type[idx] = []
            point_vertices_f_idx[idx] = []
            continue

        # 6) gather packs
        pack_e = np.empty((n_valid, 3), dtype=np.int64)
        pack_v = np.empty((n_valid, 3), dtype=np.int64)
        ecount = 0

        for j in range(m):
            if not valid[j]:
                continue
            v1 = <long>v_ids[j]
            v2 = <long>v2_ids[j]

            rid = <int>pair2ridge.get(((<long long>v1)<<32) | (<unsigned long long>v2), -1)
            if rid < 0:
                rid = <int>vertexpair2ridge_py.get((v1, v2), -1)
                if rid < 0:
                    raise KeyError(f"Missing ridge id for pair ({v1}, {v2})")

            use_p1_flag = (p1[rid] == idx)
            if use_p1_flag:
                pack_e[ecount, 0] = p1_edges_pack[rid, 0]
                pack_e[ecount, 1] = p1_edges_pack[rid, 1]
                pack_e[ecount, 2] = p1_edges_pack[rid, 2]
                pack_v[ecount, 0] = p1_verts_pack[rid, 0]
                pack_v[ecount, 1] = p1_verts_pack[rid, 1]
                pack_v[ecount, 2] = p1_verts_pack[rid, 2]
            else:
                pack_e[ecount, 0] = p2_edges_pack[rid, 0]
                pack_e[ecount, 1] = p2_edges_pack[rid, 1]
                pack_e[ecount, 2] = p2_edges_pack[rid, 2]
                pack_v[ecount, 0] = p2_verts_pack[rid, 0]
                pack_v[ecount, 1] = p2_verts_pack[rid, 1]
                pack_v[ecount, 2] = p2_verts_pack[rid, 2]
            ecount += 1

        # 7) flatten valid entries
        edges_type = []
        vertices_f_idx = []
        for j in range(n_valid):
            val = pack_e[j, 0]
            if val >= 0:
                edges_type.append(<int>val)
                vertices_f_idx.append(<int>pack_v[j, 0])
            val = pack_e[j, 1]
            if val >= 0:
                edges_type.append(<int>val)
                vertices_f_idx.append(<int>pack_v[j, 1])
            val = pack_e[j, 2]
            if val >= 0:
                edges_type.append(<int>val)
                vertices_f_idx.append(<int>pack_v[j, 2])

        if len(edges_type) != len(vertices_f_idx):
            raise ValueError("Vertex and edge number not equal!")

        point_edges_type[idx] = edges_type
        point_vertices_f_idx[idx] = vertices_f_idx

    return point_edges_type, point_vertices_f_idx


# ---------------------------------------------------------------------------------------
# Part 2
# ---------------------------------------------------------------------------------------

@cython.cfunc
cdef inline double cross2(double ax, double ay, double bx, double by) nogil:
    return ax*by - ay*bx

@cython.cfunc
cdef inline void perp(double ux, double uy, double* outx, double* outy) noexcept nogil:
    outx[0] = uy
    outy[0] = -ux

@cython.boundscheck(False)
@cython.wraparound(False)
def compute_vertex_derivatives(
    object point_edges_type,               # list of 1D int arrays/lists
    object point_vertices_f_idx,           # list of 1D int arrays/lists
    cnp.float64_t[:, :] vertices_all,      # (num_vertices_ext + 2*num_ridges, 2)
    cnp.float64_t[:, :] pts,               # (N, 2)
    double r,
    cnp.float64_t[:] A0_list,              # (N,)
    double P0,
    long num_vertices_ext,
    long num_ridges,
    cnp.int64_t[:, :] vertex_out_points    # (2*num_ridges, 2), rows sorted [i,j], col1=max(i,j)
):
    cdef Py_ssize_t N = pts.shape[0]
    cdef Py_ssize_t i, j, E
    cdef double two_pi = 6.2831853071795864769

    # outputs
    cdef object vertex_out_da_dtheta_np = np.zeros((2*num_ridges, 2), dtype=np.float64)
    cdef object vertex_out_dl_dtheta_np = np.zeros((2*num_ridges, 2), dtype=np.float64)
    cdef object dA_poly_dh_np = np.zeros((num_vertices_ext + 2*num_ridges, 2), dtype=np.float64)
    cdef object dP_poly_dh_np = np.zeros((num_vertices_ext + 2*num_ridges, 2), dtype=np.float64)
    cdef object area_list_np = np.zeros(N, dtype=np.float64)
    cdef object perimeter_list_np = np.zeros(N, dtype=np.float64)

    cdef cnp.float64_t[:, :] vertex_out_da_dtheta = vertex_out_da_dtheta_np
    cdef cnp.float64_t[:, :] vertex_out_dl_dtheta = vertex_out_dl_dtheta_np
    cdef cnp.float64_t[:, :] dA_poly_dh = dA_poly_dh_np
    cdef cnp.float64_t[:, :] dP_poly_dh = dP_poly_dh_np
    cdef cnp.float64_t[:] area_list = area_list_np
    cdef cnp.float64_t[:] perimeter_list = perimeter_list_np

    # declarations used inside loop (declare here, assign in loop)
    cdef object edges_type_obj
    cdef object v_idx_obj
    cdef cnp.int64_t[:] edges_type
    cdef cnp.int64_t[:] vertices_f_idx

    # scalars
    cdef double Pi_straight, Ai_straight, Pi_arc, Ai_arc, Pi, Ai
    cdef double cx, cy
    cdef double v1x, v1y, v2x, v2y, v0x, v0y
    cdef double v1mx, v1my, v2mx, v2my, v0mx, v0my
    cdef double s12x, s12y, s10x, s10y
    cdef double l12, l10
    cdef double a1, a2, dangle, da1_full, da2_full
    cdef double dAi_v1_x, dAi_v1_y
    cdef double dPi_v1_x, dPi_v1_y
    cdef long vidx, vprev
    cdef long outer_row, which_col
    cdef long k1, k2

    # moved here (was causing your error inside the loop)
    cdef double p2x, p2y, p0x, p0y

    # small boolean arrays per cell
    cdef object mask_str_np
    cdef cnp.uint8_t[:] mask_str
    cdef object mask_prev_str_np
    cdef cnp.uint8_t[:] mask_prev_str
    cdef object mask_arc_np
    cdef cnp.uint8_t[:] mask_arc

    # arrays per cell
    cdef object v1_idx_np
    cdef object v2_idx_np
    cdef object v0_idx_np
    cdef cnp.int64_t[:] v1_idx
    cdef cnp.int64_t[:] v2_idx
    cdef cnp.int64_t[:] v0_idx

    cdef object a1_full_np
    cdef object a2_full_np
    cdef cnp.float64_t[:] a1_full
    cdef cnp.float64_t[:] a2_full

    cdef object dangle_full_np
    cdef cnp.float64_t[:] dangle_full

    for i in range(N):
        edges_type_obj = point_edges_type[i]
        v_idx_obj = point_vertices_f_idx[i]

        # ensure ndarray[int64]
        if not isinstance(edges_type_obj, np.ndarray) or (<cnp.ndarray>edges_type_obj).dtype.num != cnp.NPY_INT64:
            edges_type_obj = np.asarray(edges_type_obj, dtype=np.int64)
        if not isinstance(v_idx_obj, np.ndarray) or (<cnp.ndarray>v_idx_obj).dtype.num != cnp.NPY_INT64:
            v_idx_obj = np.asarray(v_idx_obj, dtype=np.int64)

        # assign to memoryviews (no cdef here)
        edges_type = edges_type_obj
        vertices_f_idx = v_idx_obj
        E = vertices_f_idx.shape[0]

        if E < 2:
            area_list[i] = 3.14159265358979323846 * (r * r)
            perimeter_list[i] = 2.0 * 3.14159265358979323846 * r
            continue

        # ring indices
        v1_idx_np = np.empty(E, dtype=np.int64)
        v2_idx_np = np.empty(E, dtype=np.int64)
        v0_idx_np = np.empty(E, dtype=np.int64)
        v1_idx = v1_idx_np; v2_idx = v2_idx_np; v0_idx = v0_idx_np

        for j in range(E):
            v1_idx[j] = vertices_f_idx[j]
            v2_idx[j] = vertices_f_idx[(j+1) % E]
            v0_idx[j] = vertices_f_idx[(j-1+E) % E]

        cx = pts[i, 0]; cy = pts[i, 1]

        # masks
        mask_str_np = np.empty(E, dtype=np.uint8)
        mask_arc_np = np.empty(E, dtype=np.uint8)
        mask_prev_str_np = np.empty(E, dtype=np.uint8)
        mask_str = mask_str_np; mask_arc = mask_arc_np; mask_prev_str = mask_prev_str_np

        for j in range(E):
            mask_str[j] = 1 if edges_type[j] == 1 else 0
            mask_arc[j] = 0 if mask_str[j] else 1

        # ----- perimeter & area -----
        Pi_straight = 0.0
        Ai_straight = 0.0

        for j in range(E):
            if mask_str[j]:
                v1x = vertices_all[v1_idx[j], 0]; v1y = vertices_all[v1_idx[j], 1]
                v2x = vertices_all[v2_idx[j], 0]; v2y = vertices_all[v2_idx[j], 1]
                v1mx = v1x - cx; v1my = v1y - cy
                v2mx = v2x - cx; v2my = v2y - cy
                s12x = v1x - v2x; s12y = v1y - v2y
                l12 = (s12x*s12x + s12y*s12y) ** 0.5
                Pi_straight += l12
                Ai_straight += -0.5 * cross2(v1mx, v1my, v2mx, v2my)

        Pi_arc = 0.0
        Ai_arc = 0.0

        a1_full_np = np.empty(E, dtype=np.float64)
        a2_full_np = np.empty(E, dtype=np.float64)
        dangle_full_np = np.empty(E, dtype=np.float64)
        a1_full = a1_full_np; a2_full = a2_full_np; dangle_full = dangle_full_np

        for j in range(E):
            v1x = vertices_all[v1_idx[j], 0]; v1y = vertices_all[v1_idx[j], 1]
            v2x = vertices_all[v2_idx[j], 0]; v2y = vertices_all[v2_idx[j], 1]
            v1mx = v1x - cx; v1my = v1y - cy
            v2mx = v2x - cx; v2my = v2y - cy
            a1 = atan2(v1my, v1mx)
            a2 = atan2(v2my, v2mx)
            dangle = a1 - a2
            if dangle < 0.0:
                dangle += two_pi
            a1_full[j] = a1
            a2_full[j] = a2
            dangle_full[j] = dangle
            if mask_arc[j]:
                Pi_arc += r * dangle
                Ai_arc += 0.5 * (r * r) * dangle

        Pi = Pi_straight + Pi_arc
        Ai = Ai_straight + Ai_arc
        perimeter_list[i] = Pi
        area_list[i] = Ai

        # ----- dA_poly/dh, dP_poly/dh for v1 -----
        for j in range(E):
            # V2-R, V0-R
            v2x = vertices_all[v2_idx[j], 0]; v2y = vertices_all[v2_idx[j], 1]
            v0x = vertices_all[v0_idx[j], 0]; v0y = vertices_all[v0_idx[j], 1]
            v2mx = v2x - cx; v2my = v2y - cy
            v0mx = v0x - cx; v0my = v0y - cy

            # use predeclared p2x,p2y,p0x,p0y
            perp(v2mx, v2my, &p2x, &p2y)
            perp(v0mx, v0my, &p0x, &p0y)
            dAi_v1_x = -0.5 * p2x + 0.5 * p0x
            dAi_v1_y = -0.5 * p2y + 0.5 * p0y

            dPi_v1_x = 0.0; dPi_v1_y = 0.0

            # current edge j: between v1 and v2
            if mask_str[j]:
                v1x = vertices_all[v1_idx[j], 0]; v1y = vertices_all[v1_idx[j], 1]
                s12x = v1x - v2x; s12y = v1y - v2y
                l12 = (s12x*s12x + s12y*s12y) ** 0.5
                if l12 > 0.0:
                    dPi_v1_x += s12x / l12
                    dPi_v1_y += s12y / l12

            # previous edge j-1: between v0 and v1
            vprev = (j - 1 + E) % E
            mask_prev_str[j] = mask_str[vprev]
            if mask_prev_str[j]:
                v0x = vertices_all[v0_idx[j], 0]; v0y = vertices_all[v0_idx[j], 1]
                v1x = vertices_all[v1_idx[j], 0]; v1y = vertices_all[v1_idx[j], 1]
                s10x = v1x - v0x; s10y = v1y - v0y
                l10 = (s10x*s10x + s10y*s10y) ** 0.5
                if l10 > 0.0:
                    dPi_v1_x += s10x / l10
                    dPi_v1_y += s10y / l10

            vidx = v1_idx[j]
            dA_poly_dh[vidx, 0] += (Ai - A0_list[i]) * dAi_v1_x
            dA_poly_dh[vidx, 1] += (Ai - A0_list[i]) * dAi_v1_y
            dP_poly_dh[vidx, 0] += (Pi - P0) * dPi_v1_x
            dP_poly_dh[vidx, 1] += (Pi - P0) * dPi_v1_y

        # ----- arc endpoint sensitivities -----
        for j in range(E):
            if mask_arc[j]:
                # v1 endpoint
                k1 = v1_idx[j] - num_vertices_ext
                if k1 >= 0:
                    da1_full = 0.5 * (r*r) * (1.0 - cos(dangle_full[j]))
                    outer_row = k1
                    which_col = 1 if vertex_out_points[outer_row, 1] <= i else 0
                    vertex_out_da_dtheta[outer_row, which_col] = da1_full
                    vertex_out_dl_dtheta[outer_row, which_col] = r

                # v2 endpoint
                k2 = v2_idx[j] - num_vertices_ext
                if k2 >= 0:
                    da2_full = -0.5 * (r*r) * (1.0 - cos(dangle_full[j]))
                    outer_row = k2
                    which_col = 1 if vertex_out_points[outer_row, 1] <= i else 0
                    vertex_out_da_dtheta[outer_row, which_col] = da2_full
                    vertex_out_dl_dtheta[outer_row, which_col] = -r

    return (vertex_out_da_dtheta_np,
            vertex_out_dl_dtheta_np,
            dA_poly_dh_np,
            dP_poly_dh_np,
            area_list_np,
            perimeter_list_np)

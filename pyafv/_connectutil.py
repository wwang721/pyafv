"""
Experimental utilities for handling periodic boundary conditions and connectivity information.

These functions are not documented yet and may be subject to change or removal in future versions.
"""

import numpy as np

__all__ = ['tile_pbc', 'rebuild_connection_matrix', 'select_daughter_cluster', 'get_cluster_sizes']


def tile_pbc(pts: np.ndarray, L: float, r: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Periodic tiling of pts (N,2), with bookkeeping.

    Args:
        pts: (N,2) original positions
        L: Box size for periodic boundary conditions
        r: Maximum radius (or denoted as :math:`\ell`) used to determine the tiling range (at least :math:`2\ell`); if None, defaults to :math:`L/2`.
        
    Returns:
        An (M,2) array containing the tiled positions and an (M,) array containing indices mapping each tiled point back to its original index in 0..N-1.
    """
    pts = np.asarray(pts, dtype=float)
    N = pts.shape[0]

    if r is None:
          r = L / 2.0

    thresh = min(2.01 * r, L)

    x = pts[:, 0]
    y = pts[:, 1]

    # --- Stage 1: originals + left/right ---
    mask_l = x <= thresh
    mask_r = x >= (L - thresh)
    inds_l = np.flatnonzero(mask_l)
    inds_r = np.flatnonzero(mask_r)
    n_l, n_r = inds_l.size, inds_r.size

    NA = N + n_l + n_r
    pos_aug = np.empty((NA, 2), dtype=pts.dtype)
    idx_aug = np.empty(NA, dtype=np.int64)

    # originals
    pos_aug[:N] = pts
    idx_aug[:N] = np.arange(N, dtype=np.int64)
    i = N

    # left (+L, 0)
    if n_l:
        sl = slice(i, i+n_l)
        pos_aug[sl] = pts[inds_l]
        pos_aug[sl, 0] += L
        idx_aug[sl] = inds_l
        i += n_l

    # right (−L, 0)
    if n_r:
        sl = slice(i, i+n_r)
        pos_aug[sl] = pts[inds_r]
        pos_aug[sl, 0] -= L
        idx_aug[sl] = inds_r
        i += n_r

    # --- Stage 2: on augmented, append down then up ---
    y_aug = pos_aug[:, 1]
    inds_d = np.flatnonzero(y_aug <= thresh)
    inds_u = np.flatnonzero(y_aug >= (L - thresh))
    n_d, n_u = inds_d.size, inds_u.size

    M = NA + n_d + n_u
    pos_out = np.empty((M, 2), dtype=pts.dtype)
    idx_out = np.empty(M, dtype=np.int64)

    pos_out[:NA] = pos_aug
    idx_out[:NA] = idx_aug
    j = NA

    # down (0, +L)
    if n_d:
        sl = slice(j, j+n_d)
        pos_out[sl] = pos_aug[inds_d]
        pos_out[sl, 1] += L
        idx_out[sl] = idx_aug[inds_d]
        j += n_d

    # up (0, −L)
    if n_u:
        sl = slice(j, j+n_u)
        pos_out[sl] = pos_aug[inds_u]
        pos_out[sl, 1] -= L
        idx_out[sl] = idx_aug[inds_u]

    return pos_out, idx_out


# ------------- Connectivity (experimental feature) -------------

def rebuild_connection_matrix(N, connect):
    from scipy.sparse import csr_matrix

    i, j = connect[:, 0], connect[:, 1]
    rows = np.concatenate([i, j])
    cols = np.concatenate([j, i])
    data = np.ones(len(rows), dtype=bool)
    connect_matrix = csr_matrix((data, (rows, cols)), shape=(N, N))

    return connect_matrix


def select_daughter_cluster(N, connect):
    """Select a daughter cluster randomly from the connected components."""
    from scipy.sparse.csgraph import connected_components

    n_components, labels = connected_components(csgraph=rebuild_connection_matrix(N, connect), directed=False)

    # Randomly select a daughter cluster
    if n_components > 1:
        selected_idx = np.random.randint(n_components)
        selected_cells = np.where(labels == selected_idx)[0]
        N = len(selected_cells)

        selected_mask = np.isin(connect[:, 0], selected_cells) & np.isin(connect[:, 1], selected_cells)
        selected_connect = connect[selected_mask]

        # Use searchsorted to map global → local index
        selected_connect_local = np.searchsorted(selected_cells, selected_connect)

        return selected_cells, N, selected_connect_local
    else:
        return None, N, connect


def get_cluster_sizes(N, connect):
    from scipy.sparse.csgraph import connected_components

    n_components, labels = connected_components(csgraph=rebuild_connection_matrix(N, connect), directed=False)

    cluster_sizes = np.bincount(labels)
    
    return cluster_sizes, n_components, labels

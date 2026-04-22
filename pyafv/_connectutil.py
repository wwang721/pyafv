"""
Experimental utilities for handling periodic boundary conditions and connectivity information.

These functions are not fully documented yet and may be subject to change or removal in future versions.
"""

import numpy as np

__all__ = ['tile_pbc', 'rebuild_connection_matrix', 'select_daughter_cluster', 'get_cluster_sizes']


def tile_pbc(pts: np.ndarray, L: float, r: float | None = None) -> tuple[np.ndarray, np.ndarray]:    # pragma: no cover
    r"""
    Periodic tiling of pts (N,2), with bookkeeping.

    Args:
        pts: (N,2) original positions.
        L: Box size for periodic boundary conditions.
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

def rebuild_connection_matrix(N: int, connect: np.ndarray):    # pragma: no cover
    """
    Build a symmetric sparse adjacency matrix from a cell-cell edge list.

    Args:
        N: Total number of cells.
        connect: (E,2) edge list, where each row is a pair of cell indices.

    Returns:
        scipy.sparse.csr_matrix: An (N,N) boolean adjacency matrix, symmetrized so that both ``(i,j)`` and ``(j,i)`` are set for every input edge.
    """
    from scipy.sparse import csr_matrix

    i, j = connect[:, 0], connect[:, 1]
    rows = np.concatenate([i, j])
    cols = np.concatenate([j, i])
    data = np.ones(len(rows), dtype=bool)
    connect_matrix = csr_matrix((data, (rows, cols)), shape=(N, N))

    return connect_matrix


def select_daughter_cluster(N: int, connect: np.ndarray) -> tuple[np.ndarray | None, int, np.ndarray]:    # pragma: no cover
    """
    Randomly pick one connected component ("daughter cluster") from the connectivity graph.

    Args:
        N: Total number of cells.
        connect: (E,2) edge list, where each row is a pair of cell indices.

    Returns:
        tuple[numpy.ndarray | None, int, numpy.ndarray]: A *tuple* containing: an (N_sub,) :py:class:`numpy.ndarray` of global indices for cells in the chosen cluster, the cluster size ``N_sub``, and the (E_sub,2) :py:class:`numpy.ndarray` of edges re-indexed to local ``0..N_sub-1``. If the graph has only one connected component, returns ``(None, N, connect)`` unchanged.
    """
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


def get_cluster_sizes(N: int, connect: np.ndarray) -> tuple[np.ndarray, int, np.ndarray]:    # pragma: no cover
    """
    Compute the sizes of all connected components in the connectivity graph.

    Args:
        N: Total number of cells.
        connect: (E,2) edge list, where each row is a pair of cell indices.

    Returns:
        tuple[numpy.ndarray, int, numpy.ndarray]: A *tuple* containing: an (n_components,) :py:class:`numpy.ndarray` of component sizes, the number of connected components, and an (N,) :py:class:`numpy.ndarray` of component labels for each cell.
    """
    from scipy.sparse.csgraph import connected_components

    n_components, labels = connected_components(csgraph=rebuild_connection_matrix(N, connect), directed=False)

    cluster_sizes = np.bincount(labels)
    
    return cluster_sizes, n_components, labels

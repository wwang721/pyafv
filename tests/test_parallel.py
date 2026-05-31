import numpy as np
import pytest
import pyafv as afv


def _sort_connection_rows(connections):
    if connections.size == 0:
        return connections.reshape(0, 2)
    order = np.lexsort((connections[:, 1], connections[:, 0]))
    return connections[order]


def test_decompose_points_tracks_owned_and_local_ids():
    pts = np.array(
        [
            [0.0, 0.0],
            [0.5, 0.0],
            [1.0, 0.0],
            [0.0, 0.5],
            [0.5, 0.5],
            [1.0, 1.0],
        ]
    )

    domains = afv.decompose_points(
        pts,
        grid_shape=(2, 2),
        halo_width=0.1,
        domain_bounds=((0.0, 1.0), (0.0, 1.0)),
    )

    owned_global_ids = np.concatenate(
        [domain.local_global_ids[domain.owned_local_ids] for domain in domains]
    )

    assert len(domains) == 4
    assert owned_global_ids.size == pts.shape[0]
    np.testing.assert_array_equal(np.sort(owned_global_ids), np.arange(pts.shape[0]))
    for domain in domains:
        np.testing.assert_allclose(domain.local_pts, pts[domain.local_global_ids])


def test_decompose_points_methods_match():
    rng = np.random.default_rng(12)
    pts = rng.random((100, 2)) * np.array([8.0, 5.0])
    pts[:6] = np.array(
        [
            [0.0, 0.0],
            [4.0, 0.0],
            [8.0, 0.0],
            [0.0, 5.0],
            [4.0, 5.0],
            [8.0, 5.0],
        ]
    )

    dense_domains = afv.decompose_points(
        pts,
        grid_shape=(4, 3),
        halo_width=0.4,
        domain_bounds=((0.0, 8.0), (0.0, 5.0)),
        method="dense",
    )
    sorted_domains = afv.decompose_points(
        pts,
        grid_shape=(4, 3),
        halo_width=0.4,
        domain_bounds=((0.0, 8.0), (0.0, 5.0)),
        method="sorted_x",
    )
    binned_domains = afv.decompose_points(
        pts,
        grid_shape=(4, 3),
        halo_width=0.4,
        domain_bounds=((0.0, 8.0), (0.0, 5.0)),
        method="binned",
    )

    for dense_domain, sorted_domain, binned_domain in zip(
        dense_domains,
        sorted_domains,
        binned_domains,
    ):
        for domain in (sorted_domain, binned_domain):
            np.testing.assert_array_equal(
                domain.local_global_ids,
                dense_domain.local_global_ids,
            )
            np.testing.assert_array_equal(
                domain.owned_local_ids,
                dense_domain.owned_local_ids,
            )


def test_decompose_points_one_domain_allows_zero_span_axes():
    cases = [
        np.array([[1.0, 2.0]]),
        np.array([[0.0, 3.0], [1.0, 3.0]]),
        np.array([[4.0, 0.0], [4.0, 1.0]]),
    ]

    for pts in cases:
        domains = afv.decompose_points(pts, grid_shape=(1, 1), halo_width=0.5)

        assert len(domains) == 1
        domain = domains[0]
        np.testing.assert_array_equal(domain.local_global_ids, np.arange(pts.shape[0]))
        np.testing.assert_array_equal(domain.owned_local_ids, np.arange(pts.shape[0]))
        np.testing.assert_allclose(domain.local_pts, pts)


def test_parallel_one_domain_matches_serial_forces(phys):
    rng = np.random.default_rng(123)
    pts = rng.random((50, 2)) * 10.0

    serial = afv.FiniteVoronoiSimulator(pts, phys)
    parallel = afv.ParallelFiniteVoronoiSimulator(
        pts,
        phys,
        grid_shape=(1, 1),
        n_workers=1,
    )

    serial_diag = serial.build(connect=False)
    parallel_diag = parallel.build(connect=False)

    np.testing.assert_allclose(parallel_diag["forces"], serial_diag["forces"])
    np.testing.assert_allclose(parallel_diag["areas"], serial_diag["areas"])
    np.testing.assert_allclose(parallel_diag["perimeters"], serial_diag["perimeters"])
    np.testing.assert_allclose(parallel_diag["arclens"], serial_diag["arclens"])
    np.testing.assert_array_equal(parallel_diag["coord_nums"], serial_diag["coord_nums"])


def test_parallel_one_domain_matches_serial_connections(phys):
    rng = np.random.default_rng(456)
    pts = rng.random((50, 2)) * 10.0

    serial = afv.FiniteVoronoiSimulator(pts, phys)
    parallel = afv.ParallelFiniteVoronoiSimulator(
        pts,
        phys,
        grid_shape=(1, 1),
        n_workers=1,
    )

    serial_diag = serial.build(connect=True)
    parallel_diag = parallel.build(connect=True)

    np.testing.assert_allclose(parallel_diag["forces"], serial_diag["forces"])
    np.testing.assert_array_equal(
        _sort_connection_rows(parallel_diag["connections"]),
        _sort_connection_rows(serial_diag["connections"]),
    )


def test_parallel_plot_mode_returns_domain_geometry(phys):
    rng = np.random.default_rng(789)
    pts = rng.random((30, 2)) * 10.0

    serial = afv.FiniteVoronoiSimulator(pts, phys)
    parallel = afv.ParallelFiniteVoronoiSimulator(
        pts,
        phys,
        grid_shape=(1, 1),
        n_workers=1,
    )

    serial_diag = serial.build(connect=False)
    parallel_diag = parallel.build(connect=False, plot_mode=True)

    assert parallel_diag["plot_mode"]
    assert len(parallel_diag["diag_plot"]) == 1
    domain_diag = parallel_diag["diag_plot"][0]
    np.testing.assert_allclose(domain_diag["vertices"], serial_diag["vertices"])
    assert domain_diag["regions"] == serial_diag["regions"]
    assert domain_diag["edges_type"] == serial_diag["edges_type"]


def test_visualize_2d_parallel_returns_figure(phys):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.figure

    rng = np.random.default_rng(321)
    pts = rng.random((20, 2)) * 10.0

    parallel = afv.ParallelFiniteVoronoiSimulator(
        pts,
        phys,
        grid_shape=(2, 2),
        n_workers=1,
    )
    diag = parallel.build(connect=False, plot_mode=True)

    fig = afv.visualize_2d_parallel(
        pts,
        diag,
        phys.r,
        cell_colors=["C0"] * pts.shape[0],
        point_colors=["C1"] * pts.shape[0],
        point_size=np.arange(pts.shape[0]) + 1,
        show_points=True,
    )
    assert isinstance(fig, matplotlib.figure.Figure)

    fig = afv.visualize_2d_parallel(
        pts,
        diag,
        phys.r,
        selected=np.arange(0, pts.shape[0], 3),
        cell_colors=["C0"] * pts.shape[0],
        point_colors=["C1"] * pts.shape[0],
        point_size=np.arange(pts.shape[0]) + 1,
        show_points=True,
    )
    assert isinstance(fig, matplotlib.figure.Figure)


def test_visualize_helpers_reject_wrong_diag_shape(phys):
    rng = np.random.default_rng(654)
    pts = rng.random((20, 2)) * 10.0

    serial_diag = afv.FiniteVoronoiSimulator(pts, phys).build(connect=False)
    parallel = afv.ParallelFiniteVoronoiSimulator(
        pts,
        phys,
        grid_shape=(2, 2),
        n_workers=1,
    )
    parallel_diag = parallel.build(connect=False, plot_mode=True)

    with pytest.raises(ValueError, match="visualize_2d_parallel"):
        afv.visualize_2d(pts, parallel_diag, phys.r)
    with pytest.raises(ValueError, match="build\\(plot_mode=True\\)"):
        afv.visualize_2d_parallel(pts, serial_diag, phys.r)

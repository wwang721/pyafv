# MPI_wrap.py

import numpy as np
import pyafv as afv


def main():
    # ========================================================
    # Put MPI setup inside main() so spawned multiprocessing
    # workers do not import/initialize MPI at top level.
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()    # should be 2 for this test
    # ========================================================

    radius = 1.0
    points = np.random.default_rng(42).random((10_000, 2)) * 100.0
    phys = afv.PhysicalParams(r=radius)

    # 2x1 domain decomposition for the two MPI ranks
    if rank == 0:
        domains = afv.decompose_points(points, (2, 1), halo_width=4.01*radius)

    domains = comm.bcast(domains if rank == 0 else None, root=0)

    # Each rank processes its own domain
    domain = domains[rank]
    local_points = domain.local_pts

    # Each rank creates its own simulator with 4 workers
    sim = afv.ParallelFiniteVoronoiSimulator(local_points, phys, (2, 2), n_workers=4)

    dt = 0.01
    with sim:
        diag = sim.build()

        # gather diagnostics from all ranks
        diag_all = comm.gather(diag, root=0)

        # combine diagnostics on rank 0
        if rank == 0:
            forces = np.zeros_like(points, dtype=float)
            areas = np.zeros(points.shape[0], dtype=float)
            perimeters = np.zeros(points.shape[0], dtype=float)
            # ... add more diagnostics as needed ...

            for mpi_domain, diag in zip(domains, diag_all):
                owned_local_ids = mpi_domain.owned_local_ids
                owned_global_ids = mpi_domain.local_global_ids[owned_local_ids]

                forces[owned_global_ids] = diag["forces"][owned_local_ids]
                areas[owned_global_ids] = diag["areas"][owned_local_ids]
                perimeters[owned_global_ids] = diag["perimeters"][owned_local_ids]

            diag_combined = {
                "forces": forces,
                "areas": areas,
                "perimeters": perimeters,
            }

            # Update points
            points += forces * dt

            # upate domain decomposition with new points
            domains = afv.decompose_points(points, (2, 1), halo_width=4.01*radius)

        domains = comm.bcast(domains, root=0)
        domain = domains[rank]
        local_points = domain.local_pts
        sim.update_positions(local_points)

    print(f"Rank {rank} finished simulation.")


if __name__ == "__main__":
    main()

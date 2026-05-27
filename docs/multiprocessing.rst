Multiprocessing
===============

Starting in v0.4.12, PyAFV provides
:py:class:`pyafv.ParallelFiniteVoronoiSimulator` for domain-decomposed AFV
simulations using **Python multiprocessing** (CPU parallelism). The
simulator splits the full point set into rectangular owned domains, adds halo
points around each domain, builds a local finite Voronoi diagram for each
subdomain, and merges the owned-cell diagnostics back into the global point
ordering.

This feature is intended for large systems (:math:`N \gtrsim 10^4`) where the
cost of local Voronoi builds is high enough to offset the overhead of domain
decomposition, inter-process data transfer, and duplicated halo work. For small
or moderate systems, :py:class:`pyafv.FiniteVoronoiSimulator` may still be
faster.

.. note::

   See :ref:`bench_parallel_build` for a build-time benchmark comparing
   :py:class:`pyafv.FiniteVoronoiSimulator` with
   :py:class:`pyafv.ParallelFiniteVoronoiSimulator`. The benchmark shows that
   multiprocessing is not always faster.


Basic usage
-----------

The interface is similar to :py:class:`pyafv.FiniteVoronoiSimulator`, but the
domain grid shape and number of worker processes are supplied when the
simulator is created:

.. code-block:: python

   import numpy as np
   import pyafv

   points = np.random.default_rng(42).random((10_000, 2)) * 100.0
   phys = pyafv.PhysicalParams(r=1.0)

   sim = pyafv.ParallelFiniteVoronoiSimulator(
       points,
       phys,
       grid_shape=(4, 4),
       n_workers=16,
   )

   diag = sim.build()

PyAFV decomposes the domain into an ``a``-by-``b`` grid of subdomains set by
``grid_shape=(a, b)``. The number of subdomains is therefore :math:`ab`.
``n_workers`` is the number of worker processes to use.
In practice, we recommend setting ``n_workers`` to the number of available
CPU cores, but no larger than the number of subdomains, since any additional
workers will remain idle anyway.

By default, :py:meth:`pyafv.ParallelFiniteVoronoiSimulator.build` uses
``connect=False``. This differs from
:py:meth:`pyafv.FiniteVoronoiSimulator.build`, where ``connect=True`` by
default. This default avoids connectivity work during runs where only forces
are needed.

.. tip::

   Decomposing the whole system into smaller domains can also improve the
   accuracy of :py:class:`scipy.spatial.Voronoi` for large systems, since
   Qhull's floating-point tolerance scales with the system span; see
   `issue #38 <https://github.com/wwang721/pyafv/issues/38#issuecomment-4189891733>`_.


Repeated build steps
--------------------

For repeated calls with ``n_workers > 1``, put the time-stepping loop inside
the context manager. This creates the worker processes once and reuses them
across build steps:

.. code-block:: python

   dt = 0.01
   n_steps = 100

   with sim:
       for step in range(n_steps):
           diag = sim.build()
           points += diag["forces"] * dt
           sim.update_positions(points)

If the context manager is not used, each call to ``build`` creates and shuts
down a new process pool. That is usually slower in a loop.

.. important::

   When using multiprocessing in a Python script, put the executable code
   behind the standard Python guard:

   .. code-block:: python

      def main():
          # Initialize points, phys, and n_steps here.
          sim = pyafv.ParallelFiniteVoronoiSimulator(points, phys, (4, 4), 16)
          with sim:
              for step in range(n_steps):
                  diag = sim.build()
                  # followed by time-stepping code...

      if __name__ == "__main__":
          main()

   This guard is required when Python uses the ``spawn`` multiprocessing start
   method. This includes **Windows** and modern **macOS** by default; **Linux**
   usually defaults to ``fork``, but the guard is still recommended for
   portable scripts.

   In Jupyter notebooks, the parallel simulator may still work even if this guard is not used,
   but long production runs are usually more robust when launched from a script.


Halo width
----------

Each owned domain is expanded by ``halo_width`` in every direction before the
local Voronoi calculation is built. If ``halo_width`` is not specified, PyAFV
uses ``4.01 * phys.r`` (:math:`>4\ell`). This should be large enough that the
geometry and force for an owned cell are not affected by missing neighboring
cells outside the local domain.


Decomposition method
--------------------

The low-level helper :py:func:`pyafv.decompose_points` and the parallel
simulator both support two halo-collection methods:

.. code-block:: python

   sim = pyafv.ParallelFiniteVoronoiSimulator(
       points,
       phys,
       grid_shape=(4, 4),
       n_workers=16,
       decomposition_method="dense",
   )

- ``"dense"`` is the default. It builds a dense domain-by-point mask and is
  often faster for moderate systems.
- ``"sorted_x"`` avoids the dense temporary mask by sorting points along the
  ``x``-axis and querying candidate halo ranges. It uses less temporary memory,
  but can be slower for typical moderate-sized systems.


Visualization
-------------

Parallel plotting diagnostics are local to each domain and should be requested
explicitly:

.. code-block:: python

   import matplotlib.pyplot as plt

   diag = sim.build(plot_mode=True)
   fig, ax = plt.subplots()
   pyafv.visualize_2d_parallel(points, diag, r=phys.r, ax=ax)
   plt.show()

Use :py:func:`pyafv.visualize_2d_parallel` for diagnostics from
:py:meth:`pyafv.ParallelFiniteVoronoiSimulator.build`; ``plot_mode`` must be
set to ``True``. Use :py:func:`pyafv.visualize_2d` for diagnostics from
:py:meth:`pyafv.FiniteVoronoiSimulator.build`.


Running on clusters
-------------------

Python multiprocessing runs worker processes on the same node as the main
Python process. It does not distribute work across multiple nodes. On a SLURM
cluster, use one task with multiple CPUs, for example:

.. code-block:: bash

   #SBATCH --ntasks=1
   #SBATCH --cpus-per-task=16

Then use the same number of workers in Python:

.. code-block:: python

   sim = pyafv.ParallelFiniteVoronoiSimulator(points, phys, (4, 4), 16)

For multi-node domain decomposition, use an MPI-based implementation instead of
Python multiprocessing. PyAFV does not currently provide an MPI implementation.

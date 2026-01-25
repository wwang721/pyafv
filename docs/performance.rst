Performance
================

Measuring performance
----------------------

.. figure:: ../assets/runtime_comparison.png
   :alt: Runtime comparison
   :figwidth: 50%
   :align: right

**PyAFV** has been benchmarked against the **MATLAB** implementation of the active-finite-Voronoi model from Ref. :cite:`huang2023bridging` by measuring the wall-clock runtime for simulations of varying system sizes. The results are shown in the figure; each data point corresponds to :math:`10^3` integration steps, averaged over three independent runs. The results show that *PyAFV* exhibits **near-linear** scaling, approximately :math:`\mathcal{O}(N)`, whereas the original *MATLAB* code appears to scale as :math:`\mathcal{O}(N^{3/2})`. This difference leads to a significant speedup, particularly for large systems (:math:`N\gtrsim 10^3`).

.. note::
    
   All benchmark results were obtained on a MacBook Pro (14-in, 2024) equipped with an Apple M4 Pro chip (12-core) and 24 GB of RAM, running macOS 15.6. The *MATLAB* implementation was executed using **MATLAB R2025a**, while *PyAFV* was run using **Python 3.13.5** with the **PyAFV v0.4.1** default Cython backend.


.. _bench_backends:

Benchmarking backends
---------------------

In addition, there is a set of lightweight benchmarks in ``tests`` using **pytest-benchmark**, e.g., ``test_bench_build.py`` compares the runtimes of the Cython and pure-Python backends . To run it:

.. code-block:: console

   (.venv) $ uv run pytest tests/test_bench_build.py --benchmark-only --benchmark-warmup on --benchmark-histogram

This will display the benchmark results and generate an interactive SVG histogram file (click to see the detailed timing results for each method):

.. image:: ../assets/pytest_benchmark.svg
   :alt: Pytest benchmark histogram
   :width: 100%
   :align: center

The histogram above summarizes the runtimes of the core routines invoked by :py:meth:`pyafv.FiniteVoronoiSimulator.build` for a system of :math:`N=1000` cells. The ``test_scipy_voronoi`` benchmark measures the execution time of **SciPy**'s Voronoi tessellation, which serves as *a baseline for comparison*. This SciPy routine is called internally by :py:meth:`pyafv.FiniteVoronoiSimulator._build_voronoi_with_extensions`, corresponding to the ``test_build_voronoi`` benchmark shown in the histogram. From this comparison, we see that SciPy's Voronoi computation accounts for approximately 60% of the total runtime of that method.

.. hint::

   The suffixes ``[accel]`` and ``[fallback]`` in the benchmark names indicate whether the Cython backend or the pure-Python fallback implementation was used.


The remaining dominant cost arises from the additional per-cell processing performed in :py:meth:`pyafv.FiniteVoronoiSimulator._per_cell_geometry`. As shown in the histogram, the Cython-backed implementation substantially reduces the runtime of this step, bringing it down to a level comparable to that of SciPy's Voronoi tessellation.

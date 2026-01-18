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
    
   All benchmark results were obtained on a MacBook Pro (14-in, 2024) equipped with an Apple M4 Pro chip (12-core) and 24 GB of RAM, running macOS 15.6. The *MATLAB* implementation was executed using **MATLAB R2025a**, while *PyAFV* was run using **Python 3.13.5** with the **PyAFV v0.3.9** default Cython backend.


Benchmarking backends
---------------------

In addition, there is a set of lightweight benchmarks in ``tests/test_benchmarks.py`` that compare the **PyAFV** Cython backend with the pure-Python implementation using **pytest-benchmark**. To run them:

.. code-block:: console

   (.venv) $ uv run pytest --benchmark-only --benchmark-warmup on --benchmark-histogram

This will display the benchmark results and generate an SVG histogram file (click to see the detailed timing results for each method):

.. image:: ../assets/pytest_benchmark.svg
   :alt: Pytest benchmark histogram
   :width: 100%
   :align: center

|

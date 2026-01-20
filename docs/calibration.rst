Calibration
===========

**PyAFV** provides a dedicated subpackage :py:mod:`pyafv.calibrate` for calibrating the physical parameters of the finite-Voronoi (FV) model against a vertex-model-like deformable-polygon (DP) model. The calibration is performed by matching the steady states and detachment forces of cell doublets between the two models :cite:`wang2026divergence`.

.. important::

   *To make it clear, the calibration tools described here are intended for advanced users who are familiar with the underlying assumptions and know exactly what they are doing.*

In most use cases, the default value of the contact truncation threshold :py:attr:`delta` in :py:class:`pyafv.PhysicalParams` is sufficient and should work well.



.. note::

   In vertex and Voronoi models, the target shape index is defined as
   :math:`p_0 = P_0 / \sqrt{A_0}`.
   The DP model is expected to be valid only for
   :math:`p_0 \leqslant 2\sqrt{\pi}`, corresponding to the shape index of a perfect circle.
   This limitation arises because no additional constraints (e.g., curvature terms)
   are included in the energy to stabilize non-circular shapes in the DP model.


How to calibrate against the DP model
--------------------------------------

**PyAFV** provides a convenience function :py:func:`pyafv.calibrate.auto_calibrate`, which performs the calibration procedure automatically.

.. autofunction:: pyafv.calibrate.auto_calibrate
   :noindex:

In brief, the calibration procedure is as follows:

1. Match the steady-state geometry of a cell doublet in the FV and DP models by determining the optimal cell radius :math:`\ell_0`; this can be done by :py:meth:`pyafv.PhysicalParams.get_steady_state` or :py:meth:`pyafv.PhysicalParams.with_optimal_radius`.

2. Apply progressively increasing pulling forces to the cell doublet in the DP model until detachment occurs, and record the corresponding detachment forces; this can be done by :py:class:`pyafv.calibrate.DeformablePolygonSimulator` (see :ref:`section <DP_model_simulator>` below).

3. Identify the value of :py:attr:`delta` in the FV model that reproduces the same detachment forces observed in the DP model; this can be done by :py:meth:`pyafv.target_delta`.

.. list-table::
   :widths: 33 33 33

   * - .. figure:: ../assets/DP1.svg
          :width: 100%
          :align: center

          Steady state

     - .. figure:: ../assets/DP2.svg
          :width: 100%
          :align: center

          External forces applied

     - .. figure:: ../assets/DP3.svg
          :width: 100%
          :align: center

          Before detachment

A detailed description of the calibration procedure and the corresponding results on tissue fracture timescales are provided in Ref. :cite:`wang2026divergence`.


.. _DP_model_simulator:

Usage of the DP simulator
--------------------------

In addition to the FV simulator, **PyAFV** includes a :py:class:`pyafv.calibrate.DeformablePolygonSimulator` class for simulating cell doublets with the DP model, which can be used for calibration as well as standalone analyses.

.. autoclass:: pyafv.calibrate.DeformablePolygonSimulator
   :noindex:

   .. autosummary::

      ~DeformablePolygonSimulator.detached
      ~simulate
      ~plot_2d


Below, we present a minimal example illustrating how the DP simulator is used internally by :py:func:`pyafv.calibrate.auto_calibrate`:

.. literalinclude:: ../examples/calibrate_DP.py
   :language: python

The generated figures are shown above.

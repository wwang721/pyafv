Examples
=========


.. note::
   
   To run the examples below, install the **tqdm** package for progress bars (using *pip*). The *Jupyter Notebooks* in `examples/jupyter/ <https://github.com/wwang721/pyafv/tree/main/examples/jupyter/>`_ additionally require **jupyter** and **ipywidgets** as well.


Relaxation to mechanical equilibrium
------------------------------------

The following example shows how 100 cells relax to mechanical equilibrium from a squeezed initial configuration using the **PyAFV** package.

.. literalinclude:: ../examples/relaxation.py
   :language: python

See the plotted figures below:

.. list-table::
   :widths: 50 50

   * - .. figure:: ../assets/initial_configuration.png
          :width: 100%
          :align: center

          Initial configuration.

     - .. figure:: ../assets/relaxed_configuration.png
          :width: 100%
          :align: center

          After relaxation.

|

Active-Finite-Voronoi (AFV) dynamics
-------------------------------------

We can incorporate self-propulsion (active-Brownian-like dynamics) for each cell to model active-matter systems.
The resulting equation of motion is

.. math::

   \dot{\mathbf{r}}_i = -\mu \nabla_i E + v_0 \mathbf{n}_i,

where :math:`\mu` is the mobility, the interaction force on cell :math:`\mathbf{F}_i=-\nabla_i E`, and
:math:`\mathbf{n}_i = (\cos \theta_i, \sin \theta_i)` is a unit orientation vector.
The orientation evolves according to

.. math::

   \dot{\theta}_i = \sqrt{2 D_r}\,\eta_i(t),

where the noise satisfies :math:`\langle \eta_i(t) \rangle = 0` and :math:`\langle \eta_i(t)\eta_j(t') \rangle = \delta_{ij}\,\delta(t - t')`.

.. literalinclude:: ../examples/active_FV.py
   :language: python

See the plotted figure below:

.. image:: ../assets/active_FV.png
   :alt: AFV dynamics
   :width: 500px
   :align: center

|

Connectivity between cells
-------------------------------

**PyAFV** can directly output the cell-cell connectivity from the finite Voronoi diagram, where any two connected cells share a straight Voronoi edge.

.. literalinclude:: ../examples/connections.py
   :language: python

|

Custom plotting
----------------

See `examples/jupyter/custom_plot.ipynb <https://github.com/wwang721/pyafv/blob/main/examples/jupyter/custom_plot.ipynb>`_ for an example of custom plotting using **PyAFV**.

This example shows how to use :py:func:`pyafv.FiniteVoronoiSimulator.build` returned ``dict`` to plot the Voronoi diagram with custom styling, including coloring cells by their area and customizing edge colors and widths.

.. automethod:: pyafv.FiniteVoronoiSimulator.build
   :noindex:

.. image:: ../assets/model_illustration.png
   :alt: Custom plotting
   :width: 600px
   :align: center

|

This example also shows how to access additional internal information via :py:func:`pyafv.FiniteVoronoiSimulator._build_voronoi_with_extensions` and :py:func:`pyafv.FiniteVoronoiSimulator._per_cell_geometry` for advanced plotting. The public ``build()`` method serves as a higher-level wrapper around these two and other lower-level routines.

.. automethod:: pyafv.FiniteVoronoiSimulator._build_voronoi_with_extensions
    :noindex:

.. automethod:: pyafv.FiniteVoronoiSimulator._per_cell_geometry
    :noindex:

|

Periodic boundary conditions
----------------------------

**PyAFV** uses open boundary conditions in 2D by default, but it is also possible to implement periodic boundary conditions via a tiling of the edge regions.
See `examples/jupyter/periodic_plotting.ipynb <https://github.com/wwang721/pyafv/blob/main/examples/jupyter/periodic_plotting.ipynb>`_ for an example, and the generated figure is shown below:

.. image:: ../assets/pbc.png
   :alt: PBC example
   :width: 500px
   :align: center

|

Varying cell target areas from cell to cell
-------------------------------------------

Starting from **PyAFV** v0.3.5, the simulator supports cell-specific preferred areas, allowing the target area :math:`A_0` to vary from cell to cell.

A new read-only property :py:attr:`pyafv.FiniteVoronoiSimulator.preferred_areas` has been added. It returns the current preferred areas of all cells:

.. autoattribute:: pyafv.FiniteVoronoiSimulator.preferred_areas
    :noindex:

To modify the preferred areas, the method :py:func:`pyafv.FiniteVoronoiSimulator.update_preferred_areas()` is provided:

.. automethod:: pyafv.FiniteVoronoiSimulator.update_preferred_areas
    :noindex:

Here is an example usage:

.. code-block:: python

    import numpy as np
    from pyafv import FiniteVoronoiSimulator, PhysicalParams

    # Initialize simulator
    N = 100
    pts = np.random.rand(N, 2) * 10
    phys = PhysicalParams(r=1.0, A0=np.pi)
    sim = FiniteVoronoiSimulator(pts, phys)

    # Set varying preferred areas per cell
    varying_A0 = np.pi + 0.2 * np.random.randn(N)
    sim.update_preferred_areas(varying_A0)

    # Access via property
    print(sim.preferred_areas)  # shape: (N,)

    # Run simulation...

In addition, the :py:func:`pyafv.FiniteVoronoiSimulator.update_positions()` method now accepts an optional second argument to update the preferred areas:

.. automethod:: pyafv.FiniteVoronoiSimulator.update_positions
    :noindex:

And the :py:func:`pyafv.FiniteVoronoiSimulator.update_params()` method will also re-initialize the preferred areas for all cells using the supplied value of *A0* in *phys*:

.. automethod:: pyafv.FiniteVoronoiSimulator.update_params
    :noindex:

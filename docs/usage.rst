Getting started
===============

.. _installation:

Install
------------

**PyAFV** supports *Python* ≥ 3.9, < 3.15. You can install the package directly using *pip*:

.. code-block:: console

   (.venv) $ pip install pyafv

After installation, verify that it was successful by importing the package in *Python* and checking the version:

.. doctest::
   :options: +NORMALIZE_WHITESPACE

   >>> import pyafv
   >>> pyafv.__version__
   '0.3.4'


A simple example
----------------

Begin by importing the required libraries and generating 100 random points in two dimensions:

.. code-block:: python

   import numpy as np
   import pyafv

   N = 100                                           # number of cells
   pts = np.random.rand(N, 2) * 10                   # initial positions

Next, create a :py:class:`pyafv.PhysicalParams` object to specify the physical parameters of the simulation:

.. code-block:: python

   params = pyafv.PhysicalParams(r=1.0)              # use default parameter values

Finally, initialize the simulator by constructing :py:class:`pyafv.FiniteVoronoiSimulator` instance and visualize the resulting Voronoi diagram:

.. code-block:: python

   sim = pyafv.FiniteVoronoiSimulator(pts, params)   # initialize the simulator
   sim.plot_2d(show=True)                            # visualize the Voronoi diagram

The plotting routine ``plot_2d()`` is provided by:

.. autofunction:: pyafv.FiniteVoronoiSimulator.plot_2d 
   :noindex:




.. To retrieve a list of random ingredients,
.. you can use the ``lumache.get_random_ingredients()`` function:


.. The ``kind`` parameter should be either ``"meat"``, ``"fish"``,
.. or ``"veggies"``. Otherwise, :py:func:`lumache.get_random_ingredients`
.. will raise an exception.

.. .. autoexception:: lumache.InvalidKindError

.. For example:

.. >>> import lumache
.. >>> lumache.get_random_ingredients()
.. ['shells', 'gorgonzola', 'parsley']


Getting started
===============

.. _install:

Installation
------------

To use **PyAFV**, first install it using pip:

.. code-block:: console

   (.venv) $ pip install pyafv

A simple example
----------------

Start by importing the library:

.. code-block:: python

   import pyafv
   print(pyafv.__version__)

Then we generate 100 random points in 2D:

.. code-block:: python

   import numpy as np
   N = 100                                           # number of cells
   pts = np.random.rand(N, 2) * 10                   # initial positions

Next, we create a :py:class:`pyafv.PhysicalParams` object to define the parameters of the simulation:

.. code-block:: python

   params = pyafv.PhysicalParams(r=1.0)              # use default parameter values

Finally, we create a :py:class:`pyafv.FiniteVoronoiSimulator` instance as the simulator ``sim`` and then plot the Voronoi diagram:

.. code-block:: python

   sim = pyafv.FiniteVoronoiSimulator(pts, params)   # initialize the simulator
   sim.plot_2d(show=True)                            # visualize the Voronoi diagram

where the ``sim.plot_2d()`` function is:

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


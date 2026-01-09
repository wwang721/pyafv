Getting started
===============

.. _installation:

Installation
------------

**PyAFV** supports *Python* ≥ 3.10, < 3.15, and has been tested on major operating systems including *Linux*, *macOS*, and *Windows*, for both *x86-64* and *ARM64* architectures.

Install using pip
^^^^^^^^^^^^^^^^^^

The package is available on |PyPI|_, so you should be able to install it using *pip* directly:

.. code-block:: console

   (.venv) $ pip install pyafv

After installation, verify that it was successful by importing the package in *Python* and checking the version:

.. doctest::
   :options: +NORMALIZE_WHITESPACE

   >>> import pyafv
   >>> pyafv.__version__
   '0.3.5'


Install from source
^^^^^^^^^^^^^^^^^^^^

.. |GitHub| replace:: **GitHub**
.. _GitHub: https://github.com/wwang721/pyafv
.. |PyPI| replace:: **PyPI**
.. _PyPI: https://pypi.org/project/pyafv/


Installing from source can be necessary if *pip* installation does not work.
First, download the source code of *pyafv* from |GitHub|_ or |PyPI|_.

Required prerequisites
"""""""""""""""""""""""""""

The required packages are listed in the table below:

+----------------+-------------------------------+-------------------------------+
| Package        | Minimum Version               | Usage                         |
+================+===============================+===============================+
| numpy          | 1.26.4                        | Numerical computations        |
+----------------+-------------------------------+-------------------------------+
| scipy          | 1.13.1                        | Scientific computations       |
+----------------+-------------------------------+-------------------------------+
| matplotlib     | 3.8.4                         | Plotting and visualization    |
+----------------+-------------------------------+-------------------------------+

Unzip the downloaded source code and navigate to the root directory of the project. Then, run the following command to install the package:

.. code-block:: console

   (.venv) $ pip install .

.. note::

   A **C/C++** compiler is required if you are building from source, since some components of **PyAFV** are implemented in **Cython** for performance optimization.

Windows MinGW GCC
"""""""""""""""""""""""""""
If you are using **MinGW GCC** (rather than **MSVC**) on *Windows*, to build from the source code, add a ``setup.cfg`` at the repository root before running the installation command above, with the following content:

.. code-block:: ini

   # setup.cfg
   [build_ext]
   compiler=mingw32


A simple example
----------------

Now that you have installed **PyAFV**, here is a simple example to get you started.
Begin by importing the required libraries and generating 100 random points in two dimensions:

.. code-block:: python

   import numpy as np
   import pyafv

   N = 100                                           # number of cells
   pts = np.random.rand(N, 2) * 10                   # initial positions

Next, create a :py:class:`pyafv.PhysicalParams` object to specify the physical parameters of the simulation:

.. code-block:: python

   params = pyafv.PhysicalParams(r=1.0)              # use default parameter values

.. autoclass:: pyafv.PhysicalParams
   :noindex:

Finally, initialize the simulator by constructing a :py:class:`pyafv.FiniteVoronoiSimulator` instance and visualize the resulting Voronoi diagram:

.. code-block:: python

   sim = pyafv.FiniteVoronoiSimulator(pts, params)   # initialize the simulator
   sim.plot_2d(show=True)                            # visualize the Voronoi diagram

The plotting routine ``plot_2d()`` is provided by:

.. automethod:: pyafv.FiniteVoronoiSimulator.plot_2d 
   :noindex:

.. image:: ../assets/first_example.png
   :alt: A simple example
   :width: 500px
   :align: center

|

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


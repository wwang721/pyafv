Getting started
===============

.. _installation:

Installation
------------

**PyAFV** supports *Python* >= 3.10, < 3.15, and has been tested on major operating systems including *Linux*, *macOS*, and *Windows*, for both *x86-64* and *ARM64* architectures.

.. image:: https://github.com/wwang721/pyafv/actions/workflows/tests_all_platform.yml/badge.svg
   :target: https://github.com/wwang721/pyafv/actions/workflows/tests_all_platform.yml
   :alt: Tests on all platforms

.. note::
   
   - Python 3.14t, the **free-threaded** build that runs without the *Global Interpreter Lock (GIL)*, is also supported starting with **PyAFV** v0.3.8.
   - Python 3.10 and 3.14t (free-threaded) on *Windows ARM64* (not *x86-64*) are the only untested configurations. The builds succeed and the wheels are available on |PyPI|_, but automated testing is unavailable due to the absence of a supported **GitHub Actions** runners for these configurations.


Install using pip
^^^^^^^^^^^^^^^^^^

.. only:: html

   .. image:: https://img.shields.io/pypi/v/pyafv?cacheSeconds=300
      :target: https://pypi.org/project/pyafv
      :alt: PyPI

..
   .. image:: https://img.shields.io/pypi/dm/pyafv.svg?cacheSeconds=43200
      :target: https://pypi.org/project/pyafv
      :alt: Downloads

The package is available on |PyPI|_, so you should be able to install it using *pip* directly:

.. code-block:: console

   (.venv) $ pip install pyafv

After installation, verify that it was successful by importing the package in *Python* and checking the version:

.. subst-code-block:: pycon

   >>> import pyafv
   >>> pyafv.__version__
   '|release|'

.. note::

   On some HPC clusters, global Python path can contaminate the runtime environment. You may need to clear it explicitly using ``unset PYTHONPATH`` or prefixing the *pip* command with ``PYTHONPATH=""``.


Install from source
^^^^^^^^^^^^^^^^^^^^

.. |GitHub| replace:: **GitHub**
.. _GitHub: https://github.com/wwang721/pyafv
.. |PyPI| replace:: **PyPI**
.. _PyPI: https://pypi.org/project/pyafv/
.. |GitHub Packages| replace:: **GitHub Packages**
.. _GitHub Packages: https://github.com/wwang721/pyafv/pkgs/container/pyafv
.. |Docker Hub| replace:: **Docker Hub**
.. _Docker Hub: https://hub.docker.com/r/wwang721/pyafv


Installing from source can be necessary if *pip* installation does not work.
First, download the source code of *pyafv* from |GitHub|_ or |PyPI|_.

Required prerequisites
"""""""""""""""""""""""""""

In general, you do not need to manually install the dependencies, as *pip* will handle them automatically.
We list the required packages and minimum versions below for reference:

+----------------+-------------------------------+------------------------------------+
| Package        | Minimum Version               | Usage                              |
+================+===============================+====================================+
| numpy          | 1.26.4                        | Numerical computations             |
+----------------+-------------------------------+------------------------------------+
| scipy          | 1.13.1                        | Miscellaneous scientific functions |
+----------------+-------------------------------+------------------------------------+
| matplotlib     | 3.8.4                         | Plotting and visualization         |
+----------------+-------------------------------+------------------------------------+

Unzip the downloaded source code and navigate to the root directory of the package. Then, run the following command to install:

.. code-block:: console

   (.venv) $ pip install .

.. note::

   A **C/C++** compiler is required if you are building from source, since some components of **PyAFV** are implemented in **Cython** for performance optimization.

Windows MinGW GCC
"""""""""""""""""""""""""""
If you are using **MinGW GCC** (rather than **MSVC**) on *Windows*, to build from the source code, add a ``setup.cfg`` file at the repository root before running ``pip install .`` with the following content:

.. code-block:: ini

   # setup.cfg
   [build_ext]
   compiler=mingw32


Install offline
^^^^^^^^^^^^^^^

If you need to install **PyAFV** on a machine without internet access, you can download the corresponding wheel file from |PyPI|_ on another machine with internet access. Transfer the wheel file to the target machine, and then run the following command to install it using *pip* (make sure the required prerequisites listed above are already installed):

.. code-block:: console

   (.venv) $ pip install pyafv-<version>-<platform>.whl

Alternatively, you can build **PyAFV** from source as described in the previous section. In this case, in addition to the required prerequisites listed above, the build-time dependencies **hatchling** and **hatch-cython** must also be available.


Install using Docker
^^^^^^^^^^^^^^^^^^^^

.. only:: html
   
   .. image:: https://img.shields.io/docker/pulls/wwang721/pyafv.svg?logo=docker
      :target: https://hub.docker.com/r/wwang721/pyafv
      :alt: Docker

Pull the Docker image from |Docker Hub|_:

.. code-block:: console

   (.venv) $ docker pull wwang721/pyafv:latest

It's also available in the **GitHub Container Registry (GHCR)** under |GitHub Packages|_; use ``ghcr.io/wwang721/pyafv`` to pull from GHCR instead.
Then run Python scripts with `pyafv` using:

.. code-block:: console

   (.venv) $ docker run --rm -v $(pwd):/app wwang721/pyafv python <script_name>.py

Use ``${PWD}`` on Windows PowerShell instead of ``$(pwd)``.


A simple example
----------------

.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/pyafv/assets/blob/main/jupyter/getting_started.ipynb
   :alt: Open In Colab

Now that you have installed **PyAFV**, here is a simple example to get you started (click the **Google Colab** badge above to run the notebook directly).
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

To compute the conservative forces and extract detailed geometric information (e.g., cell areas, vertices, and edges), call:

.. code-block:: python

   diag = sim.build()                                       # compute forces and geometry

The returned object ``diag`` is a Python ``dict`` containing these quantities.

.. automethod:: pyafv.FiniteVoronoiSimulator.build
   :noindex:

For more examples and detailed usage instructions, please refer to the :doc:`examples` and :doc:`api/index` sections.

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


.. .. include:: ../README.rst

PyAFV
======

.. image-release:: https://img.shields.io/badge/Version-|release|-orange.svg?logo=git
   :target: https://pyafv.github.io
   :alt: Version

.. image:: https://img.shields.io/badge/GitHub-pyafv-brightgreen?logo=github
   :target: https://github.com/wwang721/pyafv
   :alt: GitHub

.. image:: https://img.shields.io/badge/License-MIT-green.svg
   :target: https://opensource.org/licenses/MIT
   :alt: MIT License

.. image:: ../assets/model_illustration.png
   :width: 300px
   :alt: Model illustration
   :align: right

**PyAFV** is a Python implementation of the **active finite Voronoi (AFV) model** in 2D.

The AFV framework was introduced and developed in, for example, Refs. :cite:`huang2023bridging,teomy2018confluent,wang2026divergence`.

.. It pulls data from the `Open Food Facts database <https://world.openfoodfacts.org/>`_
.. and offers a *simple* and *intuitive* API.

Check out the :doc:`usage`, :doc:`examples`, :doc:`multiprocessing`, :doc:`performance`, :doc:`calibration`, :doc:`citation`, :doc:`contributing`, and :doc:`api/index` sections for further information, including how to :ref:`install <installation>` the package, usage examples, multiprocess domain decomposition, benchmarks, local development, and the complete API reference.

.. |simulation demo| replace:: *simulation demo*
.. _simulation demo: https://dapengbi.com/paper_simulation_demos/afv_model/index.html

.. |collection| replace:: *collection*
.. _collection: https://colab.research.google.com/github/pyafv/assets/blob/main/jupyter/index.ipynb

.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/pyafv/assets/blob/main/jupyter/index.ipynb
   :alt: Open In Colab

- *Explore a* |collection|_ *of usage examples in Jupyter notebooks on Google Colab.*
- *See also an interactive* |simulation demo|_ *using PyAFV on Prof. Dapeng (Max) Bi's homepage!*

.. |GitHub| replace:: **GitHub**
.. _GitHub: https://github.com/wwang721/pyafv

.. note::

   For the latest updates, see the |GitHub|_ repository.


.. image:: ../assets/pbc.png
   :width: 300px
   :alt: Periodic boundary conditions
   :align: right


.. rubric:: Contents

.. toctree::
   :maxdepth: 2
   :numbered:

   Home <self>
   usage
   examples
   multiprocessing
   performance
   calibration
   citation
   contributing
   api/index


.. caution::

   Future versions may introduce changes to features and APIs.


.. rubric:: References

.. bibliography:: ./main.bib
   :style: unsrt


.. rubric:: Indices

* :ref:`genindex`
* :ref:`modindex`

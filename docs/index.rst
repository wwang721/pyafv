.. .. include:: ../README.rst

PyAFV
======

.. image-release:: https://img.shields.io/badge/Version-|release|-orange.svg?logo=git
   :target: https://wwang721.github.io/pyafv
   :alt: PyPI

.. image:: https://img.shields.io/badge/GitHub-pyafv-brightgreen?logo=github
   :target: https://github.com/wwang721/pyafv
   :alt: GitHub

.. image:: https://img.shields.io/badge/License-MIT-green.svg
   :target: https://opensource.org/licenses/MIT
   :alt: MIT License

.. image:: ../assets/model_illustration.png
   :width: 300px
   :align: right

**PyAFV** is a Python implementation of the **active-finite-Voronoi (AFV) model** in 2D.

The AFV framework was introduced and developed in, for example, Refs. :cite:`huang2023bridging,teomy2018confluent,wang2026divergence`.

.. It pulls data from the `Open Food Facts database <https://world.openfoodfacts.org/>`_
.. and offers a *simple* and *intuitive* API.

Check out the :doc:`usage`, :doc:`examples`, and :doc:`api/index` sections for further information, including
how to :ref:`install <installation>` the package, usage examples, and the complete API reference.

.. |GitHub| replace:: **GitHub**
.. _GitHub: https://github.com/wwang721/pyafv

.. note::

   This project is under active development, see |GitHub|_ for the latest updates.


.. image:: ../assets/pbc.png
   :width: 300px
   :align: right


Contents
--------

.. toctree::
   :maxdepth: 2
   :numbered:

   Home <self>
   usage
   examples
   calibration
   performance
   contributing
   api/index


.. warning::

   This is an early release of the software. Features and APIs may change in future versions.


References
----------

.. bibliography:: ./main.bib
   :style: unsrt

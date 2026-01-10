Contributing to PyAFV
=======================

First off, **THANK YOU** for considering contributing to **PyAFV**! We welcome contributions from the community.

Before working on a feature or major change, please raise an `issue <https://github.com/wwang721/pyafv/issues>`_ and/or get in touch with the developers. They may have insights on how to implement the feature or useful advice to save you time.

.. note::

    Much of this guide is based on the `pyqmc CONTRIBUTING.md <https://github.com/WagnerGroup/pyqmc/blob/master/CONTRIBUTING.md>`_, which itself draws from `this excellent guide <https://gist.github.com/Chaser324/ce0505fbed06b947d962>`_.


How to contribute
-----------------

Create a fork
^^^^^^^^^^^^^^^^

Click the "Fork" button on the PyAFV GitHub page:
https://github.com/wwang721/pyafv

Then clone **your fork** to your local machine and enter the repository directory

.. code-block:: bash

    cd pyafv


Set up your development environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PyAFV uses `uv <https://docs.astral.sh/uv/>`_ for package management:


After cloning, install it in editable mode and synchronize dependencies

.. code-block:: bash

    uv sync

This installs the core package dependencies along with ``pytest`` required for development and testing.

.. note::

    - If you modify the Cython source file ``./pyafv/cell_geom.pyx``, reinstall the package

        .. code-block:: bash
            
            uv sync --reinstall-package pyafv --inexact

    - For Windows MinGW GCC users, add a ``setup.cfg`` file at the repository root

        .. code-block:: ini

            # setup.cfg
            [build_ext]
            compiler=mingw32

    - See more *notes* for **local development** in the PyAFV's GitHub `README <https://github.com/wwang721/pyafv/blob/main/README.md#local-development>`_.
    


Create a feature branch
^^^^^^^^^^^^^^^^^^^^^^^^^^

Always branch from ``main``, not from another feature branch

.. code-block:: bash

    git checkout main
    git checkout -b your-feature-name


Keeping your fork up to date
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Add the upstream repository as a remote (do this once):

.. code-block:: bash

    git remote add upstream https://github.com/wwang721/pyafv.git
    git remote -v

To sync with upstream (do this regularly):

.. code-block:: bash

    git fetch upstream
    git checkout main
    git merge upstream/main

If needed, update your feature branch with the latest changes:

.. code-block:: bash

    git checkout your-feature-name
    git rebase main

.. note::
    We use ``rebase`` to keep the commit history clean.


Coding standards
----------------

1. **Single responsibility:** Keep functions small and focused on one task. Each function should do one thing well.

2. **Avoid Python loops:** Use *numpy* vectorized operations to avoid Python's performance overhead. Operate on batches of data rather than looping. Performance-critical code may be accelerated using *Cython*.

3. **Minimize dependencies**: Avoid adding new libraries unless absolutely necessary. If required, discuss with maintainers first.

4. **Code style**: Follow `PEP 8 <https://peps.python.org/pep-0008/>`_ style guidelines for Python code..


Documentation requirements
--------------------------

1. **Type annotations:** Use type hints for function arguments and return values.

2. **Array dimensions:** Add comments indicating dimensions for multidimensional arrays:

.. code-block:: python
    
    positions = np.zeros((100, 2))  # N x dimension

3. **Docstrings:** Each function should have a docstring following `PEP 257 <https://peps.python.org/pep-0257/>`_ and written in either `Google style <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html#example-google>`_ (currently used) or `Numpy style <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html>`_ so that it can be parsed by *Sphinx* via ``sphinx.ext.napoleon``. The docstring should explain:

   - Purpose of the function
   - All input parameters
   - Return values
   - Any exceptions raised


Writing tests
-------------

.. image:: https://github.com/wwang721/pyafv/actions/workflows/tests_all_platform.yml/badge.svg
   :target: https://github.com/wwang721/pyafv/actions/workflows/tests_all_platform.yml
   :alt: Tests on all platforms

.. image:: https://github.com/wwang721/pyafv/actions/workflows/tests.yml/badge.svg
   :target: https://github.com/wwang721/pyafv/actions/workflows/tests.yml
   :alt: pytest

Tests are located in the ``tests/`` directory. Run the test suite with

.. code-block:: bash

    uv run pytest

For coverage reports:

.. code-block:: bash

    uv run pytest --cov


Testing strategies (in order of preference)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **Exact solutions:** Compare numerical results to exact analytical solutions.

2. **Independent implementations:** Compare results from two independent numerical methods.

3. **Regression tests:** Ensure the function runs and produces consistent results with pre-computed references.

4. **Sanity checks:** Verify that results make physical sense (e.g., energies decrease after optimization).


Submitting a pull request
-------------------------

Before submitting, ensure you have completed this checklist:

    - All new functions are documented with docstrings and type annotations
    - Tests are written for the new feature or bug fix
    - All tests pass: ``uv run pytest``
    - Code follows the coding standards above
    - Relevant documentation is updated (README, examples, etc.)
    - The branch is up to date with ``main``


Pull request process
^^^^^^^^^^^^^^^^^^^^

1. Push your feature branch to your fork:

.. code-block:: bash

    git push origin your-feature-name

2. Go to the `PyAFV repository <https://github.com/wwang721/pyafv>`_ and click "Pull Request".

3. In your pull request description:

   - Clearly describe the new feature or bug fix
   - Reference any related issues (e.g., "Fixes #123")
   - For bug fixes, provide a minimal example demonstrating the bug and how the fix resolves it
   - For new features, explain the use case and provide example usage

4. Be responsive to feedback from reviewers and be prepared to make changes.


Reporting issues
----------------

When reporting bugs or requesting features:

1. **Search existing issues** to avoid duplicates
2. **Use a clear title** that describes the problem
3. **Provide details:**

   - For bugs: steps to reproduce, expected vs. actual behavior, error messages, environment details
   - For features: use case, proposed implementation (if any)

4. **Include code examples** when relevant (minimal reproducible examples are best)


Code review process
-------------------

All submissions require review before merging. Reviewers will check:

- Code quality and adherence to coding standards
- Test coverage and quality
- Documentation completeness
- Performance implications
- Compatibility with existing code


Questions?
----------

If you have questions about contributing, feel free to:

- Open an `issue <https://github.com/wwang721/pyafv/issues>`_ on GitHub
- Start a discussion in `GitHub Discussion <https://github.com/wwang721/pyafv/discussions>`_
- Contact the maintainer via email: ww000721@gmail.com

Thank you for helping make PyAFV better!

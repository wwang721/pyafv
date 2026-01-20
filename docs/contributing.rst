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

Click the "Fork" button on the **PyAFV** *GitHub* page:
https://github.com/wwang721/pyafv

Then clone **your fork** to your local machine and enter the repository directory

.. code-block:: console

   (.venv) $ cd pyafv


Set up your development environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. |uv| replace:: **uv**
.. _uv: https://docs.astral.sh/uv/

**PyAFV** uses |uv|_ for Python package management---a single tool to replace `pip` (⚡️10-100x faster), `venv`, and even `conda`.

.. tip::
   If you'd like to use your own Python, ensure the ``which python`` version meets the requirement so **uv** doesn't automatically download a different interpreter; otherwise, I recommend letting **uv** manage everything, including the Python interpreter.

After cloning, install **PyAFV** in editable mode and synchronize dependencies:

.. code-block:: console

   (.venv) $ uv sync

This installs the core package dependencies along with **pytest** required for development and testing.

.. note::

    - You can install additional packages as needed using ``uv add <package_name>``.
    - In some environments (like HPC clusters), global Python path can contaminate the project environment. You may need to add the ``PYTHONPATH=""`` prefix to all ``uv`` commands to isolate the project.
    - The current version uses **Cython** to translate ``.pyx`` files into ``.cpp``, (and therefore requires a working C/C++ compiler), though a fallback backend (based on early pure-Python release) is also implemented.

    - For *Windows* **MinGW GCC** users (rather than **MSVC**), add a ``setup.cfg`` file at the repository root

        .. code-block:: ini

            # setup.cfg
            [build_ext]
            compiler=mingw32

      This is equivalent to pass the ``--compiler=mingw32`` flag when invoking build commands such as ``python setup.py build_ext --inplace``.
      To avoid accidentally committing this *ad hoc* file, do not modify ``.gitignore``; instead, add it to local ``.git/info/exclude`` in the repository, which functions like ``.gitignore``.


Create a feature branch and start development
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Always branch from ``main``, not from another feature branch

.. code-block:: console

   (.venv) $ git checkout main
   (.venv) $ git checkout -b your-feature-name

You may then begin editing the codebase and developing new features.

.. note::

    If you modify any ``*.pyx`` Cython source files, you must reinstall the package to ensure the changes take effect: ``uv sync --reinstall-package pyafv --inexact`` (the ``--inexact`` flag prevents **uv** from removing any installed packages).

    - If the compiled C/C++ extension is accidentally removed or corrupted (you will see a **RuntimeWarning** about falling back to the pure-Python implementation), you can also reinstall the package.
    - For the legacy pure-Python implementation with no C/C++ compiled dependencies, see `v0.1.0 <https://github.com/wwang721/pyafv/releases/tag/v0.1.0>`_ (also on `GitLab <https://gitlab.com/wwang721/py-afv/-/releases/v0.1.0>`_). Starting from **PyAFV** v0.3.4, the pure-Python backend can be selected by passing ``backend="python"`` when creating the simulator instance.


Keeping your fork up to date
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Add the upstream repository as a remote (do this once):

.. code-block:: console

   (.venv) $ git remote add upstream https://github.com/wwang721/pyafv.git
   (.venv) $ git remote -v

To sync with upstream (do this regularly):

.. code-block:: console

   (.venv) $ git fetch upstream
   (.venv) $ git checkout main
   (.venv) $ git merge upstream/main

If needed, update your feature branch with the latest changes:

.. code-block:: console

   (.venv) $ git checkout your-feature-name
   (.venv) $ git rebase main

.. note::
    We use ``rebase`` to keep the commit history clean.


Coding standards
----------------

1. **Single responsibility:** Keep functions small and focused on one task. Each function should do one thing well.

2. **Avoid Python loops:** Use *numpy* vectorized operations to avoid Python's performance overhead. Operate on batches of data rather than looping. Performance-critical code may be accelerated using *Cython*.

3. **Minimize dependencies**: Avoid adding new libraries unless absolutely necessary. If required, discuss with maintainers first.

4. **Code style**: Follow `PEP 8 <https://peps.python.org/pep-0008/>`_ style guidelines for Python code.


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

.. image:: https://codecov.io/github/wwang721/pyafv/branch/main/graph/badge.svg?token=VSXSOX8HVS
   :target: https://codecov.io/github/wwang721/pyafv/tree/main
   :alt: Codecov

Tests are located in the ``tests/`` directory. Run the test suite with

.. code-block:: console

   (.venv) $ uv run pytest

For coverage reports:

.. code-block:: console

   (.venv) $ uv run pytest --cov

Current CI status of the test suite, run via **GitHub Actions** on Python 3.12 (with additional test jobs covering all supported platforms and Python versions), is shown in the badges above.

.. note::

   - A comparison against the **MATLAB** implementation from Ref. :cite:`huang2023bridging` is included in current test suite.
   - Unlike `v0.1.0 <https://github.com/wwang721/pyafv/releases/tag/v0.1.0>`_, the current test suite is designed to raise errors if the Cython-compiled C/C++ backend is not available, even though a pure-Python fallback implementation is provided and tested.


Testing strategies (in order of preference)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **Exact solutions:** Compare numerical results to exact analytical solutions.

2. **Independent implementations:** Compare results from two independent numerical methods.

3. **Regression tests:** Ensure the function runs and produces consistent results with pre-computed references.

4. **Sanity checks:** Verify that results make physical sense (e.g., energies decrease after optimization).


Benchmarking
^^^^^^^^^^^^^^^

There is also an implementation of small benchmarks in ``tests/test_benchmarks.py`` comparing the Cython and pure-Python backends using **pytest-benchmark**. To run them:

.. code-block:: console

   (.venv) $ uv run pytest --benchmark-only --benchmark-warmup on --benchmark-histogram

This will display the benchmark results and generate an SVG histogram file in the current directory.
You should write benchmarks for any new performance-critical code you add.


Featured examples
--------------------

To run current example scripts and notebooks in ``examples/``, install all optional dependencies (e.g., **tqdm**, **jupyter**) via ``uv sync --extra examples`` or ``uv sync --all-extras`` (add the ``--inexact`` flag if needed).
Then you can simply run the scripts with

.. code-block:: console
   
   (.venv) $ uv run <script_name>.py


- For developers to launch Jupyter Notebook: after ``uv`` has synced all extra dependencies, start Jupyter with ``uv run jupyter notebook``. Do not use your system-level Jupyter, as the Python kernel of the current ``uv`` environment is not registered there.

.. |Git LFS| replace:: **Git LFS**
.. _Git LFS: https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage

.. note::

   Jupyter notebooks and media are stored via |Git LFS|_. If you clone the repository without **Git LFS** installed, these files will appear as small text pointers. You can either install **Git LFS** to fetch them automatically or download the files manually (e.g., download the repository as a ZIP archive) from the **GitHub** web interface.


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

.. code-block:: console

   (.venv) $ git push origin your-feature-name

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

Thank you for helping make **PyAFV** better!

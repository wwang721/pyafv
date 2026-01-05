[![PyPi](https://img.shields.io/pypi/v/pyafv?color=brightgreen)](https://pypi.org/project/pyafv/)
[![DOI](https://zenodo.org/badge/1124385738.svg)](https://doi.org/10.5281/zenodo.18091659)
<!--[![pytest](https://github.com/wwang721/pyafv/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/wwang721/pyafv/actions/workflows/tests.yml?query=branch:main)-->
[![pytest](https://github.com/wwang721/pyafv/actions/workflows/tests.yml/badge.svg)](https://github.com/wwang721/pyafv/actions/workflows/tests.yml)
[![codecov](https://codecov.io/github/wwang721/pyafv/branch/main/graph/badge.svg?token=VSXSOX8HVS)](https://codecov.io/github/wwang721/pyafv/tree/main)


# PyAFV

Python code that implements the **active-finite-Voronoi (AFV) model**.
The AFV framework was introduced and developed in, for example, Refs. [[1](#huang2023bridging)&ndash;[3](#wang2026divergence)].


## Installation

`PyAFV` is now available on [**PyPI**](https://pypi.org/project/pyafv/), so you should be able to install it through `pip`:
```bash
pip install pyafv
```
Developed using Python 3.11.11, so Python 3.11+ is set as the minimum requirement. If you just want to use the package, skip directly to the [Usage](#usage) section.


## Local development

This project uses [`uv`](https://docs.astral.sh/uv/) for Python package management &ndash; a single tool to replace `pip` (⚡️10-100x faster) and `venv`.

> If you'd like to use your own Python, ensure the `which python` version meets the requirement (>=3.11) so `uv` doesn't automatically download a different interpreter; otherwise, I recommend letting `uv` manage everything, including the Python interpreter.

After cloning the repository, Linux/macOS users (Windows users: see [below](#windows-mingw-gcc)) can synchronize the dependencies with
```bash
uv sync
```
or use `uv sync --no-dev` if you only intend to run the core Python code without development dependencies (like `cython` and `pytest`).

**Notes:**
> * You can install additional packages as needed using `uv add <package_name>`.
> * In some environments (like HPC clusters), global Python path can contaminate the project environment. You may need to add the `PYTHONPATH=""` prefix to all `uv` commands to isolate the project.
> * The current version uses **Cython** to translate `.pyx` files into `.cpp`, (and therefore requires a working C/C++ compiler), though [a fallback backend](/pyafv/finite_voronoi_fallback.py) (based on early pure-Python release) is also implemented. If the compiled C++ extension is accidentally removed or corrupted (you will see a **RuntimeWarning**), you can reinstall the package with `uv sync --reinstall-package pyafv --inexact` (the `--inexact` flag prevents uv from removing any installed packages).
> * For the old pure-Python implementation with no C/C++ compiled dependencies, see [v0.1.0](https://github.com/wwang721/pyafv/releases/tag/v0.1.0) (also on [GitLab](https://gitlab.com/wwang721/py-afv/-/releases/v0.1.0)). Alternatively, remove [setup.py](/setup.py) in the root folder before running `uv sync`.


#### Windows MinGW GCC

* If you are using **MinGW GCC** (rather than MSVC) on Windows, add a `setup.cfg` at the repository root
    ```ini
    # setup.cfg
    [build_ext]
    compiler=mingw32
    ```
    It will then work in the same way.
    <!--With this configuration in place, you even no longer need to pass the `--compiler=mingw32` flag when trying to compile with `uv run python setup.py build_ext --inplace`.-->


#### Editing the Cython file

If you modify the Cython source file [pyafv/cell_geom.pyx](/pyafv/cell_geom.pyx), you must regenerate the corresponding `.cpp` file by running
```bash
uv run cython -3 --cplus pyafv/cell_geom.pyx -o pyafv/cell_geom.cpp
```
Afterward, reinstall the package to ensure the changes take effect: `uv sync --reinstall-package pyafv --inexact`.


### Running tests

Current CI status of the test suite, run via [GitHub Actions](/.github/workflows/tests.yml) on Python 3.12, is shown in the badge at the top of this file.

* To run the full test suite locally (located in [`tests`](/tests/)):
    ```bash
    uv run pytest
    ```
    You can also include coverage options such as `--cov` if desired. If you previously use `uv sync --no-dev`, you will need to run `uv sync` again to install the packages in the *dev* dependency group.

**Notes:** 
> * A comparison against the MATLAB implementation from Ref. [[1](#huang2023bridging)] is included in [test_core.py](/tests/test_core.py).
> * Unlike [v0.1.0](https://github.com/wwang721/pyafv/releases/tag/v0.1.0), the current test suite is designed to raise errors if the C++ backend is not available, even though a pure-Python fallback implementation is provided and tested.


## Usage

<!--Using `uv run python`, you should be able to import `pyafv` from anywhere within the repository directory.-->
The following example demonstrates how to construct a finite-Voronoi diagram:
```python
import numpy as np
import pyafv as afv

N = 100                                          # number of cells
pts = np.random.rand(N, 2) * 10                  # initial positions
params = afv.PhysicalParams()                    # use default parameter values
sim = afv.FiniteVoronoiSimulator(pts, params)    # initialize the simulator
sim.plot_2d(show=True)                           # visualize the Voronoi diagram
```
To compute the conservative forces and extract detailed geometric information (e.g., cell areas, vertices, and edges), call:
```python
diag = sim.build()
```
The returned object `diag` is a Python `dict` containing these quantities.


### Featured examples
To run the example scripts and notebooks in [`examples`](/examples), you need to install at least one additional dependency `tqdm`.

For local development using `uv`: in the project root, run `uv add tqdm`. Then you can simply run any script in [`examples`](/examples/) with
```bash
uv run <script_name>.py
```
You can also install all optional dependencies (e.g., `tqdm`, `jupyter`) via `uv sync --extra examples` or `uv sync --all-extras`.

* For developers to launch Jupyter Notebook: after `uv` has synced all extra dependencies, start Jupyter with `uv run jupyter notebook`. Do not use your system-level Jupyter, as the Python kernel of the current `uv` environment is not registered there.

    > Jupyter notebooks and media are stored via [**Git LFS**](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage). If you clone the repository without **Git LFS** installed, these files will appear as small text pointers. You can either install Git LFS to fetch them automatically or download the files manually (or download the repository as a ZIP archive) from the GitHub web interface.


### Simulation previews

Below are representative simulation snapshots generated using the code:

| Model illustration | Periodic boundary conditions[*](/examples/jupyter/periodic_plotting.ipynb) |
|-----------------|-----------------|
| <img src="./assets/model_illustration.png" height="373"> | <img src="./assets/pbc.png" height="385">|

| Initial configuration | After relaxation | Active dynamics enabled |
|-----------------------|-----------------------|-----------------------|
| <img src="./assets/initial_configuration.png" height="300"> | <img src="./assets/relaxed_configuration.png" height="300"> | <img src="./assets/active_FV.png" height="300"> |


## More information

See important [**issues**](https://github.com/wwang721/pyafv/issues?q=is%3Aissue%20state%3Aclosed) for additional context, such as: 
* [QhullError when 3+ points are collinear #1](https://github.com/wwang721/pyafv/issues/1) [Closed - see [comments](https://github.com/wwang721/pyafv/issues/1#issuecomment-3701355742)]
*  [Add customized plotting to examples illustrating access to vertices and edges #5](https://github.com/wwang721/pyafv/issues/5) [Completed in PR [#7](https://github.com/wwang721/pyafv/pull/7)]
* [Time step dependence of intercellular adhesion in simulations #8](https://github.com/wwang721/pyafv/issues/8) [Closed in PR [#9](https://github.com/wwang721/pyafv/pull/9)]


## Zenodo

The releases of this repository are cross-listed on [Zenodo](https://doi.org/10.5281/zenodo.18091659).


## License

This project is licensed under the [MIT License](/LICENSE), which permits free use, modification, and distribution of the code for nearly any purpose.


## References

<table>
  <tr>
    <td id="huang2023bridging" valign="top">[1]</td>
    <td>
      J. Huang, H. Levine, and D. Bi, <em>Bridging the gap between collective motility and epithelial-mesenchymal transitions through the active finite Voronoi model</em>, <a href="https://doi.org/10.1039/D3SM00327B">Soft Matter <strong>19</strong>, 9389 (2023)</a>.
    </td>
  </tr>
  <tr>
    <td id="teomy2018confluent" valign="top">[2]</td>
    <td>
      E. Teomy, D. A. Kessler, and H. Levine, <em>Confluent and nonconfluent phases in a model of cell tissue</em>, <a href="https://doi.org/10.1103/PhysRevE.98.042418">Phys. Rev. E <strong>98</strong>, 042418 (2018)</a>.
    </td>
  </tr>
  <tr>
    <td id="wang2026divergence" valign="top">[3]</td>
    <td>
      W. Wang (汪巍) and B. A. Camley, <em>Divergence of detachment forces in the finite-Voronoi model</em>, manuscript in preparation (2026).
    </td>
  </tr>
</table>

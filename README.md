[![Tests](https://github.com/wwang721/py-afv/actions/workflows/tests.yml/badge.svg?branch=feature%2Fdivergence)](https://github.com/wwang721/py-afv/actions/workflows/tests.yml?query=branch:feature%2Fdivergence)
[![codecov](https://codecov.io/github/wwang721/py-afv/branch/feature%2Fdivergence/graph/badge.svg?token=VSXSOX8HVS)](https://codecov.io/github/wwang721/py-afv/tree/feature%2Fdivergence)
[![DOI](https://zenodo.org/badge/1124385738.svg)](https://doi.org/10.5281/zenodo.18091659)

# py-afv

Python code that implements the **active-finite-Voronoi (AFV) model**.
The AFV framework was introduced and developed in, for example,
 [[Soft Matter **19**, 9389 (2023)](https://doi.org/10.1039/D3SM00327B)] and [[Phys. Rev. E **98**, 042418
(2018)](https://doi.org/10.1103/PhysRevE.98.042418)].


## Installation

This project uses [`uv`](https://docs.astral.sh/uv/) for Python package management &ndash; a single tool to replace `pip` (⚡️10-100x faster) and `venv`.

After cloning the repository, Linux/macOS users (Windows users: see [below](#windows-mingw-gcc)) can synchronize the dependencies with
```bash
uv sync
```
or use `uv sync --no-dev` if you only intend to run the core code without development dependencies (like `pytest` for running tests).

**Notes:**
> * You can install additional packages as needed using `uv add <package_name>`.
> * The current version requires **Cython** (and therefore a working C/C++ compiler), though [a fallback backend](/afv/finite_voronoi_fallback.py) (based on early pure-Python release) is also implemented. If the Cython compiled extension is accidentally removed or corrupted (you will see a **RuntimeWarning**), you can reinstall the package with `uv sync --reinstall-package py-afv --inexact` (the `--inexact` flag prevents uv from removing any installed packages) or recompile the Cython extension with `uv run setup.py build_ext --inplace`.
> * For the old pure-Python implementation with no C/C++ compiled dependencies, see **[v0.1.0](https://github.com/wwang721/py-afv/releases/tag/v0.1.0)**.


#### Windows MinGW GCC

* If you are using **MinGW GCC** (rather than MSVC) on Windows, add a `setup.cfg` at the repository root
    ```ini
    # setup.cfg
    [build_ext]
    compiler=mingw32
    ```
    It will then work in the same way.
    With this configuration in place, you even no longer need to pass the `--compiler=mingw32` flag when trying to compile with `uv run python setup.py build_ext --inplace`.


## Running tests

The current CI status of the test suite, run via [GitHub Actions](/.github/workflows/tests.yml), is shown in the badge at the top of this file.

* To run the full test suite locally (located in [`tests`](/tests/)):
    ```bash
    uv run pytest
    ```
    You can also include coverage options such as `--cov` if desired. If you previously use `uv sync --no-dev`, you will need to run `uv sync` again to install the packages in the *dev* dependency group.

**Notes:** 
> * A comparison against the MATLAB implementation from [[Soft Matter **19**, 9389 (2023)](https://doi.org/10.1039/D3SM00327B)] is included in [test_core.py](/tests/test_core.py).
> * Unlike [v0.1.0](https://github.com/wwang721/py-afv/releases/tag/v0.1.0), the current test suite is designed to raise errors if the Cython backend is not available, even though a pure-Python fallback implementation is provided and tested.


## Usage

Using `uv run python`, you should be able to import `afv` from anywhere within the repository directory.
The following example demonstrates how to construct a finite-Voronoi diagram:
```python
import numpy as np
import afv

N = 100                                      # number of cells
pts = np.random.rand(N, 2) * 10              # initial positions
params = afv.PhysicalParams()                    # use default parameter values
sim = afv.FiniteVoronoiSimulator(pts, params)    # initialize the simulator
sim.plot_2d(show=True)                       # visualize the Voronoi diagram
```
To compute the conservative forces and extract detailed geometric information (e.g., cell areas, vertices, and edges), call:
```python
diag = sim.build()
```
The returned object `diag` is a Python `dict` containing these quantities.

#### More example scripts
To run the example scripts in [`examples`](/examples), you need to install at least one additional dependency, `tqdm`, via `uv add tqdm`. Then you can simply run any script in [`examples`](/examples/) with
```bash
uv run <script_name>.py
```
You can also install all optional dependencies (e.g., `tqdm`, `jupyter`) via `uv sync --extra examples` or `uv sync --all-extras`.

* To launch Jupyter Notebook: after `uv` has synced all extra dependencies, start Jupyter with `uv run jupyter notebook`. Do not use your system-level Jupyter, as the Python kernel of the current `uv` environment is not registered there.


* Below are representative simulation snapshots generated using the code:

| Model illustration | Periodic boundary conditions[*](/examples/jupyter/periodic_plotting.ipynb) |
|-----------------|-----------------|
| <img src="./assets/model_illustration.png" height="373"> | <img src="./assets/pbc.png" height="385">|

| Initial configuration | After relaxation | Active dynamics enabled |
|-----------------------|-----------------------|-----------------------|
| <img src="./assets/initial_configuration.png" height="300"> | <img src="./assets/relaxed_configuration.png" height="300"> | <img src="./assets/active_FV.png" height="300"> |


## More information

See the important [**issues**](https://github.com/wwang721/py-afv/issues?q=is%3Aissue%20state%3Aclosed) for additional context, such as: 
* [QhullError when 3+ points are collinear #1](https://github.com/wwang721/py-afv/issues/1) [Closed]
*  [Add customized plotting to examples illustrating access to vertices and edges #5](https://github.com/wwang721/py-afv/issues/5) [Completed in PR [#7](https://github.com/wwang721/py-afv/pull/7)]
* [Time step dependence of intercellular adhesion in simulations #8](https://github.com/wwang721/py-afv/issues/8) [🚧 [feature/divergence](https://github.com/wwang721/py-afv/tree/feature%2Fdivergence) in progress]

## License

This project is licensed under the [MIT License](/LICENSE), which permits free use, modification, and distribution of the code for nearly any purpose.

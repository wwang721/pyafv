[![PyPi](https://img.shields.io/pypi/v/pyafv?cacheSeconds=300)](https://pypi.org/project/pyafv/)
[![Downloads](https://img.shields.io/pypi/dm/pyafv.svg?cacheSeconds=43200)](https://pypi.org/project/pyafv/)
[![Documentation](https://img.shields.io/badge/documentation-pyafv.readthedocs.io-yellow.svg?logo=readthedocs)](https://pyafv.readthedocs.io)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pyafv/assets/blob/main/jupyter/getting_started.ipynb)
<!--[![pytest](https://github.com/wwang721/pyafv/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/wwang721/pyafv/actions/workflows/tests.yml?query=branch:main)-->
[![Tests on all platforms](https://github.com/wwang721/pyafv/actions/workflows/tests_all_platform.yml/badge.svg)](https://github.com/wwang721/pyafv/actions/workflows/tests_all_platform.yml)
[![pytest](https://github.com/wwang721/pyafv/actions/workflows/tests.yml/badge.svg)](https://github.com/wwang721/pyafv/actions/workflows/tests.yml)
[![Codecov](https://codecov.io/github/wwang721/pyafv/branch/main/graph/badge.svg?token=VSXSOX8HVS)](https://codecov.io/github/wwang721/pyafv/tree/main)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

<!--
[![arXiv:2503.03126](https://img.shields.io/badge/arXiv-2503.03126-grey.svg?colorA=a42c25&colorB=grey&logo=arxiv)](https://doi.org/10.48550/arXiv.2503.03126)
[![PhysRevE.109.054408](https://img.shields.io/badge/Phys.%20Rev.%20E-109.054408-grey.svg?colorA=8c6040)](https://doi.org/10.1103/PhysRevE.109.054408)
[![Soft Matter](https://img.shields.io/badge/Soft%20Matter-XXXXX-63a7c2.svg?colorA=63a7c2&colorB=grey)](https://doi.org/10.1103/PhysRevE.109.054408)
-->


# PyAFV

Python code that implements the **active-finite-Voronoi (AFV) model** in 2D.
The AFV framework was introduced and developed in, for example, Refs. [[1](#huang2023bridging)&ndash;[3](#wang2026divergence)].


## Installation

**PyAFV** is available on **PyPI** and can be installed using *pip* directly:
```bash
pip install pyafv
```
The package supports Python ≥ 3.10 and < 3.15, including Python 3.14t (the free-threaded, no-GIL build).
To verify that the installation was successful and that the correct version is installed, run the following in Python:
```python
import pyafv
print(pyafv.__version__)
```

> On HPC clusters, global Python path can contaminate the runtime environment. You may need to clear it explicitly using `unset PYTHONPATH` or prefixing the *pip* command with `PYTHONPATH=""`.

### Install from source

Installing from source can be necessary if *pip* installation does not work. First, download and unzip the source code, then navigate to the root directory of the package and run:
```bash
pip install .
```

> **Note:** A C/C++ compiler is required if you are building from source, since some components of **PyAFV** are implemented in Cython for performance optimization.


#### Windows MinGW GCC

If you are using **MinGW GCC** (rather than **MSVC**) on *Windows*, to build from the source code, add a `setup.cfg` at the repository root before running `pip install .` with the following content:
```ini
# setup.cfg
[build_ext]
compiler=mingw32
```


### Install offline

If you need to install **PyAFV** on a machine without internet access, you can download the corresponding wheel file from **PyPI** and transfer it to the target machine, and then run the following command to install using *pip*:
```bash
pip install pyafv-<version>-<platform>.whl
```
Alternatively, you can build **PyAFV** from source as described in the previous section. In this case, in addition to the required prerequisites of the package, the build-time dependencies **hatchling** and **hatch-cython** must also be available.


## Usage

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pyafv/assets/blob/main/jupyter/getting_started.ipynb)

Here is a simple example to get you started, demonstrating how to construct a finite-Voronoi diagram (click the **Google Colab** badge above to run the notebook directly):
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


## Simulation previews

Below are representative simulation snapshots generated using the code:

| Model illustration | Periodic boundary conditions |
|-----------------|-----------------|
| <img src="https://media.githubusercontent.com/media/wwang721/pyafv/main/assets/model_illustration.png" width="540"> | <img src="https://media.githubusercontent.com/media/wwang721/pyafv/main/assets/pbc.png" width="385">|

| Initial configuration | After relaxation | Active dynamics enabled |
|-----------------------|-----------------------|-----------------------|
| <img src="https://media.githubusercontent.com/media/wwang721/pyafv/main/assets/initial_configuration.png" width="300"> | <img src="https://media.githubusercontent.com/media/wwang721/pyafv/main/assets/relaxed_configuration.png" width="300"> | <img src="https://media.githubusercontent.com/media/wwang721/pyafv/main/assets/active_FV.png" width="300"> |


## More information

- **Full documentation** on [readthedocs](https://pyafv.readthedocs.io) or as [a single PDF file](https://pyafv.readthedocs.io/_/downloads/en/latest/pdf/).

- See [CONTRIBUTING.md](https://github.com/wwang721/pyafv/blob/main/.github/CONTRIBUTING.md) or the [documentation](https://pyafv.readthedocs.io/latest/contributing.html) for **local development instructions**.

- See some important [**issues**](https://github.com/wwang721/pyafv/issues?q=is%3Aissue%20state%3Aclosed) for additional context, such as: 
    * [QhullError when 3+ points are collinear #1](https://github.com/wwang721/pyafv/issues/1) [Closed - see [comments](https://github.com/wwang721/pyafv/issues/1#issuecomment-3701355742)]
    *  [Add customized plotting to examples illustrating access to vertices and edges #5](https://github.com/wwang721/pyafv/issues/5) [Completed in PR [#7](https://github.com/wwang721/pyafv/pull/7)]
    * [Time step dependence of intercellular adhesion in simulations #8](https://github.com/wwang721/pyafv/issues/8) [Closed in PR [#9](https://github.com/wwang721/pyafv/pull/9)]

- Some releases of this repository are cross-listed on [Zenodo](https://doi.org/10.5281/zenodo.18091659):

  [![Zenodo](https://zenodo.org/badge/1124385738.svg)](https://doi.org/10.5281/zenodo.18091659)


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

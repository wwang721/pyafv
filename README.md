# PyAFV

<a href="https://pyafv.github.io"><img src="https://raw.githubusercontent.com/pyafv/assets/main/gif/test.gif" alt="pyafv_pbc" align="right" /></a>

[![PyPi](https://img.shields.io/pypi/v/pyafv?color=brightgreen&cacheSeconds=300)](https://pypi.org/project/pyafv/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/pyafv.svg)](https://anaconda.org/conda-forge/pyafv)
[![Docker](https://img.shields.io/docker/v/wwang721/pyafv?logo=docker&sort=semver&color=blue&label=docker)](https://hub.docker.com/r/wwang721/pyafv)
[![License](https://img.shields.io/github/license/wwang721/pyafv)](/LICENSE)
[![Tests on all platforms](https://github.com/wwang721/pyafv/actions/workflows/tests_all_platform.yml/badge.svg)](https://github.com/wwang721/pyafv/actions/workflows/tests_all_platform.yml)
[![pytest](https://github.com/wwang721/pyafv/actions/workflows/tests.yml/badge.svg)](https://github.com/wwang721/pyafv/actions/workflows/tests.yml)
[![Codecov](https://codecov.io/github/wwang721/pyafv/branch/main/graph/badge.svg?token=VSXSOX8HVS)](https://codecov.io/github/wwang721/pyafv/tree/main)
[![Documentation](https://app.readthedocs.org/projects/pyafv/badge/?version=latest)](https://pyafv.readthedocs.io)
[![arXiv:2604.15481](https://img.shields.io/badge/arXiv-2604.15481-grey.svg?colorB=a42c25&logo=arxiv)](https://doi.org/10.48550/arXiv.2604.15481)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pyafv/assets/blob/main/jupyter/index.ipynb)

<!--
[![Downloads](https://img.shields.io/pypi/dm/pyafv.svg?cacheSeconds=43200)](https://pypi.org/project/pyafv/)
[![PhysRevE.109.054408](https://img.shields.io/badge/Phys.%20Rev.%20E-109.054408-grey.svg?colorA=8c6040)](https://doi.org/10.1103/PhysRevE.109.054408)
[![Soft Matter](https://img.shields.io/badge/Soft%20Matter-XXXXX-63a7c2.svg?colorA=63a7c2&colorB=grey)](https://doi.org/10.1103/PhysRevE.109.054408)
-->

**PyAFV** is a Python package for simulating cellular tissues based on the 2D **active finite Voronoi (AFV) model**.
It provides a computational framework for investigating collective cell behaviors such as motility, adhesion, jamming, and tissue fracture in active matter and biophysical systems.
In contrast to standard vertex or Voronoi models, the AFV model incorporates finite interaction ranges and cell-medium interfaces, allowing for detachment, free boundaries, and fragmentation.
The package includes tools for geometry handling, time evolution, and analysis of cell configurations.
The AFV formalism was introduced and further developed in, for example, Refs. [[1](#huang2023bridging)&ndash;[3](#wang2026divergence)].


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

<!--
> On some HPC clusters, global Python path can contaminate the runtime environment. You may need to clear it explicitly using `unset PYTHONPATH` or prefixing the *pip* command with `PYTHONPATH=""`.
-->

As an alternative, you can install **PyAFV** via *conda* from the **conda-forge** channel:
```bash
conda install -c conda-forge pyafv
```
If you go this route, note that for Python 3.14 the **conda-forge** distribution currently provides only the GIL-enabled build.


<!--
### Install from source

Installing from source can be necessary if *pip* installation does not work. First, download and unzip the source code, then navigate to the root directory of the package and run:
```bash
pip install .
```

> **Note:** A C/C++ compiler is required if you are building from source, since some components of **PyAFV** are implemented in Cython for performance optimization.


#### Windows MinGW GCC

If you are using **MinGW GCC** (rather than **MSVC**) on *Windows*, to build from the source code, add a `setup.cfg` file at the repository root before running `pip install .` with the following content:
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
-->

> See the [documentation](https://pyafv.readthedocs.io/latest/usage.html#install-from-source) for instructions on installing the package from source or in offline environments.


### Install using Docker 🐳

**PyAFV** can also be installed via containerized environments. Pull the Docker image from **Docker Hub**:
```bash
docker pull wwang721/pyafv:latest
```
The image is also available via the **GitHub Container Registry (GHCR)** under [**GitHub Packages**](https://github.com/wwang721/pyafv/pkgs/container/pyafv); use `ghcr.io/wwang721/pyafv` to pull from GHCR instead.
<!--
Then run Python scripts with `pyafv` using:
```bash
docker run --rm -v $(pwd):/app wwang721/pyafv python <script_name>.py
```
Use `${PWD}` on Windows PowerShell instead of `$(pwd)`.
-->


## Usage

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pyafv/assets/blob/main/jupyter/getting_started.ipynb)

Here is a simple example to get you started, demonstrating how to construct a finite Voronoi diagram (click the **Google Colab** badge above to run the notebook directly):
```python
import numpy as np
import pyafv as afv

N = 100                                          # number of cells
pts = np.random.rand(N, 2) * 10                  # initial positions
params = afv.PhysicalParams(r=1.0)               # use default parameter values
sim = afv.FiniteVoronoiSimulator(pts, params)    # initialize the simulator
sim.plot_2d(show=True)                           # visualize the Voronoi diagram
```
To compute the conservative forces and extract detailed geometric information (e.g., cell areas, vertices, and edges), call:
```python
diag = sim.build()
```
The returned object `diag` is a Python `dict` containing these quantities. Refer to the [documentation](https://pyafv.readthedocs.io/latest/usage.html#a-simple-example) for more detailed usage.

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
    * [Handle degenerate 4-fold Voronoi vertices causing unpacking errors in _assemble_forces #38](https://github.com/wwang721/pyafv/issues/38) [Closed in PR [#42](https://github.com/wwang721/pyafv/pull/42)]

- Some releases of this repository are cross-listed on [Zenodo](https://doi.org/10.5281/zenodo.18091659):

  [![Zenodo](https://zenodo.org/badge/1124385738.svg)](https://doi.org/10.5281/zenodo.18091659)


## Citing the package

To cite **PyAFV**, use the following BibTeX entry:
```bibtex
@article{wang2026divergence,
  title   = {{Divergence of detachment forces in the finite Voronoi model}},
  author  = {Wang, Wei and Camley, Brian A},
  journal = {arXiv preprint arXiv:2604.15481},
  year    = {2026},
  doi     = {10.48550/arXiv.2604.15481}
}
```


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
      W. Wang (汪巍) and B. A. Camley, <em>Divergence of detachment forces in the finite Voronoi model</em> <a href="https://doi.org/10.48550/arXiv.2604.15481">arXiv:2604.15481 [cond-mat.soft] (2026)</a>.
    </td>
  </tr>
</table>

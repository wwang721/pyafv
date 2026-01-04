[![PyPi](https://img.shields.io/pypi/v/pyafv?color=brightgreen)](https://pypi.org/project/pyafv/)
[![DOI](https://zenodo.org/badge/1124385738.svg)](https://doi.org/10.5281/zenodo.18091659)
<!--[![pytest](https://github.com/wwang721/pyafv/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/wwang721/pyafv/actions/workflows/tests.yml?query=branch:main)-->
[![pytest](https://github.com/wwang721/pyafv/actions/workflows/tests.yml/badge.svg)](https://github.com/wwang721/pyafv/actions/workflows/tests.yml)
[![codecov](https://codecov.io/github/wwang721/pyafv/branch/main/graph/badge.svg?token=VSXSOX8HVS)](https://codecov.io/github/wwang721/pyafv/tree/main)


Python code that implements the **active-finite-Voronoi (AFV) model**.
The AFV framework was introduced and developed in, for example, Refs. [[1](#huang2023bridging)&ndash;[3](#wang2026divergence)].


## Installation

`PyAFV` is now available on [**PyPI**](https://pypi.org/project/pyafv/), so you should be able to install it through `pip`:
```bash
pip install pyafv
```
Python@3.11 is set as the minimum requirement.


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


## More information

GitHub: [https://github.com/wwang721/pyafv](https://github.com/wwang721/pyafv)

See important [**issues**](https://github.com/wwang721/pyafv/issues?q=is%3Aissue%20state%3Aclosed) for additional context, such as: 
* [QhullError when 3+ points are collinear #1](https://github.com/wwang721/pyafv/issues/1) [Closed - see [comments](https://github.com/wwang721/pyafv/issues/1#issuecomment-3701355742)]
*  [Add customized plotting to examples illustrating access to vertices and edges #5](https://github.com/wwang721/pyafv/issues/5) [Completed in PR [#7](https://github.com/wwang721/pyafv/pull/7)]
* [Time step dependence of intercellular adhesion in simulations #8](https://github.com/wwang721/pyafv/issues/8) [Closed in PR [#9](https://github.com/wwang721/pyafv/pull/9)]


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

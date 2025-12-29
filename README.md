[![Tests](https://github.com/wwang721/py-afv/actions/workflows/tests.yml/badge.svg)](https://github.com/wwang721/py-afv/actions/workflows/tests.yml)
[![codecov](https://codecov.io/github/wwang721/py-afv/graph/badge.svg?token=VSXSOX8HVS)](https://codecov.io/github/wwang721/py-afv)

# py-afv

Python code that implements the **active-finite-Voronoi (AFV) model**.
The AFV framework was introduced and developed in, for example,
 [[Soft Matter **19**, 9389 (2023)](https://doi.org/10.1039/D3SM00327B)] and [[Phys. Rev. E **98**, 042418
(2018)](https://doi.org/10.1103/PhysRevE.98.042418)].


## Installation

This project uses [`uv`](https://docs.astral.sh/uv/) for Python package management.

After cloning the repository, synchronize the dependencies with
```bash
uv sync --dev
```
or simply `uv sync` if you only intend to run the core code without development dependencies.

> Note: `tqdm` is included in the *dev* group. Some scripts in [`examples`](/examples/) rely on `tqdm`; if you did not sync the development dependency group, you may need to add it manually via `uv add tqdm`.


## Running tests

The current CI status of the test suite, run via [GitHub Actions](/.github/workflows/tests.yml), is shown in the badge at the top of this file.

To run the full test suite locally (located in [`test`](/tests/)):
```bash
uv run pytest
```
You can also include coverage options such as `--cov` if desired.

> Note: A comparison against the MATLAB implementation from [[Soft Matter **19**, 9389 (2023)](https://doi.org/10.1039/D3SM00327B)] is included in [test_core.py](/tests/test_core.py).


## Usage

Example scripts are provided in [`examples`](/examples/). They can be run using
```bash
uv run <script_name>.py
```

Below are representative simulation snapshots generated using the AFV model:
| Initial configuration |
|-----------------------|
| <img src="./assets/initial_configuration.png" width="600"> |

| After relaxation |
|------------------|
| <img src="./assets/relaxed_configuration.png" width="600"> |

| Active dynamics enabled |
|-----------------|
| <img src="./assets/active_FV.png" width="600"> |

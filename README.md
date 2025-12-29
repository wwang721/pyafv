[![Tests](https://github.com/wwang721/py-afv/actions/workflows/tests.yml/badge.svg)](https://github.com/wwang721/py-afv/actions/workflows/tests.yml)
[![codecov](https://codecov.io/github/wwang721/py-afv/graph/badge.svg?token=VSXSOX8HVS)](https://codecov.io/github/wwang721/py-afv)

# py-afv

Python code that implements the `active-finite-Voronoi (AFV) model`.


## Installation

I am usingg [`uv`](https://docs.astral.sh/uv/) to manage the Python packages.

After clone this repo, first sync the dependencies with `uv`:
``` bash
uv sync --dev
```
or just `uv sync` if you only need to use the code.

> I put `tqdm` in the "dev" group, so for some scripts in [`examples`](/examples/), you may need to `uv add tqdm` manually if you did sync my development dependency group.

## Run tests

Status of CI using GitHub Actions should be shown on the top badge. If you want to run all tests in [`test`](/tests/) by yourself:
``` bash
uv run pytest
```
or with coverage flags like `--cov`.


## Usage

See examples in [`examples`](/examples/).

Some simulation snapshots:
* ![initial_config](/assets/initial_configuration.png)
* ![after_relax](/assets/relaxed_configuration.png)
* ![active_dynamics](/assets/active_FV.png)

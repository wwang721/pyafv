# Contributing to PyAFV

> *This contributing guide was drafted by **GitHub Copilot** and approved by the maintainer.*

First off, **THANK YOU** for considering contributing to **PyAFV**! We welcome contributions from the community.

Before working on a feature or major change, please raise an [issue](https://github.com/wwang721/pyafv/issues) and/or get in touch with the developers. They may have insights on how to implement the feature or useful advice to save you time.

Much of this guide is based on the [pyqmc CONTRIBUTING.md](https://github.com/WagnerGroup/pyqmc/blob/master/CONTRIBUTING.md), which itself draws from [this excellent guide](https://gist.github.com/Chaser324/ce0505fbed06b947d962).

## How to contribute

### 1. Create a fork

Click the "Fork" button on the [PyAFV GitHub page](https://github.com/wwang721/pyafv). Then clone **your fork** to your local machine,

```bash
cd pyafv
```

### 2. Set up your development environment

**PyAFV** uses [`uv`](https://docs.astral.sh/uv/) for Python package management &ndash; a single tool to replace `pip` (⚡️10-100x faster), `venv`, and even `conda`.

> [!TIP]
> If you'd like to use your own Python, ensure the `which python` version meets the requirement so `uv` doesn't automatically download a different interpreter; otherwise, I recommend letting `uv` manage everything, including the Python interpreter.

After cloning, install **PyAFV** in the **editable** mode and synchronize dependencies:
```bash
uv sync
```
This installs the core package dependencies along with `pytest` required for development and testing.

> [!NOTE]
> - You can install additional packages as needed using `uv add <package_name>`.
> - You can install additional packages as needed using `uv add <package_name>`.
> - In some environments (like HPC clusters), global Python path can contaminate the project environment. You may need to add the `PYTHONPATH=""` prefix to all `uv` commands to isolate the project.
> - The current version uses **Cython** to translate `.pyx` files into `.cpp`, (and therefore requires a working C/C++ compiler), though [a fallback backend](/pyafv/cell_geom_fallback.py) (based on early pure-Python release) is also implemented.
> - For *Windows* **MinGW GCC** users (rather than **MSVC**), add a `setup.cfg` file at the repository root:
>   ```ini
>   # setup.cfg
>   [build_ext]
>   compiler=mingw32
>   ```
>   This is equivalent to pass the `--compiler=mingw32` flag when invoking build commands such as `python setup.py build_ext --inplace`.
>   To avoid accidentally committing this *ad hoc* file, do not modify `.gitignore`; instead, add it to local `.git/info/exclude` in the repository, which functions like `.gitignore`.


### 3. Create a feature branch and start development

Always branch from `main`, not from another feature branch:
```bash
git checkout main
git checkout -b your-feature-name
```
You may then begin editing the codebase and developing new features.


* If you modify any `*.pyx` Cython source files, you must reinstall the package to ensure the changes take effect: `uv sync --reinstall-package pyafv --inexact` (the `--inexact` flag prevents **uv** from removing any installed packages).

    - If the compiled C/C++ extension is accidentally removed or corrupted (you will see a **RuntimeWarning** about falling back to the pure-Python implementation), you can also reinstall the package.
    - For the legacy pure-Python implementation with no C/C++ compiled dependencies, see [v0.1.0](https://github.com/wwang721/pyafv/releases/tag/v0.1.0) (also on [GitLab](https://gitlab.com/wwang721/py-afv/-/releases/v0.1.0)). Starting from **PyAFV** v0.3.4, the pure-Python backend can be selected by passing `backend="python"` when creating the simulator instance.


### 4. Keeping your fork up to date

Add the upstream repository as a remote (do this once):

```bash
git remote add upstream https://github.com/wwang721/pyafv.git

# Verify
git remote -v
```

To sync with upstream (do this regularly):

```bash
git fetch upstream
git checkout main
git merge upstream/main
```

If needed, update your feature branch with the latest changes:

```bash
git checkout your-feature-name
git rebase main
```

Note: We use `rebase` to keep the commit history clean.

## Coding standards

<!--1. **Functional programming**: Prefer functional styles when possible. Create new classes only when absolutely necessary.-->
1. **Single responsibility**: Keep functions small and focused on one task. Each function should do one thing well.
2. **Avoid Python loops**: Use `numpy` vectorized operations to avoid Python's performance overhead. Operate on batches of data rather than looping. You can also accelerate by compiling your code to C/C++ using Cython.
3. **Minimize dependencies**: Avoid adding new libraries unless absolutely necessary. If required, discuss with maintainers first.
4. **Code style**: Follow [PEP 8](https://peps.python.org/pep-0008/) style guidelines for Python code.

## Documentation requirements

1. **Type annotations**: Use type hints for function arguments and return values.
2. **Array dimensions**: Add comments indicating dimensions for multidimensional arrays:
   ```python
   positions = np.zeros((100, 2))  # N x dimension
   ```
3. **Docstrings**: Each function should have a docstring following [PEP 257](https://peps.python.org/pep-0257/) and written in either [**Google style**](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html#example-google) (currently used) or [**Numpy style**](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html) so that it can be parsed by **Sphinx** via ``sphinx.ext.napoleon``. The docstring should explain:

   - Purpose of the function
   - All input parameters
   - Return values
   - Any exceptions raised

## Writing tests

Tests are located in the [`tests/`](tests/) directory. Run the test suite with:

```bash
uv run pytest
```

For coverage reports:

```bash
uv run pytest --cov
```

Current CI status of the test suite, run via [**GitHub Actions**](/.github/workflows/tests.yml) on Python 3.12 (with additional test jobs covering all supported platforms and Python versions), is shown in the badges at the top of [README.md](/README.md).

> [!NOTE]
> * A comparison against the **MATLAB** implementation from [Huang *et al.*, Soft Matter **19**, 9389 (2023)](https://doi.org/10.1039/D3SM00327B) is included in [test_core.py](/tests/test_core.py) and [test_vary_A0.py](/tests/test_vary_A0.py).
> * Unlike [v0.1.0](https://github.com/wwang721/pyafv/releases/tag/v0.1.0), the current test suite is designed to raise errors if the Cython-compiled C/C++ backend is not available, even though a pure-Python fallback implementation is provided and tested.


### Testing strategies (in order of preference)

1. **Exact solutions**: Compare numerical results to exact analytical solutions.
2. **Independent implementations**: Compare results from two independent numerical methods.
3. **Regression tests**: Ensure the function runs and produces consistent results with pre-computed references.
4. **Sanity checks**: Verify that results make physical sense (e.g., energies decrease after optimization).


### Benchmarking

There is also an implementation of small benchmarks in [`tests/test_benchmarks.py`](tests/test_benchmarks.py) comparing the Cython and pure-Python backends using **pytest-benchmark**. To run them:
```bash
uv run pytest --benchmark-only --benchmark-warmup on --benchmark-histogram
```
This will display the benchmark results and generate an SVG histogram file in the current directory (see [here](https://pyafv.readthedocs.io/latest/performance.html#benchmarking-backends)).
You should write benchmarks for any new performance-critical code you add.


## Featured examples

To run current example scripts and notebooks in [`examples`](/examples/), install all optional dependencies (e.g., **tqdm**, **jupyter**) via `uv sync --extra examples` or `uv sync --all-extras` (add the `--inexact` flag if needed).
Then you can simply run the scripts with
```bash
uv run <script_name>.py
```

* For developers to launch Jupyter Notebook: after `uv` has synced all extra dependencies, start Jupyter with `uv run jupyter notebook`. Do not use your system-level Jupyter, as the Python kernel of the current `uv` environment is not registered there.

    > Jupyter notebooks and media are stored via [**Git LFS**](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage). If you clone the repository without **Git LFS** installed, these files will appear as small text pointers. You can either install **Git LFS** to fetch them automatically or download the files manually (e.g., download the repository as a ZIP archive) from the **GitHub** web interface.


## Submitting a pull request

Before submitting, ensure you have completed this checklist:

- [ ] All new functions are documented with docstrings and type annotations
- [ ] You have written tests for the new feature or bug fix
- [ ] All tests pass: `uv run pytest`
- [ ] Your code follows the coding standards above
- [ ] You have updated relevant documentation (README, examples, etc.)
- [ ] Your branch is up to date with `main`

### Pull request process

1. Push your feature branch to your fork:
   ```bash
   git push origin your-feature-name
   ```

2. Go to the [PyAFV repository](https://github.com/wwang721/pyafv) on GitHub and click the "Pull Request" button.

3. In your pull request description:
   - Clearly describe the new feature or bug fix
   - Reference any related issues (e.g., "Fixes #123")
   - For bug fixes, provide an example demonstrating the bug and show how your fix resolves it
   - For new features, explain the use case and provide example usage

4. Be responsive to feedback from reviewers and be prepared to make changes.

## Reporting issues

When reporting bugs or requesting features:

1. **Search existing issues** to avoid duplicates
2. **Use a clear title** that describes the problem
3. **Provide details**:
   - For bugs: steps to reproduce, expected vs. actual behavior, error messages, environment details
   - For features: use case, proposed implementation (if any)
4. **Include code examples** when relevant (minimal reproducible examples are best)

## Code review process

All submissions require review before merging. Reviewers will check:

- Code quality and adherence to coding standards
- Test coverage and quality
- Documentation completeness
- Performance implications
- Compatibility with existing code

## Questions?

If you have questions about contributing, feel free to:

- Open an [issue](https://github.com/wwang721/pyafv/issues) on GitHub
- Start a discussion in [GitHub Discussions](https://github.com/wwang721/pyafv/discussions)
- Contact the maintainer via email: ww000721@gmail.com

Thank you for helping make **PyAFV** better!

# Contributing to PyAFV

> *This contributing guide was drafted by **GitHub Copilot** and approved by the maintainer.*

First off, **THANK YOU** for considering contributing to PyAFV! We welcome contributions from the community.

Before working on a feature or major change, please raise an [issue](https://github.com/wwang721/pyafv/issues) and/or get in touch with the developers. They may have insights on how to implement the feature or useful advice to save you time.

Much of this guide is based on the [pyqmc CONTRIBUTING.md](https://github.com/WagnerGroup/pyqmc/blob/master/CONTRIBUTING.md), which itself draws from [this excellent guide](https://gist.github.com/Chaser324/ce0505fbed06b947d962).

## How to contribute

### 1. Create a fork

Click the "Fork" button on the [PyAFV GitHub page](https://github.com/wwang721/pyafv). Then clone **your fork** to your local machine,

```bash
cd pyafv
```

### 2. Set up your development environment

PyAFV uses [`uv`](https://docs.astral.sh/uv/) for package management. After cloning, install it in "editable" mode and synchronize dependencies:

```bash
uv sync
```

This installs the core package dependencies along with `pytest` required for development and testing.

**Notes:**
- If you modify the Cython source file [`pyafv/cell_geom.pyx`](pyafv/cell_geom.pyx), reinstall the package: `uv sync --reinstall-package pyafv --inexact`.
- For Windows MinGW GCC users, add a `setup.cfg` file at the repository root:
  ```ini
  # setup.cfg
  [build_ext]
  compiler=mingw32
  ```
- See more notes for local development in the [README](/README.md).

### 3. Create a feature branch

Always branch from `main`, not from another feature branch:

```bash
git checkout main
git checkout -b your-feature-name
```

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
3. **Docstrings**: Each function should have a docstring following [PEP 257](https://peps.python.org/pep-0257/) that explains:
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

### Testing strategies (in order of preference)

1. **Exact solutions**: Compare numerical results to exact analytical solutions.
2. **Independent implementations**: Compare results from two independent numerical methods.
3. **Regression tests**: Ensure the function runs and produces consistent results with pre-computed references.
4. **Sanity checks**: Verify that results make physical sense (e.g., energies decrease after optimization).

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

Thank you for helping make PyAFV better!

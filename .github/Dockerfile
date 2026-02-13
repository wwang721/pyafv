# syntax=docker/dockerfile:1

############################
# 1) Builder: compile + wheel
############################
FROM python:3.12-slim AS builder

# System deps needed to compile C/Cython extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

# Install build tooling (PEP 517 build). hatchling is driven by pyproject.toml.
RUN pip install --no-cache-dir build

# Copy only metadata first (better layer caching)
COPY pyproject.toml README.md LICENSE ./
# If your hatch version file is needed to resolve version dynamically at build time:
COPY pyafv/_version.py pyafv/_version.py

# Now copy the rest (including .pyx, package code, etc.)
COPY pyafv/ pyafv/
# If you have a MANIFEST.in or similar, copy it too (optional)
# COPY MANIFEST.in .

# Build wheel
RUN python -m build --wheel

############################
# 2) Runtime: minimal install
############################
FROM python:3.12-slim AS runtime

WORKDIR /app

# Copy built wheel from builder and install
COPY --from=builder /src/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm -rf /tmp/*.whl

# Default command
CMD ["python"]

# Use official Python 3.12 base image
FROM python:3.12-slim

# Avoid writing .pyc files and enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install your package from PyPI
RUN pip install --no-cache-dir pyafv tqdm

# Default command (can be overridden)
CMD ["python"]


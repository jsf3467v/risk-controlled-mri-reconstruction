# syntax=docker/dockerfile:1
#
# CPU image for the eval, calibrate, sweep, test, and demo surface.
# Training stays on the host (MPS or CUDA); a Mac container has no Metal
# passthrough, so it would fall back to CPU. See DOCKER.md.

ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim

# Agg renders matplotlib without a display. PYTHONPATH mirrors the repo's
# pytest.ini so imports resolve outside pytest too.
ENV MPLBACKEND=Agg \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app:/app/src:/app/data

WORKDIR /app

# Torch from the CPU wheel index skips the bundled CUDA libraries and keeps the
# image small. Deps copied before source so their layer caches across edits.
COPY requirements.txt requirements-dev.txt ./
RUN pip install --upgrade pip \
 && pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt \
 && pip install -r requirements-dev.txt

# Copy the whole project (picks up pytest.ini and the tests). .dockerignore
# keeps the .h5 pool, checkpoints, artifacts, and notebooks out.
COPY . .

# Recreate the excluded mount points so a bare `docker run` still starts.
RUN mkdir -p data artifacts checkpoints/reconstructor

# Default runs the device preflight. Override per stage in docker-compose.yml.
CMD ["python", "data/data_processing.py"]

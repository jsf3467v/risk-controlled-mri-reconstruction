# Running in Docker

This image covers the evaluation part of the project. It runs calibration,
the eval report, the risk sweep, and the test suite, and serves as the base
for a demo. The M4Raw `.h5` files and the model checkpoints are mounted at run
time rather than baked in, which keeps the image small and portable.

The build copies the whole project, so it includes `pytest.ini` and the tests.
That `pytest.ini` sets the import paths with `pythonpath = . src data`, and the
native test job and the container both read it, so the two cannot drift apart.
The workflow at `.github/workflows/docker-image.yml` builds and checks the image
independently and leaves `ci.yml` untouched.

## Files at the repository root

```
Dockerfile
.dockerignore
docker-compose.yml
requirements.txt          (already in the repo)
requirements-dev.txt      (adds pytest)
```

The compose file expects the project's normal layout.

```
data/         data_processing.py and the unzipped M4Raw .h5 files
src/          reconstructor.py, gate.py, metrics.py, train.py
checkpoints/  reconstructor/best.pt and gate.json
artifacts/    figures and CSV, written back to the host
```

## The MPS note, worth reading first

`active_device()` prefers Apple's MPS backend, but MPS is never visible inside a
container on a Mac. Docker runs in a Linux virtual machine with no path to
Apple's Metal layer, so the device falls back to CPU. The complex FFT preflight
still passes, but it no longer guards the accelerator it was written for.

The straightforward guideline is to containerize the evaluation, inference, and demo processes while performing training on the host with active MPS. Training the 12.9 million parameter network on CPU within a container would be very slow.

## Build

```bash
docker compose build
```

## Run the evaluation pipeline

Each stage runs as its own service and shares the same mounts.

```bash
docker compose run --rm app         # device preflight, reports cpu
docker compose run --rm calibrate   # fit the gate, writes checkpoints/gate.json
docker compose run --rm eval        # report and artifacts/recon_panel.png
docker compose run --rm sweep       # frontier and resampling figures
docker compose run --rm test        # pytest
```

Calibrate runs before eval. The eval step reads
`checkpoints/reconstructor/best.pt` and `checkpoints/gate.json`, and the sweep
step reads the checkpoint as well. Figures and CSV land in `artifacts/` through
the mount.

## Notes

On Docker Desktop for Mac, file ownership maps to the active user, so files in
`artifacts/` stay owned by that user. On a plain Linux host they come out owned
by root; adding `user: "${UID}:${GID}"` to the shared block in the compose file
fixes that.

The dependency versions use lower bounds rather than exact pins, so the image is
not bit-for-bit reproducible yet. Pinning exact versions, torch first, removes
the main source of drift across the CPU, MPS, and CUDA wheels.

## Running on a CUDA host

The same code runs on CUDA without any changes because `active_device()` detects it.
A GPU image requires two edits. The CPU torch line in the Dockerfile is replaced with the
matching CUDA wheel, for example
`pip install torch --index-url https://download.pytorch.org/whl/cu121`, and the
container is granted GPU access at runtime.

```bash
docker compose run --rm --gpus all app python src/train.py
```

This requires the NVIDIA Container Toolkit on the host. A fully isolated GPU
build can base the image on an `nvidia/cuda` runtime image with Python added.
The CPU image stays the portable default.

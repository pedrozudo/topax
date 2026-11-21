# Topax

Topax is a JAX library for large-scale 2D and 3D topology optimization. It combines matrix-free finite-element operators, geometric multigrid, Galerkin coarsening, and sparse coarse-grid solves.

Topax uses [Xolky](https://github.com/pedrozudo/xolky) for sparse Cholesky factorization. Xolky builds from source against SuiteSparse CHOLMOD for CPU execution, cuDSS for NVIDIA CUDA execution, or both.

## Installation

Topax uses [Xolky](https://github.com/pedrozudo/xolky) for sparse Cholesky factorization. Xolky supports SuiteSparse CHOLMOD on CPU and NVIDIA cuDSS on GPU.

The commands below assume that GitHub SSH access is configured.

### 1. Quick CPU installation with SuiteSparse

On Ubuntu or Debian, install the compiler and SuiteSparse development files:

```bash
sudo apt-get update
sudo apt-get install -y build-essential libsuitesparse-dev
```

Install Xolky with the CHOLMOD backend, followed by Topax:

```bash
XOLKY_BUILD_CHOLMOD=on XOLKY_BUILD_CUDA=off \
python -m pip install git+ssh://git@github.com/pedrozudo/xolky.git

python -m pip install git+ssh://git@github.com/pedrozudo/topax.git
```

Other Linux distributions should install their equivalent SuiteSparse development package before installing Xolky.

### 2. NVIDIA CUDA installation with cuDSS

Install a compatible NVIDIA driver and CUDA 13 toolkit using the [NVIDIA CUDA installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/).

Configure the [NVIDIA CUDA APT repository](https://developer.nvidia.com/cuda-downloads) for your Ubuntu version, then install cuDSS:

```bash
sudo apt-get update
sudo apt-get install -y libcudss1-cuda-13 libcudss1-dev-cuda-13
```

Install JAX with CUDA 13 support:

```bash
python -m pip install "jax[cuda13]>=0.11.1"
```

Install Xolky with the CUDA backend, followed by Topax:

```bash
XOLKY_BUILD_CUDA=on XOLKY_BUILD_CHOLMOD=off \
python -m pip install git+ssh://git@github.com/pedrozudo/xolky.git

python -m pip install git+ssh://git@github.com/pedrozudo/topax.git
```

Xolky automatically detects cuDSS installed through NVIDIA's APT packages.

#### Custom installation paths

If CUDA or cuDSS is installed in a non-standard location, specify their installation prefixes before installing Xolky:

```bash
export CUDA_HOME=/path/to/cuda
export CUDSS_ROOT=/path/to/cudss
```

Then use the same Xolky and Topax installation commands above.

At runtime, CUDA and cuDSS shared libraries must be accessible to the dynamic linker. Configure `LD_LIBRARY_PATH` if necessary.

### 3. Fine-grained installation

For installations requiring both backends or additional configuration, see the [Xolky installation guide](https://github.com/pedrozudo/xolky#installation).

The available build controls are:

- `XOLKY_BUILD_CHOLMOD=auto|on|off`
- `XOLKY_BUILD_CUDA=auto|on|off`
- `CHOLMOD_ROOT=/path/to/suitesparse`
- `CUDA_HOME=/path/to/cuda`
- `CUDSS_ROOT=/path/to/cudss`

To install Topax from a local repository:

```bash
git clone git@github.com:pedrozudo/topax.git
cd topax
python -m pip install .
```

For an editable development installation with test dependencies:

```bash
python -m pip install -e ".[dev]"
pytest
```

## Examples

The [examples directory](examples) contains a small API example together with
larger reproductions of published topology-optimization problems. Start with
[the 3D compliance example](examples/compliance_3d.py) to see how to construct
a grid, apply boundary conditions and forces, configure the solver, and
minimize compliance under a volume constraint with MMA.

Run it from the repository root:

~~~bash
python examples/compliance_3d.py
~~~

Use the CLI options to change the problem and optimization size:

~~~bash
python examples/compliance_3d.py \
    --elements 16 8 8 \
    --levels 3 \
    --iterations 25 \
    --volume-fraction 0.4
~~~

### Literature reproductions

The cases in [benchmarks](benchmarks) reproduce or
compare against published topology-optimization problems. Each directory
contains notes about the problem, implementation choices, and run commands:

- [Sigmund (2007)](benchmarks/benchmark_scripts/topax/sigmund_2007)
- [Lazarov and Sigmund (2011)](benchmarks/benchmark_scripts/topax/lazarov_2011)
- [Träff et al. (2023)](benchmarks/benchmark_scripts/topax/traeff_2023)
- [Wang et al. (2025)](benchmarks/benchmark_scripts/topax/wang_2025)

Run these scripts from the repository root using the paths documented in their
respective READMEs. Some literature cases are intended as large-scale
benchmarks and may require substantial GPU memory.

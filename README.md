# Cell Diffusion

### Installation

To set up the development environment, we recommend using [mamba](https://mamba.readthedocs.io/en/latest/installation.html) for faster package resolution and installation.

#### CPU-only environment (or MPS)

```bash
mamba env create -f environment.yaml
mamba activate cell-diffusion-env
```

#### CUDA-enabled environment

```bash
mamba env create -f environment-cuda.yaml
mamba activate cell-diffusion-env-cuda
```

#### Adding a new package

1. add new package to environment.yaml
2. run `mamba env update -f environment.yaml --prune`

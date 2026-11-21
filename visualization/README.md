# Visualization

This directory contains scripts for saving and rendering complete optimization histories. These runs are separate from the performance benchmarks because transferring and writing a density field after every design iteration changes the measured runtime.

## Generate the Wang R800 history

`wang_2025/r800.py` solves the $800\times400\times400$ cantilever using four grid resolutions, with Galerkin element matrices materialized after the second coarsening. It uses the physically fixed sensitivity-filter radius from Wang et al. and saves the initial design and every update through iteration 50.

Run it from the repository root in a configured Topax environment:

```bash
python visualization/wang_2025/r800.py \
    --output-dir /path/to/run
```

The output directory must not already exist. A completed run contains `configuration.toml`, `iterations.csv`, a `density/` directory, and a `COMPLETE` marker. Each density is an uncompressed, memory-mappable `float32` NumPy array with shape `(800, 400, 400)`; the complete 51-frame history requires approximately $24.3\,\mathrm{GiB}$.

### Slurm submission

The supplied `wang_2025/r800.slurm` launcher requests one GPU and runs the complete 50-iteration optimization. Activate the Topax environment before submission and choose:

- the Slurm allocation account;
- a new output directory on storage accessible from the compute node.

Submit from the repository root with:

```bash
ACCOUNT=<allocation-account> \
VISUALIZATION_OUTPUT_DIR=/path/to/project-storage/wang_2025/r800/<run-id> \
bash visualization/wang_2025/r800.slurm
```

The launcher prints the submitted job ID. The output directory must not already exist and should have at least $25\,\mathrm{GiB}$ available. `PARTITION`, `CPUS`, and `WALLTIME` can be set to override the launcher defaults. For example:

```bash
PARTITION=<gpu-partition> \
CPUS=72 \
WALLTIME=01:00:00 \
ACCOUNT=<allocation-account> \
VISUALIZATION_OUTPUT_DIR=/path/to/project-storage/wang_2025/r800/<run-id> \
bash visualization/wang_2025/r800.slurm
```


## Render a design

Rendering requires [PyVista](https://pyvista.org/):

```bash
python -m pip install pyvista
python visualization/wang_2025/render.py \
    --run-directory /path/to/run \
    --output /path/to/iteration_050.png
```

The renderer extracts the $\rho=0.5$ material boundary directly from the continuous density field. A one-cell exterior void layer closes surfaces at the design-domain boundary without changing the computed densities. Iteration 50 is rendered by default. Pass `--show` to adjust the orthographic camera interactively; the resulting camera parameters are printed when the window closes.


## Create a video

Render the original floating-point densities using the fixed camera in `wang_2025/video.py`:

```bash
python visualization/wang_2025/video.py \
    --run-directory /path/to/run \
    --output-directory /path/to/run/video_frames
```

The original floating-point fields are used directly to preserve smooth material boundaries. Only one density field is memory-mapped at a time.

Encode the PNG sequence separately with FFmpeg:

```bash
ffmpeg -framerate 6 \
    -i /path/to/run/video_frames/frame_%03d.png \
    -vf "tpad=stop_mode=clone:stop_duration=2,fps=24" \
    -c:v libx264 -crf 18 -preset slow -pix_fmt yuv420p \
    -movflags +faststart /path/to/wang_r800.mp4
```

This displays six optimization designs per second, encodes at 24 frames per second, and holds the final design for two seconds.

## Generate the Träff R640 history

`traeff_2023/r640.py` solves the $640\times320\times320$ cantilever for 50 design iterations and saves the initial design and every update. The benchmark's original density filter is replaced by a Helmholtz density filter whose physical radius is fixed at $0.046875$ on the $2\times1\times1$ domain. This equals $1.5$ element widths on the R64 reference mesh and $15$ element widths on R640. The saved fields are the filtered physical densities used by the stiffness interpolation.

Run it from the repository root in a configured Topax environment:

```bash
python visualization/traeff_2023/r640.py \
    --output-dir /path/to/run
```

The output structure is the same as for the Wang history. Each density has shape `(640, 320, 320)`, and the 51-frame history requires approximately $12.5\,\mathrm{GiB}$.

### Slurm submission

The supplied `traeff_2023/r640.slurm` launcher requests one GPU and runs all 50 design iterations. Activate the Topax environment before submission, then provide the allocation account and a new output directory:

```bash
ACCOUNT=<allocation-account> \
VISUALIZATION_OUTPUT_DIR=/path/to/project-storage/traeff_2023/r640/<run-id> \
bash visualization/traeff_2023/r640.slurm
```

`PARTITION`, `CPUS`, and `WALLTIME` can be set to override the defaults.

Render the final design as a PNG with:

```bash
python visualization/traeff_2023/render.py \
    --run-directory /path/to/run \
    --output /path/to/iteration_050.png
```

Use `--show` to refine the initial orthographic camera and print the resulting camera parameters. Render the complete PNG sequence with:

```bash
python visualization/traeff_2023/video.py \
    --run-directory /path/to/run \
    --output-directory /path/to/run/video_frames
```

The FFmpeg command above can be reused with the Träff frame directory and a different output filename.


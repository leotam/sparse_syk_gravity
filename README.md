# SYK Holography Simulation Framework

This repository contains the reference implementation for the paper **["Geometry of Traversable Wormholes and Quantum Chaos in Magnitude-Pruned SYK Models"](https://doi.org/10.5281/zenodo.18037727)**.

It provides a high-performance, tensor-optimized framework for simulating Sachdev-Ye-Kitaev (SYK) quantum many-body systems. The codebase focuses on holographic properties, including wormhole traversability (OTOC), spectral form factors (SFF), and chaos indicators (Gap Ratio) across different sparsity levels and geometries.

## Installation

```bash
git clone https://github.com/leotam/sparse_syk_gravity
cd sparse_syk_gravity
pip install -e .
```

## Docker (Recommended)

For optimal performance, especially for the GPU-accelerated experiments (Figure 5), we recommend using the NVIDIA cuQuantum container.

Pull and run the container:

```bash
nvidia-docker run --gpus all -it --rm \
    --entrypoint /bin/bash \
    -v $(pwd):/workspace/syk-research \
    nvcr.io/nvidia/cuquantum-appliance:25.11-cuda12.9.1-devel-ubuntu24.04-x86_64
```

## Usage

The simulations are driven by a single CLI entry point: `run_experiment.py`.

### Figure 1: Compression Benchmark

Evaluates the robustness of the traversable wormhole protocol as the Hamiltonian is sparsified.

```bash
time python run_experiment.py benchmark --ensemble 40
```

### Figure 2: Phase Diagram

Sweeps system sizes (\(N=4\) to \(12\)) and sparsities to map the "Holographic Phase" boundary where the wormhole remains open.

```bash
time python run_experiment.py phase_diagram --ensemble 40
```

Data is saved to `v2syk_phase_data/` and allows for resuming interrupted runs.

### Figure 3: Linear Geometry

Simulates a 1D chain geometry (Linear SYK) to test teleportation fidelity in non-all-to-all connected systems.

```bash
time python run_experiment.py linear --ensemble 32
```

### Figure 4: Shapiro Delay

Measures the time delay of the signal as a function of the perturbation strength (simulating the gravitational Shapiro delay).

```bash
time python run_experiment.py shapiro --sparsity 0.1 --ensemble 1000
```

### Figure 5: GPU Gap Ratio

Calculates the adjacent gap ratio to detect the transition from chaotic (GUE) to integrable (Poisson) statistics.

```bash
time python run_experiment.py gpu_gap --ensemble 8
```

**Hardware Note:** The results in the paper for Figure 5 were generated using 2x NVIDIA RTX A6000 GPUs. The rest is run on a 48 core, 128GB RAM machine.

## Project Structure

- **`src/physics`**: Core logic for Hamiltonian generation, Majorana algebra, and time evolution.
- **`src/experiments`**: Standalone modules corresponding to each figure in the paper.
- **`src/config.py`**: Thread safety configurations and physics constants (J=1.0, beta=4.0).

If you want a fast smoke test run:
```bash
time python run_experiment.py benchmark --ensemble 2 --sparsity 1.0
time python run_experiment.py phase_diagram --n_values 4 --ensemble 2 --sparsity 1.0
time python run_experiment.py linear --ensemble 2 --sparsity 1.0
time python run_experiment.py shapiro --ensemble 2 --n_majoranas 8
time python run_experiment.py gpu_gap --gpus 0 --n_values 16 --ensemble 1 --sparsity 1.0
```

```
FIGURE 1: SYK COMPRESSION BENCHMARK (Ensemble=2)
Sp 100.0% | Terms: 495/495 | Peak: 0.0309 | SFF Ratio: 1.37

Generating Figure 1...
Saved figure to figures/figure1_benchmark.png

FIGURE 2: PHASE DIAGRAM SWEEP
Running N=4 Sp=1.0...
Generating Figure 2...

FIGURE 3: LINEAR GEOMETRY SWEEP
Running 10 simulations...
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 48 concurrent workers.
[Parallel(n_jobs=-1)]: Done   2 out of  10 | elapsed:    0.7s remaining:    2.8s
[Parallel(n_jobs=-1)]: Done   5 out of  10 | elapsed:    0.7s remaining:    0.7s
[Parallel(n_jobs=-1)]: Done   8 out of  10 | elapsed:    0.7s remaining:    0.2s
[Parallel(n_jobs=-1)]: Done  10 out of  10 | elapsed:    0.9s finished

FIGURE 4: SHAPIRO DELAY (N=8, Ensemble=2)
Simulating Strength 0.00...
Simulating Strength 0.39...
Simulating Strength 0.79...
Simulating Strength 1.18...
Simulating Strength 1.57...

============================================================
FIGURE 5: GPU GAP RATIO SCALING
GPUs: [0] | N_Values: [16] | Ensemble: 1
============================================================
Running 1 tasks...
Total Runtime: 5.36s
Saved figures/figure5_gap_scaling.png
```

## License & Citation

This code is released under the Apache 2.0 License. If you use this software in your research, please cite our work:

```bibtex
@article{MagSYK,
  title={Geometry of Traversable Wormholes and Quantum Chaos in Magnitude-Pruned SYK Models},
  author={Tam, Leonidas and Woodside, Barrett},
  journal={Zenodo},
  year={2025},
  doi={10.5281/zenodo.18037727},
  url={https://doi.org/10.5281/zenodo.18037727}
}
```
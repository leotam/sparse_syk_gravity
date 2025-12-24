#!/usr/bin/env python3
"""
SYK Holographic Research CLI
Usage: python run_experiment.py [experiment_name] [options]
"""

from src.config import set_thread_safety
set_thread_safety()

import argparse
from src.experiments import benchmark, phase_diagram, linear, shapiro, gpu_gap

def main():
    parser = argparse.ArgumentParser(description="SYK Experiments")
    subparsers = parser.add_subparsers(dest="experiment", help="Which experiment to run")

    # --- Figure 1: Benchmark ---
    p_bench = subparsers.add_parser("benchmark", help="Figure 1: Compression Benchmark")
    p_bench.add_argument("--ensemble", type=int, default=40, help="Ensemble size")
    p_bench.add_argument("--sparsity", type=float, default=None, help="Specific sparsity (default: runs sweep)")

    # --- Figure 2: Phase Diagram ---
    p_phase = subparsers.add_parser("phase_diagram", help="Figure 2: Phase Boundary Sweep")
    p_phase.add_argument("--n_values", nargs="+", type=int, default=[4, 6, 8, 10, 12])
    p_phase.add_argument("--sparsity", type=float, default=None, help="Specific sparsity (default: runs sweep)")
    p_phase.add_argument("--ensemble", type=int, default=40, help="Ensemble size")
    p_phase.add_argument("--data_dir", type=str, default="v2syk_phase_data")

    # --- Figure 3: Linear ---
    p_linear = subparsers.add_parser("linear", help="Figure 3: Linear Geometry")
    p_linear.add_argument("--ensemble", type=int, default=32)
    p_linear.add_argument("--sparsity", type=float, default=None, help="Specific sparsity (default: runs sweep)")

    # --- Figure 4: Shapiro ---
    p_shapiro = subparsers.add_parser("shapiro", help="Figure 4: Shapiro Delay")
    p_shapiro.add_argument("--sparsity", type=float, default=0.1, help="Sparsity level")
    p_shapiro.add_argument("--ensemble", type=int, default=100, help="Ensemble size")
    p_shapiro.add_argument("--n_majoranas", type=int, default=12, help="System size (N)")

    # --- Figure 5: GPU Gap ratio ---
    p_gpu = subparsers.add_parser("gpu_gap", help="Figure 5: GPU Gap Ratio")
    p_gpu.add_argument("--gpus", nargs="+", type=int, default=[0, 1], dest="gpu_ids")
    p_gpu.add_argument("--ensemble", type=int, default=8, help="Ensemble size per point")
    p_gpu.add_argument("--sparsity", type=float, default=None, help="Specific sparsity (default: runs sweep)")
    p_gpu.add_argument("--n_values", nargs="+", type=int, default=[16, 18, 20, 22, 24, 26, 28, 30])

    args = parser.parse_args()

    if args.experiment == "benchmark":
        benchmark.run(args)
    elif args.experiment == "phase_diagram":
        phase_diagram.run(args)
    elif args.experiment == "linear":
        linear.run(args)
    elif args.experiment == "shapiro":
        shapiro.run(args)
    elif args.experiment == "gpu_gap":
        gpu_gap.run(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
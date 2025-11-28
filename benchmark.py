#!/usr/bin/env python3
import subprocess
import re
import argparse
import os
import sys

# Try importing matplotlib for plotting
try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not found. Charts will not be generated.")
    print("Install it via: pip install matplotlib")

# --- Configuration ---
DEFAULT_BIN_DIR = "./build/bin"
NAIVE_EXE = "particles-bench"
MPI_EXE = "particles-mpi"


def parse_args():
    parser = argparse.ArgumentParser(description="Run Particle Simulation Benchmarks")

    # Simulation Parameters
    parser.add_argument(
        "-n", "--count", type=int, default=1000, help="Number of particles"
    )
    parser.add_argument(
        "-f", "--frames", type=int, default=1000, help="Number of frames"
    )
    parser.add_argument(
        "--bin-dir", default=DEFAULT_BIN_DIR, help="Path to executables"
    )
    parser.add_argument(
        "--output",
        default="benchmark_results",
        help="Prefix for output chart filenames",
    )

    # Selection Control
    parser.add_argument(
        "--run",
        nargs="+",
        choices=["naive", "omp", "mpi", "cuda"],
        default=["naive", "omp", "mpi"],
        help="Which versions to run. Example: --run omp mpi",
    )

    # Parallelism Options
    parser.add_argument(
        "--omp-threads",
        type=int,
        nargs="+",
        default=[2, 4, 8],
        help="List of OpenMP thread counts to test",
    )
    parser.add_argument(
        "--mpi-procs",
        type=int,
        nargs="+",
        default=[2, 4],
        help="List of MPI process counts to test",
    )

    return parser.parse_args()


def run_command(cmd, env=None):
    """Executes a shell command and returns the stdout."""
    try:
        full_env = os.environ.copy()
        if env:
            full_env.update(env)

        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=full_env,
            check=True,
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        # Only print stderr if it's not just a standard warning
        if e.stderr:
            print(f"Stderr: {e.stderr}")
        return None
    except FileNotFoundError:
        print(f"Executable not found in command: {cmd}")
        return None


def parse_output(output):
    """Extracts Time and UPS from the standard output."""
    if not output:
        return 0.0, 0.0

    time_match = re.search(r"Time:\s+([0-9\.]+)\s+s", output)
    ups_match = re.search(r"UPS.*:\s+([0-9\.]+)", output)

    time_val = float(time_match.group(1)) if time_match else 0.0
    ups_val = float(ups_match.group(1)) if ups_match else 0.0

    return time_val, ups_val


def plot_results(results, particle_count, filename_prefix):
    """Generates bar charts for Time and UPS."""
    if not HAS_MATPLOTLIB or not results:
        return

    # Unzip data
    labels = [f"{r[0]}\n{r[2]}" for r in results]  # e.g. "OpenMP\n4 Threads"
    times = [r[3] for r in results]
    ups = [r[4] for r in results]
    colors = ["#ff9999", "#66b3ff", "#99ff99", "#ffcc99", "#c2c2f0"]

    # 1. Plot Execution Time (Lower is Better)
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, times, color=colors[: len(labels)])
    plt.xlabel("Configuration")
    plt.ylabel("Time (seconds)")
    plt.title(f"Simulation Time (Lower is Better)\nN={particle_count} Particles")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2f}s",
            ha="center",
            va="bottom",
        )

    time_filename = f"{filename_prefix}_time.png"
    plt.savefig(time_filename)
    print(f"Chart saved to {time_filename}")
    plt.close()

    # 2. Plot UPS (Higher is Better)
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, ups, color=colors[: len(labels)])
    plt.xlabel("Configuration")
    plt.ylabel("Updates Per Second (UPS)")
    plt.title(
        f"Simulation Performance (Higher is Better)\nN={particle_count} Particles"
    )
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
        )

    ups_filename = f"{filename_prefix}_ups.png"
    plt.savefig(ups_filename)
    print(f"Chart saved to {ups_filename}")
    plt.close()


def main():
    args = parse_args()

    bench_exe = os.path.join(args.bin_dir, NAIVE_EXE)
    mpi_exe = os.path.join(args.bin_dir, MPI_EXE)

    # Store results: (Name, Particles, Resources, Time, UPS)
    results = []

    print(f"--- Starting Benchmark (N={args.count}, Frames={args.frames}) ---")

    # 1. NAIVE
    if "naive" in args.run:
        cmd = [
            bench_exe,
            "--naive",
            "-n",
            str(args.count),
            "-f",
            str(args.frames),
            "--no-color",
        ]
        out = run_command(cmd)
        if out:
            t, ups = parse_output(out)
            results.append(("Naive", args.count, "Serial", t, ups))

    # 2. OpenMP (Grid)
    if "omp" in args.run:
        for threads in args.omp_threads:
            env = {"OMP_NUM_THREADS": str(threads)}
            cmd = [
                bench_exe,
                "-n",
                str(args.count),
                "-f",
                str(args.frames),
                "--no-color",
            ]
            out = run_command(cmd, env=env)
            if out:
                t, ups = parse_output(out)
                results.append(("OpenMP", args.count, f"{threads} Threads", t, ups))

    # 3. CUDA
    if "cuda" in args.run:
        cmd = [
            bench_exe,
            "--cuda",
            "-n",
            str(args.count),
            "-f",
            str(args.frames),
            "--no-color",
        ]
        out = run_command(cmd)
        if out:
            t, ups = parse_output(out)
            results.append(("CUDA", args.count, "GPU", t, ups))

    # 4. MPI
    if "mpi" in args.run:
        for procs in args.mpi_procs:
            cmd = [
                "mpirun",
                "-np",
                str(procs),
                mpi_exe,
                "-n",
                str(args.count),
                "-f",
                str(args.frames),
            ]
            out = run_command(cmd)
            if out:
                t, ups = parse_output(out)
                results.append(("MPI", args.count, f"{procs} Ranks", t, ups))

    # --- Print Table ---
    print("\n" + "=" * 75)
    print(
        f"{'Algorithm':<10} | {'Count':<8} | {'Resources':<15} | {'Time (s)':<10} | {'UPS':<10}"
    )
    print("-" * 75)
    for res in results:
        algo, n, rsrc, t, ups = res
        print(f"{algo:<10} | {n:<8} | {rsrc:<15} | {t:<10.4f} | {ups:<10.1f}")
    print("=" * 75 + "\n")

    # --- Generate Charts ---
    if results:
        plot_results(results, args.count, args.output)
    else:
        print("No results to plot.")


if __name__ == "__main__":
    main()

import subprocess
import re
import matplotlib.pyplot as plt
import os

# --- Configuration ---
OUTPUT_BIN = "particles-bench"

# Benchmark Parameters
PARTICLE_COUNTS = [500, 1000, 2000, 4000, 8000]  # For Algo Comparison
THREAD_COUNTS = [1, 2, 4, 8, 12, 16]  # For OpenMP Scaling
SCALING_PARTICLES = 10000  # Fixed high load for thread scaling
FRAMES = 1000  # Frames to simulate per run


def run_simulation(particles, frames, mode, render, threads):
    # Set OpenMP threads environment variable
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)

    # Command: ./sim [N] [Frames] [Mode] [Render]
    cmd = [f"./{OUTPUT_BIN}", str(particles), str(frames), str(mode), str(render)]

    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        output = result.stdout

        # Extract "Time: X s" using Regex
        match = re.search(r"Time:\s+([0-9\.]+)\s+s", output)
        if match:
            return float(match.group(1))
        else:
            print(f"⚠️ Error parsing output for P={particles}, T={threads}")
            return None
    except Exception as e:
        print(f"❌ Execution failed: {e}")
        return None


def benchmark_algorithms():
    print("📊 Experiment 1: Naive vs. Grid Algorithm (1 Thread)")
    naive_times = []
    grid_times = []

    for p in PARTICLE_COUNTS:
        print(f"   Testing {p} particles...")

        # Run Naive (Mode 1)
        t_naive = run_simulation(p, FRAMES, 1, 0, 1)
        naive_times.append(t_naive)

        # Run Grid (Mode 0)
        t_grid = run_simulation(p, FRAMES, 0, 0, 1)
        grid_times.append(t_grid)

    return naive_times, grid_times


def benchmark_scaling():
    print(f"\n🚀 Experiment 2: OpenMP Scaling ({SCALING_PARTICLES} particles)")
    times = []

    for t in THREAD_COUNTS:
        print(f"   Testing {t} threads...")
        time = run_simulation(SCALING_PARTICLES, FRAMES, 0, 0, t)
        times.append(time)

    return times


def plot_results(naive, grid, scaling_times):
    plt.figure(figsize=(14, 6))

    # --- Plot 1: Algorithm Comparison ---
    plt.subplot(1, 2, 1)
    plt.plot(PARTICLE_COUNTS, naive, "o--", color="red", label="Naive O(N^2)")
    plt.plot(PARTICLE_COUNTS, grid, "s-", color="green", label="Grid O(N)")
    plt.xlabel("Number of Particles")
    plt.ylabel("Time (seconds)")
    plt.title("Algorithm Performance")
    plt.grid(True)
    plt.legend()

    # --- Plot 2: OpenMP Speedup ---
    plt.subplot(1, 2, 2)

    # Calculate Speedup (Base time / Current time)
    base_time = scaling_times[0]
    speedups = [base_time / t for t in scaling_times]

    plt.plot(THREAD_COUNTS, speedups, "D-", color="blue")

    # Add "Ideal" line
    plt.plot(
        THREAD_COUNTS,
        THREAD_COUNTS,
        "--",
        color="gray",
        alpha=0.5,
        label="Ideal Scaling",
    )

    plt.xlabel("OpenMP Threads")
    plt.ylabel("Speedup Factor (vs 1 Thread)")
    plt.title(f"Parallel Scaling ({SCALING_PARTICLES} particles)")
    plt.xticks(THREAD_COUNTS)
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig("benchmark_results.png")
    print("\n💾 Charts saved to 'benchmark_results.png'")
    plt.show()


if __name__ == "__main__":
    # Run Ex 1
    naive, grid = benchmark_algorithms()

    # Run Ex 2
    scaling = benchmark_scaling()

    # Visualize
    plot_results(naive, grid, scaling)

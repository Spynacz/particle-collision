#!/usr/bin/env python3
import subprocess
import re
import argparse
import os
import sys
import csv

# --- Konfiguracja domyślna ---
DEFAULT_BIN_DIR = "./build/bin"  # Upewnij się, że ścieżka jest poprawna
NAIVE_EXE = "particles-bench"  # Nazwa pliku wykonywalnego (OpenMP/Naive/CUDA)
MPI_EXE = "particles-mpi"  # Nazwa pliku wykonywalnego MPI

# Sprawdzenie czy mamy bibliotekę do wykresów
try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not found. Charts will not be generated.")


def parse_args():
    parser = argparse.ArgumentParser(description="Advanced Particle Benchmark Runner")

    # Parametry symulacji
    parser.add_argument(
        "-n",
        "--particles",
        type=int,
        nargs="+",
        default=[500, 1000, 2000, 3000, 4000],
        help="Lista liczby cząstek do przetestowania (np. 500 1000 2000)",
    )
    parser.add_argument(
        "-f", "--frames", type=int, default=500, help="Liczba klatek w każdej symulacji"
    )

    # Ścieżki
    parser.add_argument(
        "--bin-dir", default=DEFAULT_BIN_DIR, help="Folder z plikami exe"
    )
    parser.add_argument(
        "--output", default="benchmark_results", help="Prefix nazw plików wynikowych"
    )

    # Wybór wersji (Flagi sterujące, o które prosiłeś)
    parser.add_argument(
        "--run",
        nargs="+",
        choices=["naive", "omp", "mpi", "cuda"],
        default=["naive", "omp", "mpi"],
        help="Wybierz wersje do uruchomienia (np. --run omp mpi)",
    )

    # Zasoby
    parser.add_argument(
        "--omp-threads",
        type=int,
        nargs="+",
        default=[2, 4, 8],
        help="Liczba wątków OpenMP",
    )
    parser.add_argument(
        "--mpi-procs", type=int, nargs="+", default=[2, 4], help="Liczba procesów MPI"
    )

    return parser.parse_args()


def run_command(cmd, env=None):
    """Uruchamia komendę w shellu i zwraca stdout."""
    try:
        full_env = os.environ.copy()
        if env:
            full_env.update(env)

        # Wypisanie samej komendy dla podglądu
        cmd_str = " ".join(cmd)
        print(f"--> Running: {cmd_str}")

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
        print(f"Error running: {cmd_str}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return None
    except FileNotFoundError:
        print(f"Executable not found: {cmd[0]}")
        return None


def parse_output(output):
    """Parsuje wyjście programu w poszukiwaniu Time i UPS."""
    if not output:
        return 0.0, 0.0

    # Szukamy: "Time: 1.234 s" oraz "UPS ... : 500.0"
    time_match = re.search(r"Time:\s+([0-9\.]+)\s+s", output)
    ups_match = re.search(r"UPS.*:\s+([0-9\.]+)", output)

    time_val = float(time_match.group(1)) if time_match else 0.0
    ups_val = float(ups_match.group(1)) if ups_match else 0.0
    return time_val, ups_val


def save_csv(results, filename):
    """Zapisuje wyniki do pliku CSV."""
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Algorithm", "Particles", "Resources", "Time_s", "UPS"])
        for r in results:
            writer.writerow(r)
    print(f"\nWyniki zapisano do: {filename}")


def plot_scaling_lines(results, filename_prefix):
    """Rysuje wykres liniowy: Oś X = Liczba Cząstek, Oś Y = Czas."""
    if not HAS_MATPLOTLIB or not results:
        return

    plt.figure(figsize=(10, 7))

    # Grupowanie wyników po konfiguracji (np. "MPI (4 Ranks)")
    data_map = {}  # { "ConfigName": { particle_count: time } }

    for row in results:
        algo, n, res, time_s, ups = row
        label = f"{algo} ({res})"
        if label not in data_map:
            data_map[label] = {}
        data_map[label][n] = time_s

    # Rysowanie linii
    markers = ["o", "s", "^", "D", "x", "*"]
    for i, (label, points) in enumerate(data_map.items()):
        sorted_n = sorted(points.keys())
        sorted_times = [points[n] for n in sorted_n]
        marker = markers[i % len(markers)]
        plt.plot(sorted_n, sorted_times, marker=marker, label=label, linewidth=2)

    plt.xlabel("Number of Particles (N)")
    plt.ylabel("Execution Time (seconds)")
    plt.title("Scaling Performance: Particle Count vs Time")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    out_file = f"{filename_prefix}_scaling_lines.png"
    plt.savefig(out_file)
    print(f"Wykres liniowy zapisano do: {out_file}")
    plt.close()


def main():
    args = parse_args()

    bench_exe = os.path.join(args.bin_dir, NAIVE_EXE)
    mpi_exe = os.path.join(args.bin_dir, MPI_EXE)

    # Struktura wyników: (Algo, N, Resources, Time, UPS)
    all_results = []

    print(f"=== ROZPOCZYNAM BENCHMARK ===")
    print(f"Cząstki: {args.particles}")
    print(f"Wersje: {args.run}")

    # Główna pętla po liczbie cząstek
    for n in args.particles:
        print(f"\n--- Testing N={n} Particles ---")

        # 1. NAIVE
        if "naive" in args.run:
            # Uwaga: Naive jest bardzo wolne dla dużych N, ostrzegamy
            if n > 5000:
                print(f"Skipping Naive for N={n} (too slow)")
            else:
                cmd = [
                    bench_exe,
                    "--naive",
                    "-n",
                    str(n),
                    "-f",
                    str(args.frames),
                    "--no-color",
                ]
                out = run_command(cmd)
                if out:
                    t, ups = parse_output(out)
                    all_results.append(("Naive", n, "Serial", t, ups))

        # 2. OPENMP
        if "omp" in args.run:
            for threads in args.omp_threads:
                env = {"OMP_NUM_THREADS": str(threads)}
                cmd = [bench_exe, "-n", str(n), "-f", str(args.frames), "--no-color"]
                out = run_command(cmd, env=env)
                if out:
                    t, ups = parse_output(out)
                    all_results.append(("OpenMP", n, f"{threads} Threads", t, ups))

        # 3. MPI
        if "mpi" in args.run:
            for procs in args.mpi_procs:
                cmd = [
                    "mpirun",
                    "-np",
                    str(procs),
                    mpi_exe,
                    "-n",
                    str(n),
                    "-f",
                    str(args.frames),
                ]
                out = run_command(cmd)
                if out:
                    t, ups = parse_output(out)
                    all_results.append(("MPI", n, f"{procs} Ranks", t, ups))

        # 4. CUDA (opcjonalnie)
        if "cuda" in args.run:
            cmd = [
                bench_exe,
                "--cuda",
                "-n",
                str(n),
                "-f",
                str(args.frames),
                "--no-color",
            ]
            out = run_command(cmd)
            if out:
                t, ups = parse_output(out)
                all_results.append(("CUDA", n, "GPU", t, ups))

    # --- Raportowanie ---
    if not all_results:
        print("Brak wyników.")
        return

    # Tabela w konsoli
    print("\n" + "=" * 80)
    print(
        f"{'Algorithm':<10} | {'N':<6} | {'Resources':<15} | {'Time (s)':<10} | {'UPS':<10}"
    )
    print("-" * 80)
    for res in all_results:
        print(
            f"{res[0]:<10} | {res[1]:<6} | {res[2]:<15} | {res[3]:<10.4f} | {res[4]:<10.0f}"
        )
    print("=" * 80)

    # Zapis do CSV
    save_csv(all_results, f"{args.output}_data.csv")

    # Generowanie wykresów
    plot_scaling_lines(all_results, args.output)


if __name__ == "__main__":
    main()

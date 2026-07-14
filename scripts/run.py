#!/usr/bin/env python3

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    print("This script requires Python 3.11+.", file=sys.stderr)
    sys.exit(1)

try:
    import tomli_w
except ModuleNotFoundError:
    print(
        "Missing dependency: tomli-w\n"
        "Install it with:\n\n"
        "    pip install tomli-w\n",
        file=sys.stderr,
    )
    sys.exit(1)


INPUT_TOML = Path("src/input/input.toml")
PLOT_SCRIPT = Path("scripts/plot.py")
NC_FILE = Path("results/raw/local_energy.nc")
OUTPUT_ROOT = Path("results/processed")

BUILD_DIR = Path("build")
EXECUTABLE = Path("build/langevin_dynamics")

POTENTIALS = ["FPU", "Josephson"]

PLOT_VARIABLES = {
    "local_total_energy": ["--variable", "local_total_energy"],
    "local_kinetic_energy": ["--variable", "local_kinetic_energy"],
    "local_potential_energy": ["--variable", "local_potential_energy"],
    # normalized energies
    "normalized_total_energy": ["--variable", "normalized_total_energy"],
    "normalized_kinetic_energy": ["--variable", "normalized_kinetic_energy"],
    "normalized_potential_energy": ["--variable", "normalized_potential_energy"],
    # moments
    "first_moment_total_energy": [
        "--variable", "first_moment_total_energy",
        "--kind", "line",
    ],
    "total_energy_spread": [
        "--variable", "total_energy_spread",
        "--kind", "line",
    ],
    "total_energy_centroid_spread": [
        "--variable", "normalized_total_energy",
        "--kind", "heat",
        "--overlay-moments",
    ],
    # pearson correlators
    "pearson_bond": ["--variable", "pearson_bond_correlation"],
    "pearson_momentum": ["--variable", "pearson_momentum_correlation"],
    "pearson_position": ["--variable", "pearson_position_correlation"],
}


def run_command(command: list[str]) -> None:
    print(f"\nRunning: {' '.join(command)}")
    subprocess.run(command, check=True)


def load_toml(path: Path) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


def write_toml(path: Path, data: dict) -> None:
    with path.open("wb") as f:
        tomli_w.dump(data, f)


def set_potential(potential: str) -> None:
    data = load_toml(INPUT_TOML)

    if "model" not in data:
        raise KeyError(f"No [model] section found in {INPUT_TOML}")

    data["model"]["potential"] = potential
    write_toml(INPUT_TOML, data)

    print(f"Set potential = {potential}")


def build_cpp_code() -> None:
    if not BUILD_DIR.exists():
        raise FileNotFoundError(
            "Could not find build directory. Run CMake configure first, e.g.:\n"
            "  cmake -S . -B build"
        )

    run_command(["cmake", "--build", str(BUILD_DIR)])


def run_simulation() -> None:
    if not EXECUTABLE.exists():
        raise FileNotFoundError(
            f"Could not find executable: {EXECUTABLE}\n"
            "Try running:\n"
            "  cmake --build build"
        )

    run_command([str(EXECUTABLE)])


def make_plots(potential: str) -> None:
    output_dir = OUTPUT_ROOT / potential.lower()
    output_dir.mkdir(parents=True, exist_ok=True)

    for output_name, plot_args in PLOT_VARIABLES.items():
        output_file = output_dir / f"{output_name}.png"

        run_command(
            [
                sys.executable,
                str(PLOT_SCRIPT),
                str(NC_FILE),
                "--metadata",
                *plot_args,
                "-o",
                str(output_file),
            ]
        )


def main() -> None:
    if not INPUT_TOML.exists():
        raise FileNotFoundError(f"Could not find {INPUT_TOML}")

    if not PLOT_SCRIPT.exists():
        raise FileNotFoundError(f"Could not find {PLOT_SCRIPT}")

    backup_path = INPUT_TOML.with_suffix(".toml.bak")
    shutil.copy2(INPUT_TOML, backup_path)
    print(f"Backed up original config to {backup_path}")

    original_data = load_toml(INPUT_TOML)
    original_potential = original_data.get("model", {}).get("potential")

    try:
        build_cpp_code()

        for potential in POTENTIALS:
            print("\n" + "=" * 80)
            print(f"Running simulation for potential: {potential}")
            print("=" * 80)

            set_potential(potential)
            run_simulation()

            if not NC_FILE.exists():
                raise FileNotFoundError(
                    f"Expected output file does not exist: {NC_FILE}"
                )

            make_plots(potential)

        if original_potential in POTENTIALS:
            set_potential(original_potential)

        print("\nAll runs and plots finished.")

    except BaseException:
        print(
            "\nRun failed or was interrupted. Restoring original input.toml from backup.",
            file=sys.stderr,
        )
        shutil.copy2(backup_path, INPUT_TOML)
        raise


if __name__ == "__main__":
    main()
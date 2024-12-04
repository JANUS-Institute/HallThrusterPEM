"""This script will:

1) Check for juliaup installation and install it if not found.
2) Ensure the specified Julia version is installed using juliaup.
3) Install HallThruster.jl with the specified version.

Run as:

python install_hallthruster.py --julia-version 1.10 --hallthruster-version 0.17.2

"""
import argparse
import os
import shlex
import subprocess
import platform
from pathlib import Path

from packaging.version import Version

PLATFORM = platform.system().lower()
JULIA_VERSION_DEFAULT = "1.10"
HALLTHRUSTER_VERSION_DEFAULT = "0.17.2"


def run_command(command, capture_output=True, text=None, shell=False):
    """Run a command using subprocess."""
    try:
        if PLATFORM == 'windows':
            command = ['powershell', '-command', command]
        else:
            if not shell:
                command = shlex.split(command)
        return subprocess.run(command, capture_output=capture_output, check=True, text=text, shell=shell)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Command `{command}` failed with code {e.returncode}: {e}\nError message: {e.stderr}") from e
    except subprocess.SubprocessError as e:
        raise RuntimeError(f"Subprocess error: {e}") from e


def install_juliaup():
    print("Installing juliaup...")
    if PLATFORM == "windows":
        run_command("winget install julia -s msstore --accept-package-agreements", capture_output=False, text=True)
    else:
        run_command("curl -fsSL https://install.julialang.org | sh", capture_output=False, text=True, shell=True)


def ensure_julia_version(julia_version):
    print(f"Checking installed Julia versions using juliaup...")

    try:
        proc_ret = run_command("juliaup status", text=True, capture_output=True)
        cmd_output = proc_ret.stdout
    except:
        cmd_output = ""

    found_installed = False
    highest_version = julia_version
    highest_channel = ''
    for line in cmd_output.splitlines():
        if "Channel" not in line:
            parts = [p.strip() for p in line.strip().split()]  # [Default, Channel, Version, Update] columns
            if len(parts) > 1:
                for idx, p in enumerate(parts):
                    if 'update' in p.lower():
                        break

                    char = '+' if '+' in p else ('-' if '-' in p else '')  # Version column has "+" or "-" for installed
                    if char:
                        installed_version = p.split(char)[0]
                        if Version(installed_version) >= Version(highest_version):
                            found_installed = True
                            highest_version = installed_version
                            highest_channel = parts[idx - 1]

    if found_installed:
        print(f"Found installed version {highest_version} >= {julia_version}. Using this version.")
        run_command(f"juliaup default {highest_channel}")
    else:
        print(f"No suitable version found. Installing and setting Julia version {julia_version} as default.")
        run_command(f"juliaup add {julia_version}", capture_output=False, text=True)
        run_command(f"juliaup default {julia_version}")


def install_hallthruster_jl(hallthruster_version):
    print(f"Checking for HallThruster.jl version {hallthruster_version} in global environments...")
    global_env_dir = Path(f'~/.julia/environments/').expanduser()
    env_path = global_env_dir / f"hallthruster_{hallthruster_version}"
    if env_path.exists():
        print(f"Found HallThruster.jl version {hallthruster_version} in global environments.")
        return
    else:
        print(f"HallThruster.jl environment {env_path} not found. Creating...")
        os.makedirs(env_path)
        if PLATFORM == 'windows':
            install_cmd = rf"julia -e 'using Pkg; Pkg.activate(\"{env_path.resolve()}\"); Pkg.add(name=\"HallThruster\", version=\"{hallthruster_version}\")'"
        else:
            install_cmd = rf"""julia -e 'using Pkg; Pkg.activate("{env_path.resolve()}"); Pkg.add(name="HallThruster", version="{hallthruster_version}")'"""

        run_command(install_cmd, text=True, capture_output=False)


def main(julia_version, hallthruster_version):
    juliaup_installed = False
    try:
        run_command("juliaup --version")
        juliaup_installed = True
    except:
        pass

    if not juliaup_installed:
        install_juliaup()

    ensure_julia_version(julia_version)

    install_hallthruster_jl(hallthruster_version)
    print("HallThruster installation completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Install specified Julia and HallThruster.jl versions.")
    parser.add_argument("-jv", "--julia-version", default=JULIA_VERSION_DEFAULT,
                        help="The Julia version to install (default: 1.10)")
    parser.add_argument("-hv", "--hallthruster-version", default=HALLTHRUSTER_VERSION_DEFAULT,
                        help="The HallThruster.jl version to install (default: 0.17.2)")
    args = parser.parse_args()

    main(args.julia_version, args.hallthruster_version)

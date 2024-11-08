"""This script will:

1) Check for juliaup installation and install it if not found.
2) Ensure the specified Julia version is installed using juliaup.
3) Install HallThruster.jl with the specified version (or git ref).

Run as:

python install_hallthruster.py --julia-version 1.10 --hallthruster-version 0.17.2 --git-ref main

Note: If `git-ref` is specified, this will override the `hallthruster-version` and instead install from GitHub.

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
HALLTHRUSTER_URL = "https://github.com/UM-PEPL/HallThruster.jl"
HALLTHRUSTER_NAME = "HallThruster"


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


def install_hallthruster_jl(hallthruster_version, git_ref):
    """Install from a specified version tag; override with `git_ref` from GitHub if provided."""
    ref_name = git_ref if git_ref is not None else hallthruster_version

    print(f'Checking for HallThruster.jl ref {ref_name} in global environments...')
    global_env_dir = Path(f'~/.julia/environments/').expanduser()
    env_path = global_env_dir / f"hallthruster_{ref_name}"

    if env_path.exists():
        print(f"Found HallThruster.jl ref {ref_name} in global environments.")

        if git_ref is not None:
            print(f"Updating HallThruster.jl ref {ref_name} from GitHub...")
            if PLATFORM == 'windows':
                update_cmd = rf"""julia -e 'using Pkg; Pkg.activate(raw\"{env_path.resolve()}\"); Pkg.update(raw\"{HALLTHRUSTER_NAME}\");'"""
            else:
                update_cmd = rf"""julia -e 'using Pkg; Pkg.activate("{env_path.resolve()}"); Pkg.update("{HALLTHRUSTER_NAME}");'"""

            run_command(update_cmd, text=True, capture_output=False)

        return
    else:
        print(f"HallThruster.jl environment for ref {ref_name} not found. Creating...")
        os.makedirs(env_path)
        if PLATFORM == 'windows':
            # Powershell needs the double quotes to be escaped
            pkg_cmd = rf'Pkg.add(url=\"{HALLTHRUSTER_URL}\", rev=\"{git_ref}\")' if git_ref is not None else \
                rf'Pkg.add(name=\"{HALLTHRUSTER_NAME}\", version=\"{hallthruster_version}\")'
            install_cmd = rf"julia -e 'using Pkg; Pkg.activate(raw\"{env_path.resolve()}\"); {pkg_cmd}'"
        else:
            pkg_cmd = rf'Pkg.add(url="{HALLTHRUSTER_URL}", rev="{git_ref}")' if git_ref is not None else \
                rf'Pkg.add(name="{HALLTHRUSTER_NAME}", version="{hallthruster_version}")'
            install_cmd = rf"""julia -e 'using Pkg; Pkg.activate("{env_path.resolve()}"); {pkg_cmd}'"""

        run_command(install_cmd, text=True, capture_output=False)


def main(julia_version, hallthruster_version, git_ref):
    juliaup_installed = False
    try:
        run_command("juliaup --version")
        juliaup_installed = True
    except:
        pass

    if not juliaup_installed:
        install_juliaup()

    ensure_julia_version(julia_version)

    install_hallthruster_jl(hallthruster_version, git_ref)
    print("HallThruster installation completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Install specified Julia and HallThruster.jl versions.")
    parser.add_argument("-j", "--julia-version", default=JULIA_VERSION_DEFAULT,
                        help="The Julia version to install (default: 1.10)")
    parser.add_argument("-t", "--hallthruster-version", default=HALLTHRUSTER_VERSION_DEFAULT,
                        help="The HallThruster.jl version to install (default: 0.17.2)")
    parser.add_argument("-r", "--git-ref", default=None,
                        help="Install from this git ref (branch, hash, etc.) from the HallThruster.jl "
                             "GitHub repository.")
    args = parser.parse_args()

    main(args.julia_version, args.hallthruster_version, args.git_ref)

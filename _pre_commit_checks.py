"""Run additional pre-commit checks in the local environment.

ALWAYS run this through `pre-commit run` rather than directly as `python pre_commit_checks.py`.

Options:
- `--pytest-status` - Ensures that all new source code has been tested with `pdm run test`
"""
from pathlib import Path
import subprocess
import sys
import shlex
import argparse


PROJECT_DIR = Path(__file__).parent
PYTEST_STATUS_FILE = PROJECT_DIR / '.pytest_status'
SUCCESS = 0


def run_git_command(command):
    """Run a git command and return its console output."""
    try:
        result = subprocess.run(command, capture_output=True, check=True, text=True, shell=True)
        return result
    except subprocess.CalledProcessError as e:
        sys.exit(f'Error running {command}: {e.stderr}')


def check_pytest_status():
    """Fail if there are some new source code changes that have not been tested yet, or if latest tests failed."""
    # Get list of all added/modified/renamed/deleted/untracked files, both staged or unstaged from src/ and test/
    changed_files = [shlex.split(line)[-1] for line in run_git_command('git status --porcelain').stdout.splitlines()]
    check_files = [(f, (PROJECT_DIR / f).lstat().st_mtime) for f in changed_files if Path(f).exists() and (
            f.startswith('src/') or f.startswith('tests/'))]

    # If there are no changes in src/ or test/ then there is no need to check pytest
    if len(check_files) == 0:
        sys.exit(SUCCESS)

    if not PYTEST_STATUS_FILE.exists():
        sys.exit(f'No pytest status file found at: {PYTEST_STATUS_FILE.name}. '
                 f'Please pass all tests via `pdm run test`.')

    with open(PYTEST_STATUS_FILE, 'r') as f:
        status_code = int(f.read().rstrip())

    if status_code > 0:
        sys.exit(f'Pytest status file {PYTEST_STATUS_FILE.name} contains non-zero status code: {status_code}. '
                 f'Please pass all tests via `pdm run test`.')

    pytest_timestamp = PYTEST_STATUS_FILE.lstat().st_mtime
    most_recent_timestamp = 0
    for file, timestamp in check_files:
        if timestamp > most_recent_timestamp:
            most_recent_timestamp = timestamp

    if most_recent_timestamp > pytest_timestamp:
        sys.exit(f'Most recently edited files in src/ or test/ have not been tested yet. '
                 f'Please pass all tests via `pdm run test`.')

    sys.exit(SUCCESS)  # Congrats


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform custom local pre-commit checks.')
    parser.add_argument('--pytest-status', action='store_true', help='Check that all tests have passed.')
    args = parser.parse_args()

    if args.pytest_status:
        check_pytest_status()

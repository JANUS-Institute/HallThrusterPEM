"""Simple debug script for testing sbatch jobs on HPC."""
import argparse
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import time
import numpy as np


def io_heavy_task(wait):
    time.sleep(wait)
    return f"Completed IO heavy task with wait {wait}s"


def cpu_heavy_task(samples):
    result = sum(np.random.rand(samples))
    return f"Completed CPU heavy task with {samples} samples"


def main():
    parser = argparse.ArgumentParser(description="Execute IO or CPU heavy tasks using a specified executor.")
    parser.add_argument("-e", "--executor", choices=["thread", "process"], default="thread",
                        help="Type of parallel executor.")
    parser.add_argument("-w", "--wait", type=float, default=1.0, help="Wait time for IO heavy task")
    parser.add_argument("-s", "--samples", type=int, default=100000, help="Number of samples for CPU heavy task")
    parser.add_argument("-m", "--max_workers", type=int, default=None, help="Maximum number of workers")

    args, _ = parser.parse_known_args()

    if args.executor == "thread":
        task = io_heavy_task
        task_args = (args.wait,)
        pool_executor = ThreadPoolExecutor
    elif args.executor == "process":
        task = cpu_heavy_task
        task_args = (args.samples,)
        pool_executor = ProcessPoolExecutor
    else:
        raise ValueError(f"Unknown executor type: {args.executor}")

    with pool_executor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(task, *task_args) for _ in range(os.cpu_count())]
        for i, future in enumerate(as_completed(futures)):
            print(f"Task {i}: {future.result()}")


if __name__ == "__main__":
    main()

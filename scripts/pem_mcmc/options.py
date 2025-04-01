import os
import shutil
from argparse import Namespace
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Type

from amisc import System, YamlLoader


@dataclass
class ExecutionOptions:
    executor: Type
    directory: Path
    max_workers: int
    fidelity: str | tuple | dict = (0, 0)
    noise_std: float = 0.05
    sample_aleatoric: bool = False


def load_system_and_opts(args: Namespace) -> tuple[System, ExecutionOptions]:
    config = args.config_file
    system = YamlLoader.load(config)
    if args.output_dir is None:
        system.root_dir = Path(config).parent
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        system.root_dir = Path(args.output_dir)

    print(system.root_dir)

    system.set_logger(stdout=True)

    # Copy config file into output dir
    if Path(config).name not in os.listdir(system.root_dir):
        shutil.copy(config, system.root_dir)

    match args.executor.lower():
        case "thread":
            executor = ThreadPoolExecutor
        case "process":
            executor = ProcessPoolExecutor
        case _:
            raise ValueError(f"Unsupported executor type: {args.executor}")

    opts = ExecutionOptions(
        executor,
        max_workers=args.max_workers,
        directory=system.root_dir / "mcmc",
        fidelity=(0, args.ncharge - 1),
        sample_aleatoric=args.sample_aleatoric,
        noise_std=args.noise_std,
    )

    os.mkdir(opts.directory)

    return system, opts

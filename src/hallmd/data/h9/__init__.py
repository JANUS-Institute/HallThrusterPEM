from importlib import resources
from pathlib import Path

from ..thrusterdata import ThrusterDataset
from . import gt2024 as _gt2024
from . import um2024 as _um2024


class h9(ThrusterDataset):
    @staticmethod
    def datasets_from_names(dataset_names: list[str]) -> list[Path]:
        data_list = []
        for dataset in dataset_names:
            match dataset:
                case "gt2024":
                    data_list += gt2024()
                case "um2024":
                    data_list += um2024()
                case _:
                    raise ValueError(f"Invalid dataset {dataset} selected for the H9 thruster.")

        return data_list

    @staticmethod
    def all_data() -> list[Path]:
        return gt2024() + um2024()


def gt2024() -> list[Path]:
    dir = resources.files(_gt2024)
    with resources.as_file(dir / "data.csv") as path:
        return [path]


def um2024() -> list[Path]:
    dir = resources.files(_um2024)
    datafiles = ["jion_UMH9_fixed.csv", "uion_UMH9.csv", "vcc_UMH9.csv"]
    paths = []
    for file in datafiles:
        with resources.as_file(dir / file) as path:
            paths.append(path)

    return paths

from importlib import resources
from pathlib import Path

from ..thrusterdata import ThrusterDataset


class H9(ThrusterDataset):
    @staticmethod
    def datasets_from_names(dataset_names: list[str]) -> list[Path]:
        data_list = []
        for dataset in dataset_names:
            match dataset:
                case "gt2024":
                    data_list += _gt2024()
                case "um2024":
                    data_list += _um2024()
                case _:
                    raise ValueError(f"Invalid dataset {dataset} selected for the H9 thruster.")

        return data_list

    @staticmethod
    def all_data() -> list[Path]:
        return _gt2024() + _um2024()

    @staticmethod
    def no_data_error():
        return ImportError(
            "The H9 data is export-controlled and not stored in this repository."
            + "If you have the data, you must manually place it in the src/hallmd/data/h9 directory"
        )


def _gt2024() -> list[Path]:
    try:
        from . import gt2024
    except ImportError:
        raise H9.no_data_error()

    dir = resources.files(gt2024)
    with resources.as_file(dir / "data.csv") as path:
        return [path]


def _um2024() -> list[Path]:
    try:
        from . import um2024
    except ImportError:
        raise H9.no_data_error()

    dir = resources.files(um2024)
    datafiles = ["jion_UMH9_fixed.csv", "uion_UMH9.csv", "vcc_UMH9.csv"]
    paths = []
    for file in datafiles:
        with resources.as_file(dir / file) as path:
            paths.append(path)

    return paths

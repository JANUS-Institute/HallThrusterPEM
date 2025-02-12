from importlib import resources
from pathlib import Path

from ..thrusterdata import ThrusterDataset


class SPT100(ThrusterDataset):
    @staticmethod
    def datasets_from_names(dataset_names: list[str]) -> list[Path]:
        data_list = []
        for dataset in dataset_names:
            match dataset:
                case "diamant2014":
                    data_list += _diamant2014()
                case "sankovic1993":
                    data_list += _sankovic1993()
                case "macdonald2019":
                    data_list += _macdonald2019()
                case _:
                    raise ValueError(f"Invalid dataset {dataset} selected.")

        return data_list

    @staticmethod
    def all_data() -> list[Path]:
        return _sankovic1993() + _macdonald2019() + _diamant2014()


def _diamant2014(datasets: list[str] | str | None = None) -> list[Path]:
    from . import diamant2014

    dir = resources.files(diamant2014)

    if datasets is None:
        datasets = ["L3", "aerospace"]
    elif isinstance(datasets, str):
        datasets = [datasets]

    data_paths: list[Path] = []
    for set in datasets:
        with resources.as_file(dir / f"data_{set}.csv") as path:
            data_paths.append(path)

    return data_paths


def _macdonald2019() -> list[Path]:
    from . import macdonald2019

    dir = resources.files(macdonald2019)
    with resources.as_file(dir / "data.csv") as path:
        return [path]


def _sankovic1993() -> list[Path]:
    from . import sankovic1993

    dir = resources.files(sankovic1993)
    with resources.as_file(dir / "data.csv") as path:
        return [path]

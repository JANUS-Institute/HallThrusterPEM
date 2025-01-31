from importlib import resources
from pathlib import Path

from . import diamant2014 as _diamant2014
from . import macdonald2019 as _macdonald2019
from . import sankovic1993 as _sankovic1993

def datasets_from_names(dataset_names: list[str]) -> list[Path]:
    data_list = []
    for dataset in dataset_names:
        match dataset:
            case "diamant2014":
                data_list += diamant2014()
            case "sankovic1993":
                data_list += sankovic1993()
            case "macdonald2019":
                data_list += macdonald2019()
            case _:
                raise ValueError(f"Invalid dataset {dataset} selected.")

    return data_list

def diamant2014(datasets: list[str] | str | None = None) -> list[Path]:
    dir = resources.files(_diamant2014)

    if datasets is None:
        datasets = ["L3", "aerospace"]
    elif isinstance(datasets, str):
        datasets = [datasets]

    data_paths: list[Path] = []
    for set in datasets:
        with resources.as_file(dir / f"data_{set}.csv") as path:
            data_paths.append(path)

    return data_paths


def macdonald2019() -> list[Path]:
    dir = resources.files(_macdonald2019)
    with resources.as_file(dir / "data.csv") as path:
        return [path]


def sankovic1993() -> list[Path]:
    dir = resources.files(_sankovic1993)
    with resources.as_file(dir / "data.csv") as path:
        return [path]


def all() -> list[Path]:
    return sankovic1993() + macdonald2019() + diamant2014()

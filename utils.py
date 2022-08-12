import json
from pathlib import Path

DATA_DIR = Path(__file__).parent / 'data'


def data_load(filename):
    """Convenience function to load .json data files"""
    with open(DATA_DIR / filename, 'r') as fd:
        input_data = json.load(fd)

    return input_data


def data_write(data, filename):
    """Convenience function to write .json data files"""
    with open(DATA_DIR / filename, 'w', encoding='utf-8') as fd:
        json.dump(data, fd, ensure_ascii=False, indent=4)

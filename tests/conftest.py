import logging
import re
from pathlib import Path
from urllib.parse import quote

import coverage
from coverage import Coverage

PROJECT_DIR = Path(__file__).parent.parent
PYTEST_STATUS_FILE = PROJECT_DIR / '.pytest_status'
README_FILE = PROJECT_DIR / 'README.md'
COVERAGE_FILE = PROJECT_DIR / '.coverage'


def update_readme_coverage_badge():
    """Update the test coverage (codecov) badge in the README with most recent pytest-cov results."""
    # Get total coverage percentage from .coverage file
    try:
        cov = Coverage(data_file=COVERAGE_FILE)
        cov.load()
        total = round(cov.report())
    except coverage.exceptions.NoDataError:
        print('No coverage data found. Skipping README update...')
        return

    # Get badge color
    color_bds = [(95, "brightgreen"), (90, "green"), (75, "yellowgreen"), (60, "yellow"), (40, "orange"), (0, "red")]
    badge_color = 'lightgrey'
    try:
        xtotal = int(total)
    except ValueError:
        pass
    else:
        for low_bound, color in color_bds:
            if xtotal >= low_bound:
                badge_color = color
                break
    badge_url = f"https://img.shields.io/badge/coverage-{total}{quote('%')}-{badge_color}?logo=codecov"

    empty_badge_pattern = "![Code Coverage]()"
    existing_badge_pattern = r"\!\[Code Coverage\]\(.*?\)"
    patterns_to_match = rf"{re.escape(empty_badge_pattern)}|{existing_badge_pattern}"
    replacement_str = f"![Code Coverage]({badge_url})"

    # Use regex to replace badge icon in README
    with open(README_FILE, "r+", encoding='utf-8') as f:
        text = f.read()
        if replacement_str in text:
            logging.info('Coverage in README is up to date -- no changes made.')
        elif bool(re.search(patterns_to_match, text)) is False:
            logging.warning(f"Couldn't find the pattern {empty_badge_pattern} in README.md.")
        else:
            updated_text = re.sub(patterns_to_match, replacement_str, text)
            f.seek(0)
            f.write(updated_text)
            f.truncate()
            logging.info("README has been successfully updated with test coverage.")


def pytest_sessionfinish(session, exitstatus):
    """Do things after pytest session finishes."""
    logging.shutdown()
    update_readme_coverage_badge()

    # Write an exitstatus to file (for pre-commit checking)
    with open(PYTEST_STATUS_FILE, 'w') as f:
        f.write(str(exitstatus))

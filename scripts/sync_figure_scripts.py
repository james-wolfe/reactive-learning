"""Write every figure cell of the figure notebooks to its script in scripts/figures/.

The notebooks are the reference version of the figure code. A figure cell begins with
a line such as "# Figure S3 (scripts/figures/figS03_axis_entry_distribution.py)", and
the script is exactly that cell. With --check, report stale or orphaned scripts and
exit with an error instead of writing.
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ("notebooks/main_figs.ipynb", "notebooks/sup_figs.ipynb")
HEADER = re.compile(r"# Figure \w+ \((scripts/figures/\w+\.py)\)\n")


def figure_cells():
    """Yield (script path, script text) for every figure cell, in notebook order."""
    for notebook in NOTEBOOKS:
        for cell in json.loads((ROOT / notebook).read_text())["cells"]:
            source = "".join(cell["source"])
            if cell["cell_type"] == "code" and (match := HEADER.match(source)):
                note = f"# Extracted from {notebook} by scripts/sync_figure_scripts.py.\n"
                yield ROOT / match[1], match[0] + note + source[match.end() :].rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify instead of writing")
    check = parser.parse_args().check

    expected = dict(figure_cells())
    stale = [p for p, text in expected.items() if not p.exists() or p.read_text() != text]
    orphans = sorted(set((ROOT / "scripts/figures").glob("*.py")) - set(expected))
    if check:
        problems = [f"out of sync: {p.relative_to(ROOT)}" for p in stale]
        problems += [f"no notebook cell: {p.relative_to(ROOT)}" for p in orphans]
        if problems:
            problems.append("Run `make figure-scripts` after editing the notebooks.")
            sys.exit("\n".join(problems))
        print(f"All {len(expected)} figure scripts match the notebooks.")
        return
    for path in stale:
        path.write_text(expected[path])
        print(f"wrote {path.relative_to(ROOT)}")
    for path in orphans:
        print(f"warning: {path.relative_to(ROOT)} has no notebook cell")


if __name__ == "__main__":
    main()

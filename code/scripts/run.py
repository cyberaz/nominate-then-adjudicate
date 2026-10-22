#!/usr/bin/env python3
"""Run a script in its experiment context; --list never imports experiment code."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ("ecole-miplib", "portfolio-study")
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("experiment", nargs="?", choices=EXPERIMENTS)
    parser.add_argument("script", nargs="?")
    parser.add_argument("args", nargs=argparse.REMAINDER)
    options = parser.parse_args()
    if options.list:
        for name in EXPERIMENTS:
            print(name + ":")
            for p in sorted((ROOT / "experiments" / name / "scripts").rglob("*.py")):
                print("  " + str(p.relative_to(ROOT / "experiments" / name / "scripts")))
        return 0
    if not options.experiment or not options.script:
        parser.error("provide an experiment and script, or --list")
    directory = ROOT / "experiments" / options.experiment
    scripts = directory / "scripts"
    script = (scripts / options.script).resolve()
    if script.suffix != ".py":
        script = script.with_suffix(".py")
    if not script.is_relative_to(scripts.resolve()) or not script.is_file():
        parser.error("script must exist inside the selected experiment")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(directory), str(ROOT / "code/src")])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.call([sys.executable, str(script), *options.args], cwd=directory, env=env)
if __name__ == "__main__":
    raise SystemExit(main())

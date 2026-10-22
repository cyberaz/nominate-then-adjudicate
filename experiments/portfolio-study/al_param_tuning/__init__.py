"""Experiment solver override; all other modules use the shared project source."""
from pathlib import Path
__path__ = [str(Path(__file__).resolve().parent),
            str(Path(__file__).resolve().parents[3] / "code/src/al_param_tuning")]

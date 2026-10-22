"""LP file export utilities."""

import os
from typing import Optional


def ensure_dir(filepath: str):
    """Ensure the directory for a filepath exists."""
    dirpath = os.path.dirname(filepath)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)


def export_model_to_lp(model, output_dir: str, filename: str) -> str:
    """
    Export a PySCIPOpt model to LP format.

    Args:
        model: PySCIPOpt Model
        output_dir: Directory to write to
        filename: Filename (without extension)

    Returns:
        Full path of the written file
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{filename}.lp")
    model.writeProblem(filepath)
    return filepath

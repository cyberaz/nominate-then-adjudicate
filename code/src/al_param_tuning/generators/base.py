"""
Base generator utilities shared by all problem family generators.
"""

import numpy as np


def set_quiet(model):
    """Suppress SCIP console output."""
    model.setIntParam("display/verblevel", 0)
    return model


def make_rng(seed):
    """Create a numpy RandomState from seed."""
    if seed is not None:
        return np.random.RandomState(seed)
    return np.random.RandomState()


def export_lp(model, filepath: str) -> str:
    """
    Export a PySCIPOpt Model to LP format file.

    Args:
        model: PySCIPOpt Model object
        filepath: Output file path (should end with .lp)

    Returns:
        The filepath written to
    """
    model.writeProblem(filepath)
    return filepath

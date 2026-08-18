"""Octopus — one API for eight feature selection techniques.

Public surface (keep it small on purpose):

    from octopus import Octopus, ArmResult, SelectionResult
"""

from octopus.core.runner import Octopus
from octopus.types import ArmResult, SelectionResult, TaskType

__all__ = ["Octopus", "ArmResult", "SelectionResult", "TaskType"]
__version__ = "0.1.0"

# TODO(you): keep `from octopus import ...` import-light. Never import shap,
# plotly or jinja2 at module level here — those live behind optional extras and
# must only be imported inside the function that needs them.

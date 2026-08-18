"""Exception hierarchy.

One base class so callers can `except OctopusError` and catch everything.
"""


class OctopusError(Exception):
    """Base class for every error raised by Octopus."""


class ArmNotApplicableError(OctopusError):
    """Raised when an arm cannot run on the given data or task.

    Example: chi2 on a matrix containing negative values.

    Note: the runner CATCHES this and turns it into a skipped ArmResult when
    `on_error="skip"`. It only propagates when `on_error="raise"`.
    """


class MissingDependencyError(OctopusError):
    """Raised when an optional extra is required but not installed.

    TODO(you): the message must tell the user exactly what to run, e.g.
    'The shap arm requires the "shap" extra: pip install "octopus_fs[shap]"'.
    """


class InvalidConfigError(OctopusError):
    """Raised for contradictory configuration (e.g. rule='top_k' with k=None)."""

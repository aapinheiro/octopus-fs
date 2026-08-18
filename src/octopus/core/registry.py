"""Name -> arm class mapping.

Why a registry and not a big if/elif: `arms=["lasso", "shap"]` should work from
a YAML file, a CLI flag and a REST payload without any of them importing
classes. It also gives you a plugin seam later (entry points).
"""

from __future__ import annotations

from octopus.core.base import BaseArm

_REGISTRY: dict[str, type[BaseArm]] = {}


def register(cls: type[BaseArm]) -> type[BaseArm]:
    """Class decorator used by each arm module.

    TODO(you): raise on duplicate names — a silent overwrite is a nasty bug.
    Usage:

        @register
        class MutualInformationArm(BaseArm):
            name = "mutual_info"
    """
    raise NotImplementedError


def get_arm(name: str, **params: object) -> BaseArm:
    """Instantiate an arm by name.

    TODO(you): on unknown name, raise with a did-you-mean suggestion
    (difflib.get_close_matches over available()). Small touch, big DX win.
    """
    raise NotImplementedError


def available() -> list[str]:
    """All registered arm names, sorted."""
    raise NotImplementedError


def resolve(spec: str | list[str]) -> list[str]:
    """Turn 'all' / 'fast' / ['lasso', 'shap'] into a concrete list of names.

    TODO(you): presets live in octopus.config.ARM_PRESETS. Validate every name
    here so a typo fails before any data is touched.
    """
    raise NotImplementedError


# TODO(you): importing octopus.arms must be what populates _REGISTRY.
# Do it in octopus/arms/__init__.py, not here, to avoid a circular import.

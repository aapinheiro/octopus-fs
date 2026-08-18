"""Validated configuration objects (pydantic v2).

Why pydantic and not plain dataclasses: these objects are also the thing you
serialize into the run manifest and the thing a YAML/CLI invocation is parsed
into. Free validation + free round-tripping.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# Named presets so `arms="fast"` means something stable and documented.
ARM_PRESETS: dict[str, list[str]] = {
    "fast": ["correlation", "chi2", "mutual_info"],
    "linear": ["correlation", "lasso"],
    "trees": ["tree_importance", "permutation", "shap"],
    "all": [
        "correlation", "chi2", "mutual_info", "lasso",
        "tree_importance", "rfe", "permutation", "shap",
    ],
}


class ArmConfig(BaseModel):
    """Per-arm overrides, e.g. ArmConfig(name="correlation", params={"method": "spearman"})."""

    name: str
    params: dict[str, object] = Field(default_factory=dict)
    enabled: bool = True


class RunConfig(BaseModel):
    """Everything that defines a run. Hash this and you have a cache key."""

    arms: list[ArmConfig]
    task: Literal["auto", "regression", "binary", "multiclass"] = "auto"
    random_state: int | None = 42
    n_jobs: int = 1
    on_error: Literal["skip", "raise"] = "skip"
    sample: int | None = None  # subsample rows for the expensive arms
    consensus_method: Literal["borda", "mean_rank", "rra"] = "borda"

    # TODO(you): add a `fingerprint()` returning a stable hash of this config.
    #   - dump with model_dump(mode="json"), sort keys, hash with sha256
    #   - it goes in the manifest AND becomes the cache key in v0.2
    # TODO(you): add a validator rejecting n_jobs=0 and sample<=0.

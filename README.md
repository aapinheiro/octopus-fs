# 🐙 octopus_fs

**One API. Eight arms. Eight feature selection techniques.**

Octopus wraps the eight most used feature selection techniques behind a single
high-level API, and exposes them in the two ways they are actually consumed:

1. **At scale, inside production pipelines** — as a deterministic, serializable,
   scikit-learn compatible transformer.
2. **In discovery** — as a self-contained HTML report with rankings, agreement
   between techniques and stability insights.

> Status: **design / scaffolding**. The skeletons are in place; the arms are not
> implemented yet.

---

## The eight arms

| key | technique | family | tasks | fits a model? | cost |
|---|---|---|---|---|---|
| `correlation` | Pearson / Spearman | filter | reg, binary clf | no | 💲 |
| `chi2` | Chi-squared | filter | clf, non-negative features | no | 💲 |
| `mutual_info` | Mutual Information | filter | reg, clf | no | 💲💲 |
| `lasso` | L1 regularization | embedded | reg, clf | yes (1 fit) | 💲💲 |
| `tree_importance` | Impurity-based importance | embedded | reg, clf | yes (1 fit) | 💲💲 |
| `rfe` | Recursive Feature Elimination | wrapper | reg, clf | yes (N fits) | 💲💲💲💲 |
| `permutation` | Permutation Importance | post-hoc | reg, clf | yes (1 fit + K shuffles) | 💲💲💲 |
| `shap` | SHAP values | post-hoc | reg, clf | yes (1 fit + explainer) | 💲💲💲 |

They are **not interchangeable** — `chi2` refuses negative values, `correlation`
is univariate and linear, `shap` needs a fitted model. Octopus does not hide
that: each arm declares what it supports and is skipped, loudly, when it
does not apply.

## Two faces, one core

```python
# --- Discovery: explore and explain -------------------------------------
from octopus_fs import Octopus

result = Octopus(arms="all", task="auto", random_state=42).fit(X, y)

result.consensus.top(20)          # aggregated ranking across the arms
result.to_html("reports/churn.html")   # dashboard: per-arm ranks, agreement, stability
```

```python
# --- Production: select and move on -------------------------------------
from sklearn.pipeline import Pipeline
from octopus_fs.pipeline import OctopusSelector

pipe = Pipeline([
    ("select", OctopusSelector(arms=["mutual_info", "lasso", "tree_importance"],
                               rule="top_k", k=30, random_state=42)),
    ("model", model),
])
pipe.fit(X_train, y_train)        # selection happens INSIDE the fold — no leakage
```

Same engine underneath. The production face just drops the plotting layer and
emits a **run manifest** (input schema hash, library versions, seeds, per-arm
params) so a selection can be audited and reproduced later.

## The head: consensus

Eight rankings are eight opinions. Octopus aggregates them with rank-based
methods (Borda count, mean reciprocal rank, robust rank aggregation) and reports
**stability** — how much the ranking changes across bootstrap resamples. A
feature that only the SHAP arm loves is a different animal from one all eight
agree on, and the report says which is which.

## Design principles

1. **Arms return scores, never decisions.** Thresholding is a separate,
   explicit step.
2. **Selection is part of the model.** It belongs inside cross-validation folds.
3. **Degrade loudly.** An arm that cannot run is `status="skipped"` with a
   reason, never a silent zero.
4. **Cheap by default.** `arms="fast"` runs only the filters.
5. **Reproducible.** Same data + same seed + same versions ⇒ same selection.

## Install

```bash
pip install octopus_fs            # core: filters + embedded arms
pip install "octopus_fs[report]"  # + HTML dashboards
pip install "octopus_fs[all]"     # + shap, lightgbm
```

## Development

```bash
uv sync --all-extras      # env + dev deps
uv run pre-commit install
make test                 # pytest + coverage
make lint typecheck
make build                # dist/*.whl
```

## Layout

```
src/octopus_fs/
├── types.py         # ArmResult, SelectionResult, TaskType — the shared contract
├── config.py        # validated run configuration
├── core/            # base arm, registry, validation, orchestration
├── arms/            # the eight techniques, one file each
├── consensus/       # rank aggregation + stability
├── pipeline/        # sklearn transformer + run manifest
├── report/          # HTML dashboard (Jinja2 + Plotly)
└── cli.py           # octopus_fs run --data ... --target ...
```

## Roadmap

- [ ] v0.1 — five filter/embedded arms, Borda consensus, minimal HTML report
- [ ] v0.2 — the model-fitting arms (rfe, permutation, shap) + caching
- [ ] v0.3 — `SelectionCV`: nested selection with per-fold stability
- [ ] v0.4 — backend protocol (pandas → polars → Spark for wide tables)

## License

MIT — see [LICENSE](LICENSE).

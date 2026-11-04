# Code map and provenance

| Candidate | Purpose / research source |
|---|---|
| `scripts/run_experiment.py` | New thin CLI: dry-run by default, explicit frozen fit/score/evaluate execution |
| `scripts/inspect_results.py` | New read-only saved-metric/shape/finite-value/hash inspection, no evaluation |
| `src/ha_paano/configuration.py` | New exact config and filename-set checks; scoped environment binding |
| `src/ha_paano/runner.py` | New explicit U/M binding and fresh-output guard/receipt |
| `_reference/pipeline.py` | Relocated `main.py`, same AnomalyDetection implementation |
| `_reference/model_baseline.py` | Shared PaAno-Re backbone and heads |
| `_reference/model_FINAL_HA_PaAno.py` | LSDP and historical constructor dependency, NOT the standalone final model |
| `_reference/model_FINAL_HA_PaAno_CANONICAL.py` | Revised raw-statistics CAC definition used by U transfer |
| `_reference/cross_branch_model.py` | Actual final U-JLC, C=1 CAC and pre-Jitter raw context queue |
| `_reference/cross_branch_train.py` | Frozen Jitter injection around unchanged U trainer |
| `_reference/sensitivity_train_u.py` | Exact jitter-only augmentation function |
| `_reference/train_FINAL_U_branch.py` | Original U objective/optimizer/sampling; its historical default augmenter is ALWAYS replaced by the final entry |
| `_reference/cac_variants.py` | Actual accepted M `no_relative_norm` route; other definitions retained as private historical dependencies, not public model options |
| `_reference/train_FINAL_M_branch.py` | Original no-augmentation PaAno trainer |
| `_reference/utils/data_preprocess.py` | Original CSV parsing/prefix split, patches/loaders and ACF window |
| `_reference/utils/utils.py` | Original RevIN1d and memory-bank selection |
| `_reference/utils/evaluation.py` | Original top-k cosine patch scoring and overlap-mean point mapping |
| `_reference/utils/metrics.py`, `basic_metrics.py` | Retained TSB-AD-derived evaluator |
| `_reference/affiliation/` | Required MIT affiliation metric helpers |
| `_reference/utils/result_saver.py` | Original result saver; writes only to new run output |
| `_reference/utils/cac_activity_logger.py` | Import dependency, disabled in public runner; not presented as a validated final-CAC visualization API |
| `tests/test_core.py` | Synthetic CPU tests, no dataset downloads |

`_reference/` paths above are relative to `src/ha_paano/`. This structure intentionally avoids rewriting/merging model classes. Internal legacy functions are not advertised as supported final execution routes; unbound pipeline execution fails closed.

`SOURCE_PROVENANCE.json` contains every copied source path, source SHA256, candidate SHA256, executable-AST check and unified import-relocation diff. Original source tree absolute paths and audit machinery are excluded from publication. The only runtime changes in added wrappers are explicit final binding, portable path/device/output configuration and validation; copied numerical bodies remain unchanged.

The original `utils/utils.py::load_model()` is not a working standalone API (undefined Model). It remains an unused inherited helper; no command invokes it or claims checkpoint-only evaluation. The original first-series parameter-count metadata behavior also remains; see the reproducibility limitations.

## R2 extension

| Candidate | Role / frozen origin |
|---|---|
| `scripts/run_r2_experiment.py` | Dry-run; explicit fresh condition/seed child; list-only 51+3 jobs |
| `src/ha_paano/r2_configuration.py` | Exact split/config/hash checks and scale expansion |
| `src/ha_paano/r2_routes.py` | Original 16-route map and environment binding, no merged constructors |
| `src/ha_paano/r2_runner.py` | Original worker input/probe/trainer/bank/scoring order, portable fresh output |
| `src/ha_paano/r2_controls.py` | Fixed 8D Mean-only pre-MLP hook, state/RNG and post-best LSDP inspection |
| `_reference/train_baseline.py` | Frozen genuine baseline trainer dependency |
| `_reference/factorial_model.py` | Six missing switches; only source path injection/import plumbing adapted |
| `_reference/sensitivity_model_u.py` | Original U-JL construction, including repeated constant head |
| `_reference/urpa_transform_train.py` | Original U-J route, forced jitter-only by the R2 environment |
| `configs/r2_*.json` | Sanitized scientific/member/thread policies, no absolute data paths |
| `tests/test_r2.py`, `tests/r2_probe.py` | Synthetic CPU regression/probe; optional copied frozen-fixture comparison |

Paths beginning `_reference/` are relative to `src/ha_paano/`. Pre-R2 metadata/audit snapshots under `docs/history/pre_r2_20261104/` retain their original facts and are not validation targets for today's edited files. The original final numerical modules and entry remain byte-identical; R2's output layout is separate from the original final-entry summary/NPZ format.

# HA-PaAno local release candidate — execution report

Historical scope: counts and tests below describe the initial pre-R2 preparation. Current extension identities and validation are in `PUBLISHABLE_FILES.json`, `SOURCE_PROVENANCE.json` and `VALIDATION_RESULTS.json`; old metadata is archived under `history/pre_r2_20261104/`.

## Outcome

An actual runnable candidate has been assembled in the independent HA-PaAno directory, not just a proposed tree. The original PaAno research project is unchanged within the recorded verification scope. This is a local candidate for human review, not an uploaded or production-validated release.

## Delivered structure

```text
HA-PaAno/
  README.md
  LICENSE
  CITATION.cff
  THIRD_PARTY_NOTICES.md
  requirements.txt
  .gitignore
  configs/
    final_U.json
    final_M.json
    evaluation_U.json
    evaluation_M.json
  src/ha_paano/
    __init__.py
    binding.py
    configuration.py
    runner.py
    _reference/
      pipeline.py
      model_baseline.py
      model_FINAL_HA_PaAno.py
      model_FINAL_HA_PaAno_CANONICAL.py
      cross_branch_model.py
      cross_branch_train.py
      sensitivity_train_u.py
      cac_variants.py
      train_FINAL_U_branch.py
      train_FINAL_M_branch.py
      utils/{data_preprocess,utils,evaluation,metrics,basic_metrics,result_saver,cac_activity_logger}.py
      affiliation/{generics,metrics,_affiliation_zone,_integral_interval,_single_ground_truth_event}.py
      package initializers
  scripts/{run_experiment,inspect_results}.py
  tests/test_core.py
  licenses/{PaAno-MIT,TSB-AD-Apache-2.0,affiliation-MIT}.txt
  results/README.md
  docs/
    RELEASE_AUDIT.md
    FILE_SELECTION_PLAN.md
    INSTALLATION.md
    DATA_PREPARATION.md
    REPRODUCIBILITY.md
    CODE_MAP.md
    RELEASE_CHECKLIST.md
    RELEASE_REPORT.md
    SOURCE_PROVENANCE.json
    VALIDATION_RESULTS.json
    ORIGINAL_TREE_VERIFICATION.json
    SAFETY_VERIFICATION.json
    PUBLISHABLE_FILES.json
  .release_audit/    PRIVATE, GIT-IGNORED: inventories, local reference fixture, tools, toy outputs
  .git/              LOCAL ONLY: unborn codex/release-candidate branch
```

PUBLISHABLE_FILES.json is the authoritative complete publishable path/size/SHA256 listing. Its own hash is excluded to avoid recursive self-hashing; its count includes itself. The private audit folder must never be uploaded. No fake examples/empty framework or copied external-baseline environment is added.

## Audit and file decisions

The before/after source inventory contains 164,148 regular files: 12,881 source/launcher files, 25,799 configuration/metadata files, 8,956 results/tables, 50,510 score/intermediate arrays, 11,698 logs, 10,001 caches, 1,674 data files, 49 checkpoints, 565 figures/documents, 834 documents/licenses, 33,780 environment files, 30 Git metadata files and 7,371 others. Categories are a coarse first pass, not all unique experiments. Actual import/call/freeze inspection, not filenames alone, selected the dependency closure.

Kept 22 required research source modules, original requirements, PaAno/TSB-AD license texts and final-protocol identifiers. Added a portable wrapper, exact configs, minimal CPU tests and English documentation. Excluded signals, frozen results, NPZ/NPY, weights, logs, caches, figures/manuscripts, external baselines, original Git history and private authorization evidence. No original file was moved.

## Actual changes and scientific boundaries

Inherited modules were relocated, not merged or redesigned. Imports use package-relative paths; one commented private path was removed. The public diff redacts that removed private comment rather than republishing it. Source and candidate hashes are retained, and executable AST excluding imports is identical for all 22 copied modules.

The final bindings are U Jitter+LSDP+revised CAC with C=1 raw context, and M average pooling+revised CAC without relative normalization. Historical default model and augmenter functions remain only as private dependencies; public unbound execution fails closed. M constructor order and U raw-context hook order are preserved. No AMP/C1/C2 optimization, model/loss/optimizer/scheduler/data/sampling change is made.

The wrapper centralizes paths, device, seed and output root. It records input/config hashes and refuses existing outputs. The original scorer evaluates the full series including prefix; that potentially important benchmark characteristic is documented instead of silently changed. The public runner is not a detached resource-guarded queue or resume controller. It cannot overwrite old runs and does not automatically launch any seed grid.

## Commands actually executed and results

All test execution working directories were the new candidate, using the existing read-only Python3.12.3 research interpreter with `-B -X utf8`. No environment install occurred.

| Command / action | Actual result |
|---|---|
| Private `source_snapshot.py before` | 164,148 files, 18,386 selected SHA256, source Git/index state recorded |
| Private `analyze_source.py` | Source call/import/license/privacy audit; 16 frozen source hashes match |
| Private `build_candidate.py` | 22 research modules relocated; executable AST except imports unchanged; no data copied |
| `python -B -m unittest discover -s tests -v` | 11 tests PASS; no CUDA initialization |
| `python -B scripts/run_experiment.py --help` | Actual CLI matches documented options |
| `python -B scripts/inspect_results.py --help` | Actual stored-value reader; no fit/evaluation |
| Private `equivalence_probe.py reference U/M` and `candidate U/M` | Separate processes against local byte-copy fixture; exact equality of initialization, forward, loss observations, gradients, six updates, AdamW state, RNG, scores and metrics |
| Private `validate_candidate.py` | 32 Python files parse; 25 source/license/requirements hashes match; real formal U350/M180 dry runs have no writes/training; U/M synthetic fit/bank/score/evaluate/save PASS |
| Private `source_snapshot.py after`, `compare_original.py` | No path/size/mtime changes across 164,148 files; no selected hash changes across 18,386 files; source Git HEAD/branch/status/diffs/index identical |
| New-directory `git init --initial-branch=codex/release-candidate` | Local unborn branch only; zero commits; no stage/remote/upload |
| `git ls-files --others --exclude-standard`, `git check-ignore` | Actual publishable scope inspected; private fixture/inventories ignored |

Synthetic joint pipeline checks use only 512 toy points, 6 iterations and batch 8. Those settings bound tests and do not replace the frozen 100-iteration/batch-512 configurations. Toy metrics are not paper results and are not included as experiment tables. Both reference probes include an active pretext phase. Small exact CPU equality is not full CUDA reproducibility proof.

Inherited affiliation docstring escape warnings and occasional undefined-precision toy warnings were observed, without changing metrics to hide them. Final wrapper binding/output and stored-result inspection are unit-tested without executing a full training job. Final safety scan/file enumeration details are machine-readable in SAFETY_VERIFICATION.json.

## Licenses and author confirmation

The user explicitly confirmed HA-PaAno additions as MIT, Copyright (c) 2026 Peng Zhang; software citation author Peng Zhang only. PaAno's separate MIT, TSB-AD evaluator Apache-2.0 and affiliation MIT notices are preserved. The RevIN file has verifiable PaAno MIT provenance and an existing CATCH attribution; the maintainer also confirmed redistribution authorization and public-source permission. This is not a claim of independent legal clearance, no extra contractual conditions, or a blanket CATCH license. Keep any rights evidence private and comply with its conditions before publication. No personal contact/email/ORCID, DOI or acceptance status is invented.

## Remaining unverified matters

1. Clean-environment reinstall, CUDA execution, full 350/180 × ten-seed reproduction, machine resource thresholds and cross-platform exactness were not run.
2. Historical metric-only jobs do not supply all model checkpoints/point scores. No claim of complete artifact recovery or standalone checkpoint-only evaluation is made.
3. Original per-seed parameter metadata caches its first series' count; that is not a reliable varying-C per-series complexity measurement. The unused inherited load_model helper is not a supported API. CAC activity logging remains unexposed/unvalidated for final visualizations.
4. Dataset signal rights remain with their respective providers; identifiers are not permission to publish the signals. Independent data-byte matching against historical full data was not certified.
5. Before actual publication, the maintainer must review the exact publishable manifest, authorization-specific terms and paper metadata. Remote publication requires new explicit approval. This task is complete as a local candidate, not as a legal or full-experiment certification.

## Reader commands

Minimum test, from this directory with dependencies already installed:

```powershell
python -B -m unittest discover -s tests -v
```

Formal preflight (no training/writes), with authorized data:

```powershell
python -B scripts/run_experiment.py --track U --seed 2000 --data-dir data/TSB-AD-U-Eva-350 --output-dir outputs/U_seed2000
python -B scripts/run_experiment.py --track M --seed 2000 --data-dir data/TSB-AD-M-Eva-180 --output-dir outputs/M_seed2000
```

Only after a separate decision to run, add `--device cuda --execute` (or `--device cpu --execute`). Those full-run commands were NOT executed here. Choose a fresh output path and run seeds serially; repeat intentionally for seeds2000–2009 if a full replication is desired. All numerical source provenance and protection limitations remain as stated above.

**Official experimental data modified: NO. Original model/training/evaluator source modified: NO. GPU training started: NO. Remote created/push/upload/commit: NO.**

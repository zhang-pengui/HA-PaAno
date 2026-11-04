# HA-PaAno release-source audit

This document describes the local candidate, not a public release or an assertion of publication acceptance.

## Scope and protection

The research source is read-only. An initial inventory records 164,148 files and 18,386 selected source/configuration/license/summary hashes. Source Git HEAD is `d4c67116190efa4592dc6a8a157ced0def68b6af`; existing tracked and untracked changes belong to the researcher and are not reset. Detailed paths, original Git state and private inventory are held only in the excluded `.release_audit/` folder. An end-of-task comparison is required.

## Actual final implementation, not historical defaults

| Component | Research source | Role |
|---|---|---|
| Backbone | `model_baseline.py` | PaAno-Re CNN, projection/classification heads, RevIN integration |
| LSDP | `model_FINAL_HA_PaAno.py` | 64-dimensional learnable average/maximum mixture; also legacy constructor dependency |
| Revised CAC | `model_FINAL_HA_PaAno_CANONICAL.py` | Raw pre-RevIN standard deviation and absolute first-difference statistics, no relative normalization |
| Final U-JLC | `revision_experiments/cross_branch_full_pilot_3seed/scripts/cross_branch_model.py` | C=1 CAC enabled; queued pre-Jitter raw context |
| Final U trainer | same directory, `cross_branch_train.py`; `train_FINAL_U_branch.py`; sensitivity `sensitivity_train_u.py` | Frozen Jitter hook; original training objective and sampling |
| Final M-C | `internal_ablation_seed2000/models/cac_variants.py`, variant `no_relative_norm` | Average pooling plus revised CAC; retains original constructor/RNG order |
| M trainer | `train_FINAL_M_branch.py` | Identical bytes to `train_baseline.py`; no augmentation |
| Fit and score orchestration | `main.py` | Per-series fitting, training-prefix bank, full-series scoring, metrics |
| Data/scoring/metrics | `utils/` and `affiliation/` | Original parsing, patches, memory bank, score mapping and evaluator |

Sixteen current source hashes match the frozen cross-branch runner-start record. Final U's ten-seed extension uses the same U-JLC entry as its three-seed pilot. Final M uses the accepted no-relative-normalization route. The inspected latest manuscript agrees with these final paths.

The root default `model.py` and historical `U_final`/relative-normalization CAC are NOT selected as final model definitions. Historical main defaults (patch 64, 200 iterations, RevIN off) are NOT final protocol defaults. The packaged wrapper must bind explicit final configurations.

## Scientific protocol

- Patch length 96, stride 1, 100 training iterations, batch 512, AdamW at 1e-4, weight decay 1e-4; cosine schedule toward 1e-5.
- RevIN on, affine off, epsilon/floor 1e-5; 64-dimensional embedding.
- U: Jitter probability 0.50, local-standard-deviation noise multiplier 0.010; LSDP alpha initialized to 0.05; revised CAC gate initialized to 0.10. No scale or shift augmentation in final U-JLC.
- M: average pooling, revised CAC without relative normalization, no training augmentation. The legacy CAC object is constructed and replaced as in the actual frozen route; simplifying that order would change RNG consumption.
- PaAno's triplet and pretext objectives, temporal/hard-negative sampling, checkpoint rule, memory-bank selection and top-k=3 cosine score remain foundations, not HA-PaAno inventions.
- Separate model fitted for each series. Prefix boundary comes from `tr_*` in the filename. The frozen scorer evaluates the FULL series, including the training prefix; it is not a held-out-suffix-only protocol. This characteristic must be disclosed, not silently changed.
- Evaluation: U 350 files, M 180 files; core seeds 2000–2009. Formal seed statistics use sample standard deviation (ddof=1). PaAno-Re is not the originally reported PaAno result.

## Data and dependencies

The loader reads CSVs with features followed by a label column; labels and anomaly intervals must come from the authorized TSB-AD distribution. No dataset or download automation is included. The matched-subset project is a separate pairing/statistics resource, NOT the full model repository. PyTorch, NumPy, pandas, SciPy, scikit-learn, statsmodels and tqdm are actual runtime imports; Matplotlib is retained where inherited modules import it. Versions are taken from the original requirements and inspected environment rather than upgraded.

## Selection and exclusions

Only dependency-closure source files are to be relocated. Exclude datasets, NPZ/NPY scores, weights, results, full logs, manuscript/figure assets, external-baseline repositories/environments, old launch controllers, cache, personal paths and private authorization evidence. Do not copy the source Git history.

## Security and rights findings

Selected source scan found no credential literal. Potential key/certificate filenames occur only in excluded third-party environments; filename matches alone are not credential findings. A commented private data path in the loader will be removed from the candidate. Experiment controllers contain local absolute paths and will not be copied.

PaAno's original MIT notice is retained. The inherited `utils/utils.py`, including RevIN1d, is unchanged from the pinned PaAno commit. Its comment credits CATCH; CATCH's inspected repository did not expose a root license. This is a provenance-chain question, not proof of infringement, and must not be described as a demonstrated lack of permission for the PaAno file. The researcher subsequently explicitly confirmed redistribution authorization and permission for public source redistribution. Attribution is retained; private authorization evidence is excluded. Compliance with any authorization-specific conditions remains the maintainer's responsibility, not an independent legal-clearance claim.

Evaluator code credits TSB-AD (Apache-2.0). Affiliation metrics credits Alexis Huet and others (MIT). Their license texts and attribution are retained; the new project license does not replace them. The researcher explicitly confirmed `Copyright (c) 2026 Peng Zhang, MIT` for HA-PaAno additions and sole software author Peng Zhang for CITATION.cff. No authors were inferred from the paper, and the separate matched-subset license was not generalized.

## Known limitations / candidate acceptance

Many final experiments retained metrics only; historical checkpoints and point scores cannot all be reconstructed from retained artifacts. No assertion is made that all historical artifacts are bundled. Original `load_model()` references an undefined Model and is not a supported checkpoint-only evaluation API. The reliable final flow fits, builds a training-prefix bank, scores and evaluates in one job. The candidate must explain this rather than invent a standalone checkpoint workflow.

The completed bounded CPU equivalence, formal dry runs and original-tree comparison are recorded in `VALIDATION_RESULTS.json`, `ORIGINAL_TREE_VERIFICATION.json` and `RELEASE_REPORT.md`. Full ten-seed/GPU reproduction is outside this task and remains NOT VERIFIED. Publication still requires human approval and compliance with third-party/data terms; no remote operation is authorized or performed.

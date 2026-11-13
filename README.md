# HA-PaAno

Official implementation and reproducibility code for **HA-PaAno: Raw-Statistics-Aware Patch Representation Refinement for Univariate and Multivariate Time-Series Anomaly Detection**.

The source code is publicly available in this repository. The manuscript is currently under revision at Applied Intelligence. No acceptance, DOI or publication details are claimed; no GitHub Release or tag has been created.

HA-PaAno studies how raw statistical context and temporal refinements complement normalized patch representations within PaAno. It retains PaAno's learning objectives and memory-bank anomaly scoring. The paper reports both CAC-only (C-only) and JLC on U Eva-350 and C-only on M Eva-180. U-JLC was selected after examining the Eva-350 module results. On U Tuning-48, C-only ranked first on average, but its lead was small relative to variation across seeds and it did not rank first for any individual seed.

## Final pathways

| Setting | Training augmentation | Pooling | Context refinement |
|---|---|---|---|
| Univariate (U) | Jitter only | LSDP | Revised CAC, enabled at C=1 |
| Multivariate (M) | None | Average | Revised CAC without relative normalization |

- **Jitter** uses the frozen probability 0.50 and noise multiplier 0.010 of the local population standard deviation. It is training-only. Scale/shift are not part of the final U-JLC pathway.
- **Learnable Statistical Dual Pooling (LSDP)** mixes average and maximum pooling in a 64-dimensional embedding. Its per-coordinate mixing coefficients are trainable and shared across patches within each fitted model. In the U Tuning-48 JLC runs, the fitted coefficients stayed close to their initial value of 0.05, yielding approximately 95% average pooling and 5% maximum pooling. Mixed pooling is an existing mathematical family.
- **Context-Aware Calibration (CAC)** maps eight statistics of raw pre-RevIN channel dispersion and first-difference activity to a patch-dependent residual. A sigmoid-parameterized scalar scales that residual; the scalar is shared across patches within each fitted model and is not an input-dependent gate. Its effective value is initialized to 0.10 and averaged 0.10003088 in the archived full-C M Tuning-20 checkpoints. The small change in the scalar does not establish how much the CAC MLP learned. In U training, CAC uses the original pre-jitter patch. Its C=1 descriptor has repeated entries; it is not an attention map or a channel-identity-preserving dependency model.
- **RevIN** belongs to the reproduction backbone, not the HA-PaAno contribution.

Historical root defaults and older relative-normalization CAC are not the final bindings. The documented entry points use internal modules included in this repository under `src/ha_paano/_reference/`. The modules are kept separate to preserve initialization and random-number consumption; the entry points do not require the original research checkout.

## Reproducibility scope

| Public entry point | Supported experiments |
|---|---|
| [`scripts/run_experiment.py`](scripts/run_experiment.py) | U-JLC on U Eva-350 and M-C on M Eva-180, one seed per invocation (2000–2009). |
| [`scripts/run_r2_experiment.py`](scripts/run_r2_experiment.py) | Eight module combinations on U Tuning-48 and M Tuning-20, plus M Mean-only CAC on M Tuning-20 and M Eva-180 (seeds 2000–2002). |

There are no direct public entry points for the standalone pooling controls (Table 10), the historical CAC input controls (Table 11, Panel A), the R3 Major 2 matched-constructor residual comparison (Table 11, Panel B), the full evaluation-split factorial, or the formal multi-seed PaAno-Re evaluation. The separate U C-only evaluation reported in the paper is not exposed by the final-model entry point. These limits do not affect the supported routes listed above.

## Installation

The initial candidate was checked with Python 3.12.3; R2 CPU regression uses Python 3.11.15. Runtime versions are pinned to the original research environment in `requirements.txt`, including PyTorch 2.7.1, NumPy 2.3.2 and scikit-learn 1.7.1. No packages were installed during candidate preparation. See [installation details](docs/INSTALLATION.md).

From the candidate root, in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

Environment creation/install commands are instructions for the reader, not actions already performed in this task. A clean-environment reinstall and full CUDA reproduction remain unverified. No `pip install -e .` step or packaging build system is required.

## Obtain and prepare data

Obtain TSB-AD data separately from the [official benchmark](https://github.com/TheDatumOrg/TSB-AD), respecting each source dataset's terms. No raw signals, labels, scores or weights are bundled or downloaded automatically.

Use a flat directory of authorized CSVs for each track. Each CSV contains feature columns followed by the label column, with a filename such as `..._tr_320_1st_400.csv`. The `tr_*` field defines the training-prefix boundary. Formal membership is checked against `configs/evaluation_U.json` (350) or `configs/evaluation_M.json` (180). These contain identifiers only, not signal data. See [data preparation](docs/DATA_PREPARATION.md).

The [TSB-AD U/M matched-subset resource](https://github.com/zhang-pengui/TSB-AD-UM-Matched-Subset) is a separate pairing/statistics project. It is not the full HA-PaAno training repository, and no availability/visibility claim about that repository is made here.

## Read-only preflight

The default is dry-run: it checks the actual parser, frozen configuration, file membership and CSV header shapes. It does not import PyTorch, create an output directory or start training. Substitute your authorized local dataset location:

```powershell
python -B scripts/run_experiment.py --track U --seed 2000 --data-dir data/TSB-AD-U-Eva-350 --output-dir outputs/U_seed2000
python -B scripts/run_experiment.py --track M --seed 2000 --data-dir data/TSB-AD-M-Eva-180 --output-dir outputs/M_seed2000
```

## Training, inference and evaluation

The retained research pipeline **fits a separate model per series, builds its training-prefix memory bank, then scores and evaluates that series in one job**. There is no verified standalone checkpoint-only evaluation entry, and no checkpoint/weights download. Do not treat this command as inference-only.

To explicitly execute one seed (potentially long-running), after reviewing dry-run output:

```powershell
python -B scripts/run_experiment.py --track U --seed 2000 --data-dir data/TSB-AD-U-Eva-350 --output-dir outputs/U_seed2000 --device cuda --execute
python -B scripts/run_experiment.py --track M --seed 2000 --data-dir data/TSB-AD-M-Eva-180 --output-dir outputs/M_seed2000 --device cuda --execute
```

CUDA is optional; CPU is the default and `--device cpu` hides CUDA from the inherited runtime checks in a fresh process. Run one job at a time. The wrapper is not a detached/resource-guarded research queue controller and does not promise crash-safe resume. It refuses existing output directories; it never resumes, deletes or overwrites a previous run. Diagnostic subsets require explicit `--scope diagnostic` and are not benchmark evidence.

Frozen settings: patch 96, stride 1, 100 iterations, batch 512, AdamW at 1e-4, cosine schedule, RevIN on, top-k=3, core seeds 2000–2009. One invocation runs one seed; there is no automatic ten-seed loop. See [reproducibility](docs/REPRODUCIBILITY.md) for scope and limits.

The reproduced triplet rule selects the legal off-diagonal positive projection with the **largest cosine distance** in the batch (farthest-batch negative), not the nearest hard negative.

**Protocol disclosure:** the inherited final scorer evaluates the full series, including its training prefix; fitting and bank construction use the prefix only. This is not a held-out-suffix-only protocol. Oracle-threshold F1 metrics are not deployment threshold-selection procedures.

## R2 combinations and Mean-only CAC

The independent `scripts/run_r2_experiment.py` supports **B, J, L, C, JL, JC, LC, JLC** on both U Tuning-48 and M Tuning-20, using seeds **2000–2002**. B is the genuine PaAno-Re baseline; J/L/C mean Jitter/LSDP/CAC. Scale/shift are disabled. Historical constructor paths and per-series seed resets are retained; the same scalar seed is **not** strict cross-condition common-random-number control.

Use flat, separately obtained `data/TSB-AD-U-Tuning/` and `data/TSB-AD-M-Tuning/` directories. R2 verifies exact members, sizes and SHA256 against `configs/r2_tuning_U.json`, `configs/r2_tuning_M.json` and `configs/r2_evaluation_M.json`. The same scientific p96/stride1/batch512/100-step settings are pinned in `configs/r2_protocol.json`. This is not the original entry's relaxed diagnostic mode.

Default commands only validate; they do not train, import Torch, spawn a worker or create output:

```powershell
python -B scripts/run_r2_experiment.py --list-jobs
python -B scripts/run_r2_experiment.py --track U --condition B --seed 2000 --split tuning --data-dir data/TSB-AD-U-Tuning --output-dir outputs/R2_U_B_seed2000
python -B scripts/run_r2_experiment.py --track M --condition C --seed 2000 --split tuning --data-dir data/TSB-AD-M-Tuning --output-dir outputs/R2_M_C_seed2000
```

All combinations and seeds can be checked with this **dry-run** loop. Add `--execute --device cuda` only when intentionally fitting each job; do not run GPU jobs concurrently.

```powershell
foreach ($condition in @('B','J','L','C','JL','JC','LC','JLC')) {
  foreach ($seed in @(2000,2001,2002)) {
    python -B scripts/run_r2_experiment.py --track U --condition $condition --seed $seed --split tuning --data-dir data/TSB-AD-U-Tuning --output-dir "outputs/R2_U_${condition}_seed${seed}"
    python -B scripts/run_r2_experiment.py --track M --condition $condition --seed $seed --split tuning --data-dir data/TSB-AD-M-Tuning --output-dir "outputs/R2_M_${condition}_seed${seed}"
  }
}
```

Mean-only is allowed only on **M-C**. It multiplies the complete finite eight-dimensional activity descriptor by `[1,0,0,0,1,0,0,0]` immediately before the original MLP. It retains activity means across channels, not raw temporal means. No two-dimensional replacement MLP, reinitialization, parameter registration or old std-only/diff-only variant is used. The same hook remains active in training, best-state restoration, bank construction and query scoring.

```powershell
python -B scripts/run_r2_experiment.py --track M --condition C --mean-only --seed 2000 --split tuning --data-dir data/TSB-AD-M-Tuning --output-dir outputs/R2_M20_MEAN_ONLY_seed2000
python -B scripts/run_r2_experiment.py --track M --condition C --mean-only --seed 2000 --split evaluation --data-dir data/TSB-AD-M-Eva-180 --output-dir outputs/R2_M180_MEAN_ONLY_seed2000
```

Repeat with seeds 2001 and 2002 and fresh output paths. Adding `--execute` explicitly fits, builds the bank, scores and evaluates in a fresh condition/seed child; CPU is the default, CUDA requires `--device cuda`. The parent waits and propagates its child's return code; there is no implicit retry or multi-condition interpreter. The original final-model entry is unchanged.

The tuning matrix plus M20 Mean-only is **51 jobs / 1692 fits**. M180 Mean-only adds **3 jobs / 540 fits**. `--list-jobs` only expands identities, not data validity or execution. M180 uses the archived seed-specific thread environment from `configs/r2_eva_execution.json`; tuning inherits native settings.

For example, this command intentionally executes one U-JLC tuning job after its dry run (it is not executed during release preparation):

```powershell
python -B scripts/run_r2_experiment.py --track U --condition JLC --seed 2000 --split tuning --data-dir data/TSB-AD-U-Tuning --output-dir outputs/R2_U_JLC_seed2000 --device cuda --execute
```

R2 writes a receipt and `series/<id>/{result.npz,best_model.pt,metrics.json,metadata.json,COMPLETE.json}`; L conditions also save `lsdp_weights.json`. Payloads are hashed and checkpoints are CPU-readback checked before COMPLETE. Outputs are fresh-only; failures retain partial files and stop. This is a portable serial entry, not the internal Windows controller, resource guard or crash-safe resume service. It never reuses historical metrics.

## Results and metrics

Original final-entry runs write `run_receipt.json`, `results/npz/<series>.npz`, `results/summary.csv` and `results/dataset_summary.json` under the requested fresh output root. The receipt records seed, pathway, config hash and input hashes. Those NPZs contain raw input and labels; R2 NPZs contain scores and labels. Neither is bundled or automatically published. No historical experiment results are copied into this candidate.

Six main output metrics are VUS-PR, VUS-ROC, Range-F1, AUC-PR, AUC-ROC and Point-F1. The retained evaluator also computes auxiliary metrics. Inspect original final-entry results without rerunning evaluation (this inspector is not the R2 directory parser):

```powershell
python -B scripts/inspect_results.py --run-dir outputs/U_seed2000
```

For paper-style core summaries, aggregate equal-weight series means per seed, then mean and **sample standard deviation (ddof=1)** across ten seeds. PaAno-Re denotes a local controlled reproduction, not the originally reported PaAno result. Do not report toy test metrics as paper performance.

## Verification and known limits

CPU tests cover both pathways, C=1 raw context, Jitter RNG, pooling, parsing, score alignment, metric evaluation and output. An isolated reference/candidate six-step CPU comparison matched initialization, forward, loss observations, gradients, optimizer states, updates, RNG and scores exactly. This bounded check is not proof of all-device/full-experiment bitwise reproducibility. Current bounded R2 checks are recorded in `docs/VALIDATION_RESULTS.json`; pre-R2 evidence is preserved under `docs/history/pre_r2_20261104/`. No complete 1692-fit/540-fit CUDA reproduction or clean reinstall was performed for this release repair.

Historical experiments often retained metrics only: not every checkpoint or point-score artifact survives. The inherited per-seed summary caches parameter count from its first series; with varying M channel count, do not treat that cached metadata as a per-series complexity measurement. Auxiliary historical CAC activity logger keys are retained for dependency closure but are not exposed as a validated final visualization feature.

## License, attribution and citation

HA-PaAno additions/packaging: MIT, Copyright (c) 2026 Peng Zhang, confirmed by the maintainer. PaAno MIT, TSB-AD evaluator Apache-2.0 and affiliation MIT notices remain applicable to their respective portions. See [third-party notices](THIRD_PARTY_NOTICES.md), `licenses/`, and [release checklist](docs/RELEASE_CHECKLIST.md). Data licensing is separate. The maintainer confirmed public redistribution authorization for the included RevIN1d; no independent legal clearance or wider CATCH relicensing is claimed.

`CITATION.cff` contains the explicitly confirmed software author only. No ORCID, DOI, acceptance or remote URL is invented. A complete source map, relocation diff and hash records are provided in [CODE_MAP](docs/CODE_MAP.md) and `docs/SOURCE_PROVENANCE.json`.

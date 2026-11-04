# Reproducibility scope

## Frozen configuration binding

`configs/final_U.json` binds the actual Jitter+LSDP+revised-CAC U pathway; `final_M.json` binds the accepted M average-pooling/no-relative-normalization CAC route. The public wrapper checks their exact contents against the reviewed protocol, rather than quietly allowing stale defaults. `--track`, `--seed`, `--data-dir`, `--output-dir`, `--device`, `--scope` and explicit `--execute` are the original final-entry options. There is no invented checkpoint-evaluation flag or parameter sweep.

Use the README U/M commands first without `--execute`; this is read-only. Adding `--execute` fits one independent model per series, constructs the prefix bank and performs full-series scoring/evaluation. This task did not execute those full commands. A run for ten seeds requires ten intentional serial invocations with distinct fresh output directories. Do not run concurrent GPU jobs on an unvalidated machine.

The original experiment drivers are not redistributed: they contain local paths, resume/process orchestration and metric-only writers. Saving NPZs here changes artifact retention, not the objective or scores; its I/O falls outside the inherited scientific runtime interval. Runtime equality with historical metric-only workers is not asserted.

## Randomness and construction order

The original run seeds Python, NumPy, Torch and CUDA. PatchCreator reseeds Torch per series; the initial `next(iter(train_loader))` precedes model construction and must not be removed. U raw-context queueing and temporary Jitter hooks preserve the original call order. M creates the historical module before replacing its CAC, preserving initialization RNG consumption. No cudnn deterministic option, precision mode, loss, optimizer, scheduler, batch size, sample ordering or memory-bank mathematics was changed.

The source model is retained as separate private modules because collapsing constructors or optimizing unused parts could alter RNG/parameter semantics. Only package imports and one commented private path were changed in inherited files. The import-excluded executable AST is identical for all 22 copied Python modules; file hashes and diffs are in SOURCE_PROVENANCE.json.

## Completed validation versus unverified claims

- CPU imports, model initialization/forward, U/M shapes, raw statistics, Jitter RNG, score alignment, metrics and saving were checked on synthetic data.
- U/M each had an exact reference/candidate comparison for six optimizer steps, including one active-pretext phase. Reference source was a private byte-copy fixture in this new directory, never imported from the protected original tree.
- Synthetic end-to-end joint jobs used 512 points, 6 iterations and batch 8 solely to bound the smoke test. These are not frozen-protocol results and not paper evidence.
- Formal CLI dry runs verified the actual 350/180 filename lists and shapes, with no training or output writes.
- Full 100-step, batch-512 ten-seed reproduction on real data/CUDA, production resource limits, crash-safe resume and cross-platform/clean-install reproducibility are NOT VERIFIED.

Exact CPU hashes for small probes do not imply cross-device bitwise identity. CUDA runtime validation must be separately authorized and performed before making a stronger reproducibility claim. No AMP, C1/C2 speed optimization or altered scientific path is deployed.

## Results and statistics

Future outputs are intentionally separate from the original experiment archive. Preserve receipts and inputs to audit any new run. Summarize the six common metrics by equal-weight series means within each seed, then ten-seed mean ± sample std (ddof=1). Do not confuse local PaAno-Re with the original published/reported PaAno. External three-seed baselines must not be labelled ten-seed results.

This candidate does not bundle historical experiments, main-table CSVs, checkpoints or qualitative scores. Some historical runs are metric-only; there is no promise of checkpoint-only recovery of those scores. The public result inspector reads stored values only and does not rerun the evaluator.

## Independent R2 execution

`scripts/run_r2_experiment.py` adds 16 U/M tuning routes and the M-C Mean-only control on M20/M180, with seeds2000–2002. It checks the exact filename set, bytes, SHA256, prefix/channel identity and scientific config pins. Dry-run does not write or train. Explicit execute runs one fresh condition/seed child, propagates failure and never resumes/overwrites. The original final entry and its numerical modules remain unchanged.

`r2_routes.py` ports the formal map, preserving genuine B, U-JL's repeated constant LSDP constructor, and M-C's historical canonical constructor. New model dependencies only adapt package imports; factorial path injection is replaced with a relative import. `r2_runner.construct_training_input` keeps the actual worker's per-series Python/NumPy/Torch/CUDA seed reset, unused prefix stats, loader probe, CPU constructor and device transfer. Original trainers restore the complete best-loss state including BN before bank/scoring. Do not merge constructors or remove the loader probe.

`r2_controls.install_mean_only` registers the fixed [1,0,0,0,1,0,0,0] pre-MLP hook on canonical M-C only. No registered state or constructor RNG changes; repeated installation and U/non-C use are rejected. The same hook remains across train, restored checkpoint, bank and query. Full C is unchanged when the hook is absent. Eight-dimensional input and the original MLP parameters remain intact.

Same seeds do not create strict cross-condition CRN, remove historical floating-point nondeterminism, or erase exploratory evaluation use/content aliases. U-JLC is the reported evaluation configuration, not the latest tuning winner. M180 follows its archived per-seed thread environment; tuning inherits native thread settings. No deterministic/precision/AMP/scoring change is introduced.

R2 saves per-series result/metrics/metadata, a complete best-restored checkpoint and payload hashes; L conditions retain their post-best 64 coefficients. Checkpoints are CPU-readback checked before COMPLETE. Partial failure files remain, no automatic retry occurs. Current tests are CPU/synthetic and bounded; source equivalence cannot be extrapolated to full CUDA or clean-environment reproducibility.

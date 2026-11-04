# Reviewed file-selection and relocation plan

Audit completed before copying the dependency closure. Private source inventory remains outside the publishable set.

1. Relocate the actual final U-JLC and M-C dependency closure into `src/ha_paano/_reference/`. Keep separate source modules to preserve their execution and initialization order. Adjust package imports only; remove one commented private path. Record source/candidate SHA256 and unified diffs.
2. Keep the original joint fit/score/evaluate orchestrator and saver/logger dependencies. Bind model and trainer explicitly through a small public wrapper, with frozen JSON configs and fresh output directories. Avoid copied research controllers, implicit old defaults, downloads or automatic full-grid execution.
3. Vendor only required metric modules with TSB-AD and affiliation notices. Retain the PaAno MIT grant independently of any new-code licensing decision.
4. Add CPU-only tests for import, U/M construction, final descriptors, augmentation/raw-context behavior, data layout, score mapping, output, evaluator and reference/candidate comparisons. Never import/run the original tree. Reference comparisons use an excluded local byte-copy fixture, not public duplicate source.
5. Add English documentation and exact tested commands; distinguish completed smoke tests from untested long GPU reproduction. No data, experiment scores, weights, private evidence or manuscript assets are distributed.
6. Create an explicit publishable file manifest, verify .gitignore against actual Git inclusion, compare source before/after, and stop without committing or contacting a remote.

Completion update: the dependency closure, wrappers, configurations, documentation and CPU tests have been implemented locally. The user confirmed the new-code MIT author/year, sole software citation author, and RevIN public-source redistribution permission. Original third-party notices remain separate. No private evidence, experiment signals/results or source Git history is part of the publishable set. See RELEASE_REPORT.md for completed evidence and unverified/full-GPU limits. Human approval before any remote publication remains required.

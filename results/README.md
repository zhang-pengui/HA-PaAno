# Results policy

No research dataset, frozen results, score arrays, checkpoints, logs or paper figures are included here. Original assets remain in the protected research project and were not modified. Necessary identifier manifests and source/config provenance are small text files under `configs/` and `docs/`.

Future explicitly authorized runs produce outputs in a fresh user-selected directory, normally `outputs/` (Git-ignored). NPZs contain raw input and labels, so do not add them to a public repository without a separate data-rights and privacy review. Toy CPU test outputs are private diagnostic files, not experiment results or benchmark claims.

The public result inspector checks stored metrics and scores without recomputation. Historical checkpoints/scores are not guaranteed to exist for metric-only final runs. Reproduction means rerunning the documented model/protocol with authorized data, not pretending absent weights are recoverable.

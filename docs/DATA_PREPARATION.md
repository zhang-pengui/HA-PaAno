# Data preparation and protocol boundaries

Acquire authorized TSB-AD data from https://github.com/TheDatumOrg/TSB-AD and follow the original sources' conditions. The evaluator's software license does not establish rights to redistribute every signal. This candidate contains neither signal data nor a downloader.

Expected user-supplied layout:

```text
data/
  TSB-AD-U-Eva-350/
    <original series filename>.csv
  TSB-AD-M-Eva-180/
    <original series filename>.csv
```

One flat directory per track; files are processed in sorted filename order. Preserve original filenames, columns, row order and values. The last column is the anomaly label; preceding columns are features. U requires one feature, M more than one. The `_tr_<boundary>_` token defines the prefix used for training and memory-bank construction. A CSV must contain a header row. The dry run checks membership, header/first-row shape and a usable filename boundary; it is not a full numerical/content validation of all rows.

The two identifier manifests were generated from the inspected frozen evaluation directories, with exact counts 350 and 180. The public wrapper refuses a different formal filename set; `--scope diagnostic` is explicit and non-benchmark. Filename identity does not establish byte identity to an independently downloaded data distribution. On execution, input SHA256 values are recorded in the local receipt; the historical signal bytes are not shipped for comparison.

RevIN is enabled for the final protocol. The original main does not additionally apply prefix z-score normalization when RevIN is enabled. Patch length=96, stride=1; all overlapping windows are retained. CAC receives raw pre-RevIN context, and final U training captures context before Jitter. No preprocessing/downsampling/interpolation is added.

The held-out suffix is not the only scored region in the retained protocol: train and suffix are concatenated back into the full series for score mapping and evaluation. Do not silently crop to the suffix and claim the same benchmark protocol. The threshold-independent metrics use the inherited evaluator and ACF-derived window; Point-F1/Range-F1 follow its oracle-threshold procedure.

The separate matched-subset project is useful for strict source/instance pairing statistics, not a replacement for full evaluation data or this model's training code. Its source permissions and licenses do not automatically authorize this complete model project or signal redistribution.

## R2 exact subsets

The R2 entry uses `configs/r2_tuning_U.json` (48), `r2_tuning_M.json` (20) and `r2_evaluation_M.json` (180). They contain identifiers, domain/source metadata, file sizes and SHA256 from accepted formal manifests, not signals or labels. Supply flat `data/TSB-AD-U-Tuning/`, `data/TSB-AD-M-Tuning/` and the existing M Eva-180 directory. Preserve every byte, including headers/line endings. If an independently obtained distribution differs, resolve the identity mismatch rather than bypassing checks or relabeling files. Tuning cannot be replaced with diagnostic scope; M180 Mean-only is not a new independent selection set.

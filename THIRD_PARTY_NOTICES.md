# Third-party code and acknowledgements

The root MIT license applies only to original additions/packaging that Peng Zhang is entitled to license. It does not relicense third-party material or datasets.

## PaAno foundation — MIT

Source: https://github.com/jinnnju/PaAno

Inspected source Git commit: `d4c67116190efa4592dc6a8a157ced0def68b6af`. Preserve `licenses/PaAno-MIT.txt`, Copyright (c) 2026 jinnnju. The PaAno patch framework, learning objectives, local-positive sampling and farthest-batch negative construction (maximum cosine distance among legal off-diagonal positive projections, not the nearest hard negative), pretext task and memory-bank anomaly scoring are inherited foundations. The reproduction backbone also includes RevIN; it is not an HA-PaAno innovation.

The relocated research modules have a mixed PaAno/HA-PaAno origin, detailed file by file in `docs/SOURCE_PROVENANCE.json`. Modifications for this candidate are import relocation, entry binding and omission of one private commented example path, not new mathematical model definitions.

## RevIN provenance

The inherited `utils/utils.py` is unchanged from the pinned PaAno source before packaging. Its original comment credits CATCH:
https://github.com/decisionintelligence/CATCH/blob/master/ts_benchmark/baselines/catch/layers/RevIN.py

Keep this attribution. No claim is made that CATCH itself offers an MIT grant; the inspected root had no recognized license. The project maintainer confirmed redistribution authorization for the included RevIN1d implementation and explicitly confirmed that public source redistribution is permitted. No private authorization document is distributed. The maintainer remains responsible for complying with any additional authorization conditions; no independent legal clearance or blanket CATCH relicensing is asserted here.

## TSB-AD evaluator and sliding-window helper — Apache-2.0

Source: https://github.com/TheDatumOrg/TSB-AD

The research source credits `TSB_AD/evaluation/metrics.py`, `basic_metrics.py` and `TSB_AD/utils/slidingWindows.py`. Relocated files: `_reference/utils/metrics.py`, `basic_metrics.py` and the ACF helper in `data_preprocess.py`. Preserve source citations and `licenses/TSB-AD-Apache-2.0.txt`. Local modifications relative to research files are limited to package imports and a private commented path; original research adaptations are retained, not claimed to be verbatim upstream releases. An exact upstream evaluator commit for those historical adaptations was not retained and is not invented.

The Apache license on evaluation software does NOT grant redistribution rights for all benchmark signals. No dataset signals, labels, scores or checkpoints are bundled.

## Affiliation metrics — MIT

Source: https://github.com/ahstat/affiliation-metrics-py

License source: https://github.com/ahstat/affiliation-metrics-py/blob/main/LICENSE

Copyright (c) 2022 Alexis Huet and others; see `licenses/affiliation-MIT.txt`. Five actual metric modules are relocated under `_reference/affiliation/`. Original research adaptations, including output dictionary key names, are preserved. A proper package initializer is added; no metric formula is changed. The original root contained a misnamed initializer (`__init__ .py`); that filename is not propagated.

## Installed dependencies and data

PyTorch, NumPy, pandas, scikit-learn, SciPy, statsmodels, Matplotlib and tqdm are installed separately and retain their own licenses. Their source trees are not vendored here. TSB-AD users must obtain data separately and observe original dataset/source conditions. Filename manifests identify evaluation membership only; they contain no signal values or labels.

## R2 extension provenance

Four additional model/trainer dependencies and the historical routing/Mean-only adapters are ported from the frozen R2 source snapshot. Original additions/packaging retain the author's MIT scope; inherited portions retain the notices above. No raw dataset, internal controller or private authorization material is distributed. See `docs/SOURCE_PROVENANCE.json` for file-specific origins and engineering adaptations.

# Installation and tested environment

The table below is the original candidate's tested environment. The R2 repair uses existing Python3.11.15/PyTorch2.7.1+cu128 with CUDA hidden for CPU regression; no new installation or full CUDA reproduction is claimed. Current bounded outcomes are in `VALIDATION_RESULTS.json`.

The candidate uses the source requirements unchanged:

| Package | Version |
|---|---:|
| Python (tested interpreter) | 3.12.3 |
| PyTorch | 2.7.1 |
| NumPy | 2.3.2 |
| pandas | 2.3.1 |
| scikit-learn | 1.7.1 |
| SciPy | 1.16.1 |
| statsmodels | 0.14.5 |
| Matplotlib | 3.10.5 |
| tqdm | 4.66.5 |

Torch/NumPy/pandas/scikit-learn/SciPy/statsmodels/tqdm are runtime dependencies. Matplotlib is also currently required by inherited imports in the M trainer and ACF utilities even when no figure is drawn; it is not incorrectly labelled optional. pandas reads CSVs; scikit-learn provides bank clustering and metric helpers; statsmodels/SciPy support the original ACF window estimator. Core source has no dependency on the old external-baseline environments.

For Windows PowerShell, from the candidate root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

These install commands were not executed during preparation. Existing installed dependencies were reused read-only for CPU tests. Reinstallability on a clean machine is NOT VERIFIED. For a CUDA environment, obtain the appropriate official PyTorch 2.7.1 build for your platform from https://pytorch.org/ and verify driver compatibility; this task did not change any driver or install a CUDA toolkit. No unverified wheel URL or speed claim is supplied.

No package installation of this repository is needed: public scripts resolve `src` relative to their own file location. Run commands from the repository root. Quote data/output paths containing spaces. CPU tests set CUDA visibility to disabled and use one compute thread; `-B` prevents bytecode output in the reused environment/source tree. No GPU test or full experiment was performed.

Known inherited warnings: Python 3.12 reports invalid escape sequences in affiliation mathematical docstrings; sklearn may emit undefined-precision warnings for toy threshold candidates. They were observed and not hidden by changing metric definitions.

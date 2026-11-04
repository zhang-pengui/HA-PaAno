# Local candidate release checklist

Status meanings: PASS = verified within stated scope; BLOCKED = unresolved prerequisite; NOT VERIFIED = insufficient execution/evidence. A local candidate is not a published or full-GPU-certified release.

| Check | Status | Evidence / boundary |
|---|---|---|
| Core model source dependency closure | PASS | 22 relocated research Python modules plus package initializers; source hashes/diffs |
| Final training/inference/evaluation entry | PASS | Explicit final binding, default dry-run; actual joint pipeline retained |
| U-JLC and M-C configurations | PASS | Final source routes, frozen config checks, U350/M180 dry runs |
| Scientific parameters | PASS | Patch96/stride1/iterations100/batch512/lr1e-4/RevIN/top-k3; source-derived |
| Sample order/RNG preservation on bounded CPU cases | PASS | Exact reference/candidate hashes for six optimizer steps per track |
| No architecture/objective/optimizer/scheduler changes | PASS | Import-excluded executable AST match for copied modules, bounded numerical comparison |
| Dependency versions / existing environment | PASS | Python3.12.3 and pinned original requirements checked |
| Fresh environment reinstall | NOT VERIFIED | No installs authorized/performed during preparation |
| README CLI names and files | PASS | --help, real U/M formal dry runs, unit tests and stored-result inspector |
| Dataset source and layout documented | PASS | Official benchmark URL, prefix/full-series disclosure, no downloader |
| All raw row values of independently acquired data match historical data | NOT VERIFIED | Formal dry-run validates identifiers/headers; does not ship historical signal hashes |
| No dataset/score/checkpoint/cache/log in publishable set | PASS | Actual Git inclusion and explicit file/hash inventory; private audits excluded |
| No unhandled credential/private absolute path in publishable files | PASS | Targeted literal/path scan of actual candidate file set; not a guarantee against all possible secrets |
| PaAno/TSB-AD/affiliation license and notices | PASS | Separate original license texts, source citations and retained attribution |
| RevIN public-source permission | PASS | Explicit maintainer confirmation; no blanket CATCH licensing or independent legal certification |
| Any authorization-specific conditions checked by rights holder | NOT VERIFIED | Maintainer must retain evidence privately and comply before public release |
| HA-PaAno additions license / software author | PASS | User-confirmed MIT, 2026 Peng Zhang; CITATION.cff Peng Zhang only |
| Formal paper publication/DOI metadata | NOT VERIFIED | Not supplied; manuscript status only, no invented bibliographic fields |
| CPU light tests | PASS | 11 unit tests; U/M six-step reference probes; U/M toy joint pipeline |
| Full CUDA/ten-seed reproduction and resource sizing | NOT VERIFIED | Not run; no scientific result claim from smoke tests |
| Production queue/resource guard/crash-safe resume | NOT VERIFIED | Research controllers deliberately not copied; wrapper is fresh-output, foreground, serial |
| Original project unchanged | PASS | 164,148 file metadata records; 18,386 selected hashes; Git/index identical |
| No remote publication/upload/commit | PASS | New local unborn branch only; no remotes, commits, staged files, push, tag or release |

**Local candidate: READY FOR HUMAN REVIEW. Automatic public release: NOT AUTHORIZED.** No known technical blocker remains for the documented local CPU smoke workflow. The NOT VERIFIED items must not be promoted to PASS or converted into claims of complete historical artifact recovery/full-device reproducibility.

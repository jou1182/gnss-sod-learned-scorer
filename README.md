# Learned scorer for second-order design (SOD) of GNSS control networks

Code, frozen protocol and result files for the paper *AI-Assisted Observation Planning for GNSS Control Networks along
Linear Megaprojects in Saudi Arabia: A Gradient-Boosting Scorer Distilled from Expert Search* (simulation study).

**Everything here is a simulation.** Error, cost and accuracy-limit parameters are planning assumptions; the "Saudi"
corridors are schematic polylines between approximate city coordinates, not survey alignments. No field data are included.

## What is in the repository
| Path | Content |
|---|---|
| `sod/env.py` | Corridor simulator, GNSS error model, constraints, independent verifier |
| `sod/bdp.py` | Backward elimination with exact rank-one feasibility masks and the 21 features |
| `sod/gbm.py` | Gradient-boosting scorer, greedy solve, LNS refinement |
| `sod/baselines.py`, `sod/strong.py` | Conventional layouts, greedy rules, penalty GA/BPSO/SA, repair-based GA/SA, exact branch-and-bound |
| `gen_expert.py`, `train_gbm.py`, `train_gbm_seed.py` | Expert plans, DAgger training, extra training seeds |
| `run_eval.py`, `run_strong.py`, `run_usecase.py`, `eval_seeds.py` | Frozen-protocol evaluation, post-hoc stronger baselines / exact solver, design-space use case, seed study |
| `analyze*.py`, `extra_p1.py` | Statistics, feature importance, isolated timing |
| `policy.py`, `train.py`, `train_distill.py` | The neural attempts that did **not** work (kept for transparency) |
| `PROTOCOL.md` | Hypotheses and thresholds fixed before the test run, with amendments and post-hoc additions |
| `results/` | Per-instance results (JSON), trained scorers and expert plans (pickle) |

## Reproduce
```bash
pip install -r requirements.txt
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1   # one BLAS thread per worker
python gen_expert.py 20000 20060 results/expert_val.pkl        # expert plans (long: SA with 200k evaluations)
python gen_expert.py 10000 10660 results/expert_train.pkl
python train_gbm.py 5                                          # DAgger rounds -> results/gbm.pkl
python run_eval.py test    # also: ood, saudi, sens_eps08, sens_eps125, sens_r005, sens_r015
python analyze.py test ood saudi
```
The multiprocessing pools assume about 28-30 CPU workers; edit `Pool(...)` for your machine. Timings in the paper were measured
single-process on an otherwise idle machine (`extra_p1.py timing`).

## Notes and caveats
* Test seeds (30000-30199), out-of-distribution seeds (40000-40049) and Saudi schematic corridors were never used for training or model selection.
* The stronger baselines and the exact solver (7-8 stations only) were added after the frozen results; see `PROTOCOL.md`.
* The `.pkl` files are Python pickles; only load pickles from sources you trust.
* License: MIT (code). Please cite via `CITATION.cff`.

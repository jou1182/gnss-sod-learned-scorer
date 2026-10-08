# Pre-registered protocol — Research 1 (frozen before any test-set result was produced)

Question: can a learned removal policy (amortised optimisation) design GNSS corridor
control-network observation plans that are cheaper than classical heuristics and competitive
with metaheuristics run at a far larger evaluation budget, while remaining feasible by construction?

Hypotheses
- H1  Policy (greedy) cost < best classical greedy rule (backward-ratio) on held-out in-distribution corridors.
- H2  Policy best-of-16 cost within 5% of SA@200k evaluations (median) and >=100x faster.
- H3  Zero-shot generalisation: advantage over greedy rules persists on out-of-distribution
      (longer, denser) corridors and on schematic Saudi corridors (not seen in training).
- H4  Penalty-based GA/BPSO at the 20k-evaluation budget used in the SOD literature are
      dominated by the policy (feasibility rate and cost).

Data (all seeds fixed): train seeds 10000+ ; validation 20000+ (model selection only);
TEST seeds 30000..30199 (200 corridors, never used in training/tuning); OOD seeds 40000..40049
(N=45-70, length 300-600 km); Saudi schematic corridors: Haramain HSR (Makkah-Jeddah-KAEC-Madinah),
Riyadh-Dammam, Jeddah-Riyadh Landbridge, 170 km linear city.
Constraints: eps_h=10 mm, eps_u=25 mm (1-sigma), r_min=0.10, control spacing ~100 km.
Metrics: feasibility (independent verifier), cost ratio to best-known, wall time,
paired Wilcoxon signed-rank (two-sided, Holm correction), bootstrap 95% CI of median ratio.
Sensitivity (reported, not tuned): eps scaled x0.8 / x1.25; r_min 0.05 / 0.15.
No change to model, features or thresholds after the first test-set evaluation.

## Amendments (recorded before the corresponding runs; test-set run was NOT affected)
- OOD corridors: (N, length) are redrawn until the FULL network meets the accuracy limits (generation failed for some draws).
- Saudi schematic corridors: the full network did not meet 10/25 mm for three of four corridors (longer, sparser controls),
  so limits were set to eps_h=12 mm, eps_u=35 mm for this set (full network feasible in 10/10 seeds of all four). The model was not retrained.

## Post-hoc additions (after the frozen test results; NOT part of the protocol)
- Stronger baselines SA-repair / GA-repair (feasible-space, warm-started from BE-ratio), CPU budgets 3 s and 26 s, on first 60 held-out and 20 OOD corridors (`run_strong.py test|ood`).
- Exact branch-and-bound on 60 small networks (7-8 stations), limits = 1.5/1.8 x full-network precision (`run_strong.py small`); optimality proven for all 60.
- Finding: at 26 s warm-started SA beats the scorer+LNS in distribution; the scorer's advantage is at matched time and on larger corridors; on small networks GA/SA find the optimum more often.

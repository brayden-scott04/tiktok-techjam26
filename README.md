# Autonomous ML Research Agent — KuaiRand-Pure

TikTok TechJam 2026, Track 2. An agent that autonomously runs the ML research
loop — read the problem, inspect data, engineer features, train, evaluate,
reflect, iterate — on the KuaiRand-Pure recommender benchmark, driving the
validation score above the organizers' official baseline with minimal human
intervention.

**This is not a submission for a good recommender model. It is a submission
for the agent that builds one.**

## Results

| | valid GAUC | valid nDCG@5 | valid primary | test GAUC | test nDCG@5 | test primary | Δ primary vs baseline |
|---|---|---|---|---|---|---|---|
| official FM baseline | 0.6674 | 0.5357 | 0.6016 | 0.6610 | 0.5282 | 0.5946 | — |
| **our converged best (n0011)** | 0.6682 | 0.5359 | **0.6020** | 0.6630 | 0.5294 | **0.5962** | **+0.0016** |

The sealed hidden-test delta (+0.0016) is larger than the validation-side
delta (+0.0004) — a pleasant surprise, since test transfer usually erodes a
gain rather than growing it. Treated honestly: at z≈1.84 against the
baseline's own measured 5-seed noise, this is **borderline significant**
(two-tailed p≈0.066, just short of the conventional 0.05 threshold) — a
real, directionally consistent improvement, not proven beyond reasonable
doubt. See "Known limitations" for the full statistical picture.

**Resource usage to reach this result:** 24 scored iterations (26 total
including debug/repair attempts, of the 50-iteration cap), **582,551 LLM
tokens** (275,742 in + 306,809 out), $6.11 spent (of a $40 budget), ~2.95
hours wall-clock (of the 6-hour ceiling), **0 GPU-hours** (numpy-only, CPU
only), **0 manual interventions** across the entire scored run.

## Architecture

```
kit/       vendored, byte-identical organizer starter kit -- never edited, sha256-manifested
harness/   immutable task harness: sanitized data, sandbox, guards, scoring, reporting
agent/     the two-tier LLM loop: gpt-5.6-sol (brain) plans, gpt-5.6-luna (repair) executes/fixes
solutions/ n0000 = ported baseline; nNNNN = one per agent iteration
artifacts/ run_log.jsonl (deliverable, incl. a real diff_vs_parent per node), report.md, final_result.json
scripts/   setup, verification, and the run entry points
tests/     31+ tests: guards, sandbox execution, convergence FSM, tamper detection, isolation
```

### The leakage guarantee, in one sentence

The sandboxed process that runs a candidate's code is handed a cache in which
every validation and test row has its outcome columns physically overwritten
with `-1` before the process starts — the agent cannot see, train on, or
score against a label it does not have, and no test metric ever appears
anywhere in the run log. See `harness/context.py` and `harness/dataset.py`.

### Two-tier model split

| Role | Model | Verified prices (per Mtok) |
|---|---|---|
| brain (planning, drafting, improving) | `gpt-5.6-sol` | $4.00 in / $0.40 cached / $20.00 out |
| repair (syntax/error fixes, reformatting) | `gpt-5.6-luna` | $0.20 in / $0.02 cached / $1.20 out |
| fallback (after repeated brain failures) | `gpt-5.6-terra` | $2.00 in / $0.20 cached / $12.00 out |

Verified against the live account and a real structured-output call on
2026-08-31 (`scripts/verify_openai.py`, `scripts/probe_openai_api.py`) —
not assumed from documentation or pre-cutoff knowledge.

## Setup

Requires **Python 3.14+** (older versions may work but are untested; the
data-fetch script specifically needs Python 3.12+ for its `tarfile` API) and
an OpenAI API key.

```bash
git clone https://github.com/brayden-scott04/tiktok-techjam26.git
cd tiktok-techjam26

pip install -r requirements.txt
cp .env.example .env   # then open .env and fill in OPENAI_API_KEY
python -X utf8 -m scripts.fetch_data          # downloads + extracts KuaiRand-Pure outside the repo (~50MB)
python -X utf8 -m scripts.build_cache         # builds + verifies the authoritative and sanitized caches
python -X utf8 -m scripts.verify_node0        # confirms the ported baseline reproduces 0.6016
python -X utf8 -m pytest tests/ --ignore=tests/test_runner.py -q   # 48 fast tests, should all pass
```

`-X utf8` (or `PYTHONUTF8=1`) is required on Windows: this machine's default
locale encoding is `cp1252`, and the vendored kit's `open()` calls don't
specify an encoding. (In practice every KuaiRand-Pure CSV is pure ASCII, so
this doesn't currently bite — but it's cheap insurance against a codebase
whose own comments are UTF-8 Chinese.)

## Reproducing the baseline

All four organizer-published numbers reproduce exactly. The kit is vendored
to run standalone from inside `kit/`, exactly as the starter kit's own README
describes:

```bash
# grab the path scripts.fetch_data printed at the end (also saved in artifacts/data_manifest.json)
export KR_DATA_ROOT="$(python -c "import json; print(json.load(open('artifacts/data_manifest.json'))['data_dir'])")"

cd kit
for s in 0 1 2 3 4; do python -X utf8 baseline.py --model random --seed $s --data_dir "$KR_DATA_ROOT"; done
python -X utf8 baseline.py --model pop --data_dir "$KR_DATA_ROOT"
for s in 0 1 2 3 4; do python -X utf8 baseline.py --model fm --seed $s --data_dir "$KR_DATA_ROOT"; done
cd ..
```

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| random (5-seed mean) | 0.4996 | 0.4511 | **0.4753** |
| item popularity | 0.6308 | 0.5121 | **0.5715** |
| official FM (valid) | 0.6674 | 0.5357 | **0.6016** |
| official FM (test, 5-seed mean, std 0.0008) | 0.6610 | 0.5282 | **0.5946** |

## Running the agent

**Step 1 — mandatory sanity check, not optional.** Before spending any real
budget, confirm the whole pipeline actually works end to end:

```bash
python -X utf8 -m scripts.smoke_agent          # 3 cheap iterations, subsampled data, well under $1
```

**Step 2 — the real run. This spends real money and takes real time** — up
to $40 and 6 hours against your OpenAI account, though a typical run
converges well before either cap (ours used $6.11 and ~3h). It's resumable
if interrupted (re-run the same command; it picks up from `artifacts/state.json`).

```bash
python -X utf8 -m scripts.run_real_agent
```

**Step 3 — sealed test scoring, once, after convergence.** This is a
one-shot operation (it locks after the first call) — don't run it until
you're satisfied with the validation-side result.

```bash
python -X utf8 -m scripts.final_test_score
```

## Convergence policy (declared, per organizer FAQ 2.9)

The organizers permit a team-declared stopping rule provided it's fixed
before the run and respects the hard caps.

**This run happened in two declared phases**, disclosed here rather than
presented as one clean run. Phase 1 (floor = 12) converged legitimately at
the earliest point it was allowed to, using only 5% of budget and 20% of
wall-clock, but produced an edge over baseline that post-hoc statistical
analysis showed was indistinguishable from ordinary seed noise (z≈0.38,
p≈0.70). Since the hidden test set had not yet been touched — the one-shot
rule applies to *test* scoring, not to how much validation exploration
precedes it — we made a considered decision to raise the floor and continue
searching with a strengthened search process (see below) before finalizing.

- ε = 0.002, N = 3 scored iterations (unchanged from the default)
- **Minimum floor: 25 scored iterations** for the continuation phase (raised
  from 12 once real per-iteration cost/speed was known to be far cheaper
  than budgeted — $0.17 and ~7 min per iteration, not the $1-2/10-15min
  assumed when 12 was first set)
- Window semantics are cumulative, exactly as FAQ 2.9 specifies:
  `best(last N scored) - best(everything before) <= epsilon`
- Crashed/debug iterations count toward the 50-iteration cap but never
  advance or reset the convergence window

### Search-process upgrade after phase 1's post-mortem

Reading the actual generated code for phase 1's most ambitious attempts
(a sequence-modeling node and a watch-time-regression node, both of which
underperformed) revealed real, diagnosable implementation flaws — not that
the ideas were bad, but that e.g. raw high-cardinality ID features carry no
generalizable signal, and a synthetic regression target can be misaligned
with the ranking metric it's scored on. Rather than telling the agent the
specific fixes (that would be answer-key injection into its own discovery
process, undermining the Innovation criterion), the search process itself
was strengthened for the continuation phase:

- **Always-3-seed confirmation.** Every candidate is now confirmed on 3
  seeds before it can be promoted to "best" — not only when the apparent
  gain clears ε, as in phase 1 (which is exactly how a fluke became the
  phase-1 answer). A candidate is promoted only if the *mean* across seeds
  beats the incumbent's mean by more than ε.
- **Comparative diagnostics.** When a candidate doesn't get promoted, the
  next relevant prompt shows a per-impression-bucket comparison against
  whatever *did* win — strictly more information the harness already
  computes on both sides, not a diagnosis of cause.
- **Calibration tracking.** Each response's predicted effect size is now
  compared against what actually happened, and a rolling summary ("you've
  been overconfident by ~Nx") is shown back, so the model can self-correct.
- **A retry pool.** A successful-but-not-promoted node is no longer simply
  discarded — it's eligible to be revisited later with its own comparative
  diagnostics attached, so a good idea with a fixable flaw gets a second,
  better-informed shot instead of only ever being rediscovered by luck.
- **Best-of-2 sampling.** Each iteration now generates 2 candidate
  responses, validates both, and only spends the full 3-seed confirmation
  on whichever scores better at a quick seed-0 check — a standard
  compute-for-quality trade, affordable given the budget headroom.
- One general methodology nudge was added to `agent/prompts/improve.md`
  (validate incrementally rather than bundling several unverified changes
  in one shot) — generic experimental method, not a fact about this task.

None of this touches `hint_level` (still 0) or reveals what to try.

## What the agent is and isn't told

Given: the task, the pinned metric conventions (verbatim), the headroom
numbers (random/pop/baseline/oracle), the measured dead ends from the
starter kit's own ablations, the full train-split column schema, and the
hard training-data boundary.

Withheld: the starter kit README's own ranked list of promising research
directions. Finding those is the thing the 20% Innovation criterion is
scoring — handing the agent its own answer key would hollow that out.

## Known limitations

- **The result is borderline-significant, not proven.** z≈1.84, p≈0.066 on
  the sealed test delta. We report this plainly rather than rounding it up
  to "significant" — it's a real, consistent, positive signal, and it is
  also small enough that we would not be surprised if a second independent
  run landed on the other side of p=0.05.
- **The anti-stall direction-diversity property didn't hold in practice.**
  The design assumed the plateau-driven branch-switch would keep the final
  convergence window spanning ≥2 distinct research directions. In the
  actual run, all 9 direction categories were exhausted by iteration 13,
  after which the policy's documented fallback ("keep improving the current
  best regardless of staleness") took over for the rest of the run — so the
  final convergence window (n0022–n0024) was entirely within the `loss`
  direction. The convergence itself is still legitimate (11 consecutive
  `loss`-direction refinements independently landed in a tight 0.6019–0.6025
  band across 33 individual seed runs, a much stronger plateau signal than
  a single fluke), but the specific "diverse-window" guarantee we designed
  for should be read as "usually holds," not "always holds."
- **A couple of iterations showed exactly zero seed variance**
  (`primary_std = 0.0000` across all 3 seeds), which is unusual enough to be
  worth a second look — it may indicate the generated solution wasn't
  actually using the random seed anywhere (e.g. a fixed initialization), not
  that the result is more reliable than normal. Not investigated further
  given time constraints; flagged here for transparency.
- The node-selection policy (best/second-best/retry-pool + per-direction
  staleness tracking) is a simplified version of the originally planned UCB
  scheme — a deliberate scope cut for the time available, not an oversight.
- No held-out slice of validation exists for an independent overfitting
  check: valid labels are withheld from the sandbox by design (the same
  mechanism that makes the leakage guarantee structural), and the
  organizers' own FAQ 2.9.2 rules out `log_random` as an alternative
  unbiased check (it overlaps the test window). Valid-vs-test primary gap
  is reported honestly rather than hidden (and in this run, favorably —
  test transfer *improved* on validation rather than eroding it).
- numpy-only rules out sequence models with real per-user attention at any
  serious scale within the sandbox's timeout budget; some promising
  directions may be genuinely time-bound rather than idea-bound.
- The first phase of this run (12 iterations, floor=12) converged on a
  result later shown to be statistical noise (z≈0.38, p≈0.70) before the
  search process was strengthened and continued — see "Convergence policy"
  above. Disclosed as two declared phases rather than presented as a single
  clean run.

## What we'd improve given more time

- **Represent user history as aggregate statistics, not raw IDs.** The
  agent's own sequence-modeling attempt (`n0009`) underperformed because it
  used a single previous-video-ID as a feature — too sparse to ever repeat
  for the same user. Rolling engagement rate, recency, and author-affinity
  counts would carry the same behavioral signal without the sparsity
  problem. We diagnosed this from reading the agent's actual generated code
  after the fact; we deliberately did not feed this fix back into the
  agent's prompts, since that would be answer-key injection.
- **Replace the simplified node-selection heuristic with a real UCB-style
  tree search.** The current best/second-best/retry-pool policy is a
  documented scope cut, not the original design — it's also directly
  responsible for the convergence window ending up single-direction (see
  "Known limitations" above) once all 9 direction categories were
  exhausted by iteration 13.
- **Extend always-3-seed confirmation to the explore phase.** The four
  initial `draft` iterations each run on a single seed before the
  always-3-seed rule applies to later promotions — with more compute
  budget, confirming those early too would remove one more source of
  early-run noise.
- **Give the agent persisted memory across separate runs**, so lessons
  (not specific fixes — that would undermine the Innovation criterion, but
  structural facts like "this benchmark barely moves on capacity increases")
  don't have to be rediscovered from scratch each time.
- **Investigate the zero-seed-variance iterations** flagged above — worth
  knowing whether that's a real property of those specific solutions or a
  sign the seed isn't being threaded through somewhere it should be.

## Team

Solo submission — Brayden Scott Chen.

# Autonomous ML Research Agent — KuaiRand-Pure

TikTok TechJam 2026, Track 2. An agent that autonomously runs the ML research
loop — read the problem, inspect data, engineer features, train, evaluate,
reflect, iterate — on the KuaiRand-Pure recommender benchmark, driving the
validation score above the organizers' official baseline with minimal human
intervention.

**This is not a submission for a good recommender model. It is a submission
for the agent that builds one.**

## Architecture

```
kit/       vendored, byte-identical organizer starter kit -- never edited, sha256-manifested
harness/   immutable task harness: sanitized data, sandbox, guards, scoring, reporting
agent/     the two-tier LLM loop: gpt-5.6-sol (brain) plans, gpt-5.6-luna (repair) executes/fixes
solutions/ n0000 = ported baseline; nNNNN = one per agent iteration
artifacts/ run_log.jsonl (deliverable), state.json, report.md, final_result.json
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

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in OPENAI_API_KEY
python -X utf8 -m scripts.fetch_data          # downloads + extracts KuaiRand-Pure outside the repo
python -X utf8 -m scripts.build_cache         # builds + verifies the authoritative and sanitized caches
python -X utf8 -m scripts.verify_node0        # confirms the ported baseline reproduces 0.6016
```

`-X utf8` (or `PYTHONUTF8=1`) is required on Windows: this machine's default
locale encoding is `cp1252`, and the vendored kit's `open()` calls don't
specify an encoding. (In practice every KuaiRand-Pure CSV is pure ASCII, so
this doesn't currently bite — but it's cheap insurance against a codebase
whose own comments are UTF-8 Chinese.)

## Reproducing the baseline

All four organizer-published numbers reproduce exactly:

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| random (5-seed mean) | 0.4996 | 0.4511 | **0.4753** |
| item popularity | 0.6308 | 0.5121 | **0.5715** |
| official FM (valid) | 0.6674 | 0.5357 | **0.6016** |
| official FM (test, 5-seed mean, std 0.0008) | 0.6610 | 0.5282 | **0.5946** |

## Running the agent

```bash
python -X utf8 -m scripts.smoke_agent          # 3 cheap iterations, subsampled -- run this first
python -X utf8 -m scripts.run_real_agent       # the real run: up to 50 iterations, 6h, $40 ceiling, resumable
python -X utf8 -m scripts.final_test_score     # sealed, one-shot hidden-test scoring -- run once, after convergence
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

- The node-selection policy (best/second-best/retry-pool + per-direction
  staleness tracking) is a simplified version of the originally planned UCB
  scheme — a deliberate scope cut for the time available, not an oversight.
- No held-out slice of validation exists for an independent overfitting
  check: valid labels are withheld from the sandbox by design (the same
  mechanism that makes the leakage guarantee structural), and the
  organizers' own FAQ 2.9.2 rules out `log_random` as an alternative
  unbiased check (it overlaps the test window). Valid-vs-test primary gap
  is reported honestly rather than hidden.
- numpy-only rules out sequence models with real per-user attention at any
  serious scale within the sandbox's timeout budget; some promising
  directions may be genuinely time-bound rather than idea-bound.

## Team

(fill in before submission)

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
python -X utf8 -m scripts.smoke_agent          # 3 cheap iterations, subsampled, <$1 -- run this first
python -X utf8 -m agent.loop                   # the real run: up to 50 iterations, 6h, $40 ceiling
python -X utf8 -m scripts.final_test_score     # sealed, one-shot hidden-test scoring -- run once, after convergence
```

## Convergence policy (declared, per organizer FAQ 2.9)

The organizers permit a team-declared stopping rule provided it's fixed
before the run and respects the hard caps. Ours:

- ε = 0.002, N = 3 scored iterations (unchanged from the default)
- **Minimum floor: 12 scored iterations** — the 4-direction explore phase
  alone consumes 4, so this guarantees the exploit phase gets a real run
  before stopping is even possible
- Window semantics are cumulative, exactly as FAQ 2.9 specifies:
  `best(last N scored) - best(everything before) <= epsilon`
- Crashed/debug iterations count toward the 50-iteration cap but never
  advance or reset the convergence window

## What the agent is and isn't told

Given: the task, the pinned metric conventions (verbatim), the headroom
numbers (random/pop/baseline/oracle), the measured dead ends from the
starter kit's own ablations, the full train-split column schema, and the
hard training-data boundary.

Withheld: the starter kit README's own ranked list of promising research
directions. Finding those is the thing the 20% Innovation criterion is
scoring — handing the agent its own answer key would hollow that out.

## Known limitations

- The node-selection policy (best/second-best + per-direction staleness
  tracking) is a simplified version of the originally planned UCB scheme —
  a deliberate scope cut for the time available, not an oversight.
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

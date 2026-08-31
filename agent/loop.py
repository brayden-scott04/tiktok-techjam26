"""Main agent controller: solution tree search, convergence check, run-log
emission, checkpoint/resume.

Scope note: this implements the core loop faithfully (two-tier LLM calls,
validation gate, multi-seed confirmation, convergence FSM, debug/repair
ladder, run-log) but simplifies the node-selection heuristic relative to the
full plan (a straightforward best/second-best + per-direction staleness
tracker, rather than a fully general UCB score). This is a deliberate scope
cut, documented here and in the README, not an oversight.
"""
import hashlib
import json
import os
import time
import difflib

import numpy as np

from agent import memory
from agent.llm import LLMClient, LLMCallError
from agent.policy import check_convergence
from agent.schema import RESPONSE_SCHEMA, REPAIR_SCHEMA
from agent.validate import validate_candidate
from harness import dataset as ds
from harness.diagnostics import compute_diagnostics, render_diagnostics_markdown
from harness.eval_client import eval_valid_client  # noqa: F401 (documents that loop.py itself never scores test)
from harness.runner import run_node
from harness import task_spec as spec


def _sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class AgentLoop:
    def __init__(self, repo_root, run_config, smoke=False, max_iterations_override=None):
        self.root = repo_root
        self.cfg = run_config
        self.smoke = smoke
        self.llm = LLMClient()

        cache_path = os.path.join(repo_root, "artifacts", "cache", "kuairand_pure")
        self.sanitized_path = os.path.join(repo_root, "artifacts", "cache", "kuairand_pure_sanitized")
        auth_cache, _ = ds.load_cache(cache_path, skip_raw_meta=True)
        valid_c = auth_cache["valid"]
        self.valid_user_ids = list(valid_c["user_id_raw"])
        self.valid_labels = [int(x) for x in valid_c["long_view"]]
        self.valid_dates = list(valid_c["date"])

        self.run_id = time.strftime("run-%Y%m%d-%H%M%S")
        self.solutions_dir = os.path.join(repo_root, "solutions")
        self.nodes_dir = os.path.join(repo_root, "artifacts", "nodes")
        self.run_log_path = os.path.join(repo_root, "artifacts", "run_log.jsonl")
        self.state_path = os.path.join(repo_root, "artifacts", "state.json")
        os.makedirs(self.nodes_dir, exist_ok=True)

        caps = self.cfg["caps"]
        self.max_total_iterations = max_iterations_override or caps["max_total_iterations"]
        self.max_wall_seconds = caps["max_wall_seconds"]
        self.max_cost_usd = caps["max_cost_usd"]
        self.oracle_alarm = caps["oracle_proximity_alarm"]

        conv = self.cfg["convergence"]
        self.epsilon = conv["epsilon"]
        self.N = conv["N"]
        self.min_scored = conv["min_scored_iterations"]

        tree = self.cfg["tree"]
        self.explore_drafts = tree["explore_drafts"]
        self.plateau_misses = tree["plateau_same_direction_misses"]
        self.plateau_min_gain = tree["plateau_min_gain"]
        self.eps_greedy_p = tree["epsilon_greedy_revisit_p"]

        self.debug_cfg = self.cfg["debug"]
        self.sandbox_cfg = self.cfg["sandbox"]

        self.state = self._load_or_init_state()
        self._wall_start = time.time() - self.state["wall_seconds"]
        self.rng = np.random.default_rng(0)

    # ---------------------------------------------------------- state

    def _load_or_init_state(self):
        if os.path.exists(self.state_path):
            with open(self.state_path, encoding="utf-8") as fh:
                return json.load(fh)
        return {
            "run_id": self.run_id,
            "nodes": {},
            "order": [],
            "scored_primaries": [],
            "scored_directions": [],
            "known_ast_hashes": [],
            "best_node_id": None,
            "best_primary": -1.0,
            "second_best_node_id": None,
            "second_best_primary": -1.0,
            "total_iterations": 0,
            "scored_iterations": 0,
            "wall_seconds": 0.0,
            "cost_usd": 0.0,
            "direction_stats": {},  # direction -> {"attempts": n, "best_primary": x, "consecutive_misses": n}
            "converged": False,
            "convergence_trigger": None,
            "manual_interventions": 0,
        }

    def _save_state(self):
        self.state["wall_seconds"] = time.time() - self._wall_start
        with open(self.state_path, "w", encoding="utf-8") as fh:
            json.dump(self.state, fh, indent=2)

    def _append_log(self, record):
        with open(self.run_log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    # ---------------------------------------------------------- helpers

    def _budget_block(self):
        return memory.render_budget_block(
            {
                "total_iterations": self.state["total_iterations"],
                "max_total_iterations": self.max_total_iterations,
                "scored_iterations": self.state["scored_iterations"],
                "min_scored_iterations": self.min_scored,
                "wall_seconds": time.time() - self._wall_start,
                "max_wall_seconds": self.max_wall_seconds,
                "cost_usd": self.state["cost_usd"],
                "max_cost_usd": self.max_cost_usd,
            }
        )

    def _journal_nodes(self):
        out = []
        for nid in self.state["order"]:
            n = self.state["nodes"][nid]
            out.append(
                {
                    "node_id": nid, "parent_node_id": n.get("parent_node_id"),
                    "action": n["action"], "direction": n.get("direction"),
                    "hypothesis": n.get("hypothesis"), "status": n["status"],
                    "metrics": n.get("metrics"), "delta_vs_baseline": n.get("delta_vs_baseline"),
                    "seconds": n.get("seconds"), "est_runtime_sec": n.get("est_runtime_sec"),
                }
            )
        return out

    def _best_node_diagnostics_md(self):
        best_id = self.state["best_node_id"]
        if best_id is None:
            return "(no scored node yet)"
        node = self.state["nodes"][best_id]
        scores_path = node.get("valid_scores_path")
        if not scores_path or not os.path.exists(scores_path):
            return "(diagnostics unavailable for current best node)"
        scores = np.load(scores_path)
        diag = compute_diagnostics(self.valid_user_ids, self.valid_labels, scores, dates=self.valid_dates)
        return render_diagnostics_markdown(diag)

    def _stale_directions(self):
        return {
            d for d, s in self.state["direction_stats"].items()
            if s["consecutive_misses"] >= self.plateau_misses
        }

    def _unused_directions(self):
        from agent.schema import DIRECTION_ENUM

        used = set(self.state["direction_stats"].keys())
        return [d for d in DIRECTION_ENUM if d not in used]

    def _pick_action(self):
        """Returns (action, parent_node_id_or_None, forced_direction_or_None)."""
        n_scored = self.state["scored_iterations"]
        if n_scored < self.explore_drafts:
            return "draft", None, None

        stale = self._stale_directions()
        unused = self._unused_directions()

        use_second_best = (
            self.state["second_best_node_id"] is not None and self.rng.random() < self.eps_greedy_p
        )
        candidate_parent = self.state["second_best_node_id"] if use_second_best else self.state["best_node_id"]
        candidate_direction = self.state["nodes"][candidate_parent].get("direction") if candidate_parent else None

        if candidate_direction in stale:
            if unused:
                return "draft", None, None
            # all explored directions stale and none unused: fall back to
            # improving the global best regardless of staleness -- better
            # than stalling, and the convergence FSM will end the run anyway
            # once genuinely nothing is working.
            return "improve", self.state["best_node_id"], None

        return "improve", candidate_parent, None

    def _record_direction_attempt(self, direction, gain):
        stats = self.state["direction_stats"].setdefault(
            direction, {"attempts": 0, "best_primary": -1.0, "consecutive_misses": 0}
        )
        stats["attempts"] += 1
        if gain is not None and gain >= self.plateau_min_gain:
            stats["consecutive_misses"] = 0
        else:
            stats["consecutive_misses"] += 1

    # ---------------------------------------------------------- LLM calls

    def _call_llm_for_candidate(self, action, parent_id, forced_direction):
        journal_md = memory.render_journal(self._journal_nodes())
        diag_md = self._best_node_diagnostics_md()
        budget_md = self._budget_block()

        if action == "draft":
            used = list(self.state["direction_stats"].keys())
            messages = memory.build_draft_messages(journal_md, diag_md, budget_md, used)
        else:
            parent = self.state["nodes"][parent_id]
            with open(parent["code_path"], encoding="utf-8") as fh:
                parent_code = fh.read()
            messages = memory.build_improve_messages(journal_md, diag_md, budget_md, parent_code, parent.get("hypothesis", ""))

        parsed, usage = self.llm.call("brain", messages, RESPONSE_SCHEMA, "solution_response")
        self.state["cost_usd"] += usage["cost_usd"]
        return parsed, usage

    def _repair(self, code, error_summary, hypothesis):
        messages = memory.build_debug_messages(code, error_summary, hypothesis)
        parsed, usage = self.llm.call("repair", messages, REPAIR_SCHEMA, "repair_response")
        self.state["cost_usd"] += usage["cost_usd"]
        return parsed["code"], usage

    # ---------------------------------------------------------- one iteration

    def _run_full(self, solution_path, seed, timeout_sec):
        return run_node(
            solution_path=solution_path,
            out_dir=os.path.join(self.nodes_dir, f"tmp_seed{seed}"),
            sanitized_cache_path=self.sanitized_path,
            authoritative_valid_user_ids=self.valid_user_ids,
            authoritative_valid_labels=self.valid_labels,
            seed=seed,
            smoke=self.smoke,
            timeout_sec=timeout_sec,
            max_rss_mb=self.sandbox_cfg["max_rss_mb"],
            eval_call_cap=self.sandbox_cfg["eval_valid_call_cap"],
            repo_root=self.root,
        )

    def step(self):
        """Runs exactly one iteration (one LLM-authored node, possibly after
        internal repair attempts). Returns the run_log record for it."""
        action, parent_id, _ = self._pick_action()
        events = []
        t_iter0 = time.time()

        try:
            parsed, usage = self._call_llm_for_candidate(action, parent_id, None)
        except LLMCallError as e:
            events.append({"type": "llm_call_failed", "reason": str(e)})
            self.state["total_iterations"] += 1
            record = self._make_record(action, parent_id, None, None, "llm_error", events, [])
            self._append_log(record)
            self._save_state()
            return record

        code = parsed["code"]
        direction = parsed["direction"]
        node_num = self.state["total_iterations"] + 1
        node_id = f"n{node_num:04d}"
        node_dir = os.path.join(self.solutions_dir, node_id)
        os.makedirs(node_dir, exist_ok=True)
        code_path = os.path.join(node_dir, "solution.py")

        usages = [usage]
        known_hashes = set(self.state["known_ast_hashes"])
        status = None
        smoke_timeout = self.sandbox_cfg["smoke_timeout_sec"]

        for attempt in range(1, self.debug_cfg["max_debug_attempts_per_node"] + 2):
            with open(code_path, "w", encoding="utf-8") as fh:
                fh.write(code)
            result = validate_candidate(
                code, known_hashes, self.sanitized_path, self.valid_user_ids, self.valid_labels,
                self.root, smoke_timeout_sec=smoke_timeout,
            )
            if result.passed:
                status = "validated"
                break
            events.append({"type": "guard_or_smoke_reject", "stage": result.stage, "detail": result.detail[:2000]})
            if attempt > self.debug_cfg["max_debug_attempts_per_node"]:
                status = "dead"
                break
            try:
                code, repair_usage = self._repair(code, f"[{result.stage}] {result.detail}", parsed["hypothesis"])
                usages.append(repair_usage)
                events.append({"type": "debug_repair_attempt", "attempt": attempt, "resolved": None})
            except LLMCallError as e:
                events.append({"type": "llm_call_failed", "reason": str(e)})
                status = "dead"
                break

        self.state["total_iterations"] += 1
        ast_hash = result.ast_hash if status == "validated" else None

        metrics = None
        delta = None
        seconds = None
        eval_calls = None

        if status == "validated":
            full_timeout = self.sandbox_cfg["full_timeout_sec"]
            r0 = self._run_full(code_path, seed=0, timeout_sec=full_timeout)
            if r0.status != "ok":
                events.append({"type": "full_run_failed", "detail": (r0.traceback or r0.status)[:2000]})
                status = "error"
            else:
                from kit.evaluate import evaluate as kit_evaluate

                m0 = kit_evaluate(self.valid_user_ids, self.valid_labels, r0.valid_scores)
                incumbent = self.state["best_primary"]
                seconds = r0.seconds
                eval_calls = r0.eval_calls

                if m0["primary"] > incumbent + self.epsilon:
                    # Noise guard: a single-seed gain this size is only ~2.5x
                    # the baseline's own seed std (0.0008), so confirm on two
                    # more seeds before accepting -- and average ALL THREE
                    # metrics (not just primary) so GAUC/nDCG@5/primary stay
                    # internally consistent in the log (primary should equal
                    # their mean, not seed-0's GAUC/nDCG@5 paired with a
                    # 3-seed primary).
                    r1 = self._run_full(code_path, seed=1, timeout_sec=full_timeout)
                    r2 = self._run_full(code_path, seed=2, timeout_sec=full_timeout)
                    seeds_ok = [r for r in (r0, r1, r2) if r.status == "ok"]
                    per_seed = [kit_evaluate(self.valid_user_ids, self.valid_labels, r.valid_scores) for r in seeds_ok]
                    metrics = {
                        k: float(np.mean([m[k] for m in per_seed])) if per_seed else m0[k]
                        for k in ("GAUC", "nDCG@5", "primary")
                    }
                    metrics["primary_std"] = float(np.std([m["primary"] for m in per_seed])) if len(per_seed) > 1 else 0.0
                    seconds = sum(r.seconds for r in (r0, r1, r2))
                else:
                    metrics = {"GAUC": m0["GAUC"], "nDCG@5": m0["nDCG@5"], "primary": m0["primary"]}

                delta = metrics["primary"] - spec.BASELINE_VALID_PRIMARY
                status = "ok"

                valid_scores_path = os.path.join(node_dir, "valid_scores.npy")
                test_scores_path = os.path.join(node_dir, "test_scores.npy")
                np.save(valid_scores_path, r0.valid_scores)
                np.save(test_scores_path, r0.test_scores)

                gain = metrics["primary"] - incumbent
                self._record_direction_attempt(direction, gain)
                self._record_direction_best(direction, metrics["primary"])

                if metrics["primary"] > incumbent:
                    self.state["second_best_node_id"] = self.state["best_node_id"]
                    self.state["second_best_primary"] = incumbent
                    self.state["best_node_id"] = node_id
                    self.state["best_primary"] = metrics["primary"]

                self.state["scored_iterations"] += 1
                self.state["scored_primaries"].append(metrics["primary"])
                self.state["scored_directions"].append(direction)

                if metrics["primary"] > self.oracle_alarm:
                    events.append({"type": "leak_alarm", "primary": metrics["primary"]})

        if ast_hash:
            self.state["known_ast_hashes"].append(ast_hash)

        node_record = {
            "action": action, "parent_node_id": parent_id, "direction": direction,
            "hypothesis": parsed["hypothesis"], "status": status,
            "code_path": code_path, "ast_hash": ast_hash,
            "metrics": metrics, "delta_vs_baseline": delta, "seconds": seconds,
            "est_runtime_sec": parsed.get("est_runtime_sec"),
            "valid_scores_path": os.path.join(node_dir, "valid_scores.npy") if metrics else None,
            "test_scores_path": os.path.join(node_dir, "test_scores.npy") if metrics else None,
        }
        self.state["nodes"][node_id] = node_record
        self.state["order"].append(node_id)

        record = self._make_record(action, parent_id, direction, node_id, status, events, usages, parsed, metrics, delta, seconds)
        record["iteration"] = self.state["total_iterations"]
        record["node_id"] = node_id
        record["wall_seconds_this_iteration"] = time.time() - t_iter0
        self._append_log(record)
        self._save_state()
        return record

    def _record_direction_best(self, direction, primary):
        stats = self.state["direction_stats"].setdefault(
            direction, {"attempts": 0, "best_primary": -1.0, "consecutive_misses": 0}
        )
        stats["best_primary"] = max(stats["best_primary"], primary)

    def _make_record(self, action, parent_id, direction, node_id, status, events, usages, parsed=None, metrics=None, delta=None, seconds=None):
        total_cost = sum(u["cost_usd"] for u in usages if u)
        total_in = sum(u["input_tokens"] for u in usages if u)
        total_out = sum(u["output_tokens"] for u in usages if u)
        return {
            "schema_version": 1,
            "run_id": self.run_id,
            "scored_iteration": self.state["scored_iterations"] if metrics else None,
            "action": action, "parent_node_id": parent_id, "direction": direction, "node_id": node_id,
            "hypothesis": parsed.get("hypothesis") if parsed else None,
            "rationale": parsed.get("rationale") if parsed else None,
            "references": parsed.get("references") if parsed else None,
            "changes": parsed.get("changes") if parsed else None,
            "expected_effect": parsed.get("expected_effect") if parsed else None,
            "risks": parsed.get("risks") if parsed else None,
            "est_runtime_sec": parsed.get("est_runtime_sec") if parsed else None,
            "status": status,
            "metrics": metrics,
            "delta_vs_baseline_valid_primary": delta,
            "seconds": seconds,
            "best_so_far": {"node_id": self.state["best_node_id"], "valid_primary": self.state["best_primary"]},
            "events": events,
            "usage": {"calls": len(usages), "input_tokens": total_in, "output_tokens": total_out, "cost_usd": total_cost},
            "cumulative": {
                "total_iterations": self.state["total_iterations"],
                "scored_iterations": self.state["scored_iterations"],
                "cost_usd": self.state["cost_usd"],
                "wall_seconds": time.time() - self._wall_start,
            },
        }

    # ---------------------------------------------------------- stop conditions

    def check_stop(self):
        if self.state["total_iterations"] >= self.max_total_iterations:
            return "hard_cap"
        if time.time() - self._wall_start >= self.max_wall_seconds:
            return "wall_clock"
        if self.state["cost_usd"] >= self.max_cost_usd:
            return "budget"
        conv = check_convergence(
            self.state["scored_primaries"], self.epsilon, self.N, self.min_scored,
            directions=self.state["scored_directions"],
        )
        if conv["converged"]:
            self.state["converged"] = True
            self.state["convergence_detail"] = conv
            return "converged"
        return None

    def run(self):
        if not self.state["order"]:
            self._seed_node0()
        while True:
            trigger = self.check_stop()
            if trigger:
                self.state["convergence_trigger"] = trigger
                self._save_state()
                self._append_log({"schema_version": 1, "run_id": self.run_id, "event": "run_end", "trigger": trigger, "state": self.state})
                return trigger
            self.step()

    def _seed_node0(self):
        """Scores the pre-existing n0000 baseline once to seed best_primary.
        Not LLM-authored, so it does not count against total_iterations."""
        code_path = os.path.join(self.solutions_dir, "n0000", "solution.py")
        r0 = self._run_full(code_path, seed=0, timeout_sec=self.sandbox_cfg["full_timeout_sec"])
        if r0.status != "ok":
            raise RuntimeError(f"failed to seed node0: {r0.status} {r0.traceback}")
        from kit.evaluate import evaluate as kit_evaluate

        m0 = kit_evaluate(self.valid_user_ids, self.valid_labels, r0.valid_scores)
        node_dir = os.path.join(self.solutions_dir, "n0000")
        np.save(os.path.join(node_dir, "valid_scores.npy"), r0.valid_scores)
        np.save(os.path.join(node_dir, "test_scores.npy"), r0.test_scores)
        self.state["nodes"]["n0000"] = {
            "action": "root", "parent_node_id": None, "direction": "baseline",
            "hypothesis": "official FM baseline, ported to the fit_predict contract",
            "status": "ok", "code_path": code_path,
            "metrics": {"GAUC": m0["GAUC"], "nDCG@5": m0["nDCG@5"], "primary": m0["primary"]},
            "delta_vs_baseline": m0["primary"] - spec.BASELINE_VALID_PRIMARY,
            "seconds": r0.seconds, "est_runtime_sec": 40,
            "valid_scores_path": os.path.join(node_dir, "valid_scores.npy"),
            "test_scores_path": os.path.join(node_dir, "test_scores.npy"),
        }
        self.state["order"].append("n0000")
        self.state["best_node_id"] = "n0000"
        self.state["best_primary"] = m0["primary"]
        self.state["scored_iterations"] += 1
        self.state["scored_primaries"].append(m0["primary"])
        self.state["scored_directions"].append("baseline")
        self._save_state()

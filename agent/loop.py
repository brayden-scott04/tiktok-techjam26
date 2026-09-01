"""Main agent controller: solution tree search, convergence check, run-log
emission, checkpoint/resume.

Post-mortem upgrade (session checkpoint 2, after run 1 converged on a
fluke-sized, unconfirmed edge): every candidate is now always confirmed on
3 seeds before being allowed to become "best" (no more single-seed noise
driving the search); failed-but-successfully-run candidates are tracked in a
retry pool with their own bucketed comparison against whatever beat them, so
a good idea with a fixable flaw gets a second, better-informed shot instead
of being discarded; the model's own predicted vs. actual effect is tracked
as a calibration signal; and each iteration samples multiple candidates
(best-of-N) rather than committing to a single LLM sample. None of this
touches hint_level or reveals *what* to try -- it only makes the search
more statistically honest and self-aware.

Scope note: this still simplifies the node-selection heuristic relative to
the full plan (a straightforward best/second-best/retry-pool selector with
per-direction staleness tracking, rather than a fully general UCB score).
This is a deliberate scope cut, documented here and in the README, not an
oversight.
"""
import difflib
import json
import os
import time

import numpy as np

from agent import memory
from agent.llm import LLMClient, LLMCallError
from agent.policy import check_convergence
from agent.schema import RESPONSE_SCHEMA, REPAIR_SCHEMA
from agent.validate import validate_candidate
from harness import dataset as ds
from harness.diagnostics import (
    compute_diagnostics, render_diagnostics_markdown,
    compute_comparative_diagnostics, render_comparative_diagnostics_markdown,
)
from harness.eval_client import eval_valid_client  # noqa: F401 (documents that loop.py itself never scores test)
from harness.runner import run_node
from harness import task_spec as spec


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
        self.retry_pool_p = tree.get("retry_pool_p", 0.15)
        self.retry_pool_max_size = tree.get("retry_pool_max_size", 5)

        self.debug_cfg = self.cfg["debug"]
        self.sandbox_cfg = self.cfg["sandbox"]
        self.n_candidates = self.cfg.get("search", {}).get("candidates_per_iteration", 1)

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
            "retry_pool": [],
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
        base = memory.render_budget_block(
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
        calib = self._calibration_summary()
        return f"{base}\n{calib}" if calib else base

    def _calibration_summary(self):
        pairs = []
        for nid in self.state["order"][-6:]:
            n = self.state["nodes"][nid]
            ee = n.get("expected_effect")
            if ee and ee.get("delta") is not None and n.get("calibration_error") is not None:
                pairs.append((ee["delta"], n["calibration_error"]))
        return memory.render_calibration_summary(pairs)

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
                    "expected_effect": n.get("expected_effect"), "calibration_error": n.get("calibration_error"),
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
        """Returns (action, parent_node_id_or_None)."""
        n_scored = self.state["scored_iterations"]
        if n_scored < self.explore_drafts:
            return "draft", None

        stale = self._stale_directions()
        unused = self._unused_directions()
        retry_pool = self.state.get("retry_pool", [])

        roll = self.rng.random()
        if retry_pool and roll < self.retry_pool_p:
            # give the most recent good-idea-but-not-promoted node another,
            # better-informed shot (see comparison_vs_incumbent_md wiring).
            return "improve", retry_pool[-1]

        use_second_best = (
            self.state["second_best_node_id"] is not None
            and roll < self.retry_pool_p + self.eps_greedy_p
        )
        candidate_parent = self.state["second_best_node_id"] if use_second_best else self.state["best_node_id"]
        candidate_direction = self.state["nodes"][candidate_parent].get("direction") if candidate_parent else None

        if candidate_direction in stale:
            if unused:
                return "draft", None
            # all explored directions stale and none unused: fall back to
            # improving the global best regardless of staleness -- better
            # than stalling, and the convergence FSM will end the run anyway
            # once genuinely nothing is working.
            return "improve", self.state["best_node_id"]

        return "improve", candidate_parent

    def _record_direction_attempt(self, direction, gain):
        stats = self.state["direction_stats"].setdefault(
            direction, {"attempts": 0, "best_primary": -1.0, "consecutive_misses": 0}
        )
        stats["attempts"] += 1
        if gain is not None and gain >= self.plateau_min_gain:
            stats["consecutive_misses"] = 0
        else:
            stats["consecutive_misses"] += 1

    def _record_direction_best(self, direction, primary):
        stats = self.state["direction_stats"].setdefault(
            direction, {"attempts": 0, "best_primary": -1.0, "consecutive_misses": 0}
        )
        stats["best_primary"] = max(stats["best_primary"], primary)

    # ---------------------------------------------------------- LLM calls

    def _call_llm_for_candidate(self, action, parent_id):
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
            messages = memory.build_improve_messages(
                journal_md, diag_md, budget_md, parent_code, parent.get("hypothesis", ""),
                comparison_md=parent.get("comparison_vs_incumbent_md"),
            )

        parsed, usage = self.llm.call("brain", messages, RESPONSE_SCHEMA, "solution_response")
        self.state["cost_usd"] += usage["cost_usd"]
        return parsed, usage

    def _repair(self, code, error_summary, hypothesis):
        messages = memory.build_debug_messages(code, error_summary, hypothesis)
        parsed, usage = self.llm.call("repair", messages, REPAIR_SCHEMA, "repair_response")
        self.state["cost_usd"] += usage["cost_usd"]
        return parsed["code"], usage

    def _validate_with_repairs(self, code, hypothesis, known_hashes):
        """Runs the compile->guard->dedup->smoke gate, repairing on failure up
        to max_debug_attempts_per_node times. Returns
        (final_code, ast_hash_or_None, status, events, usages)."""
        events = []
        usages = []
        status = None
        ast_hash = None
        smoke_timeout = self.sandbox_cfg["smoke_timeout_sec"]

        for attempt in range(1, self.debug_cfg["max_debug_attempts_per_node"] + 2):
            result = validate_candidate(
                code, known_hashes, self.sanitized_path, self.valid_user_ids, self.valid_labels,
                self.root, smoke_timeout_sec=smoke_timeout,
            )
            if result.passed:
                status = "validated"
                ast_hash = result.ast_hash
                break
            events.append({"type": "guard_or_smoke_reject", "stage": result.stage, "detail": result.detail[:2000]})
            if attempt > self.debug_cfg["max_debug_attempts_per_node"]:
                status = "dead"
                break
            try:
                code, repair_usage = self._repair(code, f"[{result.stage}] {result.detail}", hypothesis)
                usages.append(repair_usage)
                events.append({"type": "debug_repair_attempt", "attempt": attempt, "resolved": None})
            except LLMCallError as e:
                events.append({"type": "llm_call_failed", "reason": str(e)})
                status = "dead"
                break

        return code, ast_hash, status, events, usages

    # ---------------------------------------------------------- execution

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

    def _diff_vs_parent(self, parent_id, code_path):
        """Computes a real unified diff against the parent node's committed
        code -- the deliverable asks for "the code diff applied" per
        iteration, and an LLM-written `changes` bullet list is a description,
        not a diff. Harness-generated, not model-generated, so it can't be
        gamed or hallucinated. Returns None for a `draft` with no parent, or
        if the parent's code is unavailable for any reason."""
        if not parent_id:
            return None
        parent = self.state["nodes"].get(parent_id)
        if not parent or not parent.get("code_path") or not os.path.exists(parent["code_path"]):
            return None
        try:
            with open(parent["code_path"], encoding="utf-8") as fh:
                parent_code = fh.readlines()
            with open(code_path, encoding="utf-8") as fh:
                new_code = fh.readlines()
        except OSError:
            return None
        diff = difflib.unified_diff(
            parent_code, new_code,
            fromfile=f"{parent_id}/solution.py", tofile="solution.py",
        )
        return "".join(diff)

    def _mean_metrics(self, results):
        """results: list of NodeResult, some possibly failed. Returns
        (metrics_dict_or_None, n_seeds_ok)."""
        from kit.evaluate import evaluate as kit_evaluate

        ok = [r for r in results if r.status == "ok"]
        if not ok:
            return None, 0
        per_seed = [kit_evaluate(self.valid_user_ids, self.valid_labels, r.valid_scores) for r in ok]
        metrics = {k: float(np.mean([m[k] for m in per_seed])) for k in ("GAUC", "nDCG@5", "primary")}
        metrics["primary_std"] = float(np.std([m["primary"] for m in per_seed])) if len(per_seed) > 1 else 0.0
        metrics["n_seeds_ok"] = len(ok)
        return metrics, len(ok)

    def step(self):
        """Runs exactly one iteration: samples up to n_candidates LLM
        responses (best-of-N), validates each through the repair ladder,
        quick-checks validated survivors at seed 0, and spends the full
        3-seed confirmation only on the best-scoring one. Returns the
        run_log record for it."""
        action, parent_id = self._pick_action()
        events = []
        usages = []
        t_iter0 = time.time()
        node_num_preview = self.state["total_iterations"] + 1
        print(f"[loop] iteration {node_num_preview}: action={action} parent={parent_id}", flush=True)

        candidates = []
        for i in range(self.n_candidates):
            print(f"[loop] iteration {node_num_preview}: generating candidate {i+1}/{self.n_candidates}", flush=True)
            try:
                parsed, usage = self._call_llm_for_candidate(action, parent_id)
                usages.append(usage)
            except LLMCallError as e:
                events.append({"type": "llm_call_failed", "reason": str(e), "candidate": i})
                continue
            known_hashes = set(self.state["known_ast_hashes"]) | {
                c["ast_hash"] for c in candidates if c.get("ast_hash")
            }
            final_code, ast_hash, status, sub_events, sub_usages = self._validate_with_repairs(
                parsed["code"], parsed["hypothesis"], known_hashes
            )
            usages.extend(sub_usages)
            for e in sub_events:
                e["candidate"] = i
                events.append(e)
            print(f"[loop] iteration {node_num_preview}: candidate {i+1} validation -> {status}", flush=True)
            candidates.append(
                {"parsed": parsed, "code": final_code, "ast_hash": ast_hash, "status": status, "direction": parsed["direction"]}
            )

        self.state["total_iterations"] += 1
        node_num = self.state["total_iterations"]
        node_id = f"n{node_num:04d}"
        node_dir = os.path.join(self.solutions_dir, node_id)
        os.makedirs(node_dir, exist_ok=True)
        code_path = os.path.join(node_dir, "solution.py")

        metrics = delta = seconds = direction = parsed = comparison_md = calibration_error = None
        ast_hash = None
        validated = [c for c in candidates if c["status"] == "validated"]

        if not validated:
            status = "dead" if candidates else "llm_error"
            if candidates:
                parsed = candidates[0]["parsed"]
                direction = candidates[0]["direction"]
                with open(code_path, "w", encoding="utf-8") as fh:
                    fh.write(candidates[0]["code"])
        else:
            full_timeout = self.sandbox_cfg["full_timeout_sec"]
            print(f"[loop] iteration {node_num_preview}: quick seed-0 check on {len(validated)} validated candidate(s)", flush=True)
            quick = []
            for c in validated:
                tmp_path = os.path.join(node_dir, "_candidate_tmp.py")
                with open(tmp_path, "w", encoding="utf-8") as fh:
                    fh.write(c["code"])
                r0 = self._run_full(tmp_path, seed=0, timeout_sec=full_timeout)
                print(f"[loop] iteration {node_num_preview}: candidate seed-0 -> status={r0.status} seconds={r0.seconds:.1f}", flush=True)
                quick.append((c, r0))

            ok_quick = [(c, r) for c, r in quick if r.status == "ok"]
            if not ok_quick:
                status = "error"
                best_c, best_r = quick[0]
                parsed, direction, ast_hash = best_c["parsed"], best_c["direction"], best_c["ast_hash"]
                events.append({"type": "full_run_failed", "detail": (best_r.traceback or best_r.status)[:2000]})
                with open(code_path, "w", encoding="utf-8") as fh:
                    fh.write(best_c["code"])
                seconds = sum(r.seconds for _, r in quick)
            else:
                from kit.evaluate import evaluate as kit_evaluate

                scored = [(c, r, kit_evaluate(self.valid_user_ids, self.valid_labels, r.valid_scores)) for c, r in ok_quick]
                best_c, best_r0, _ = max(scored, key=lambda t: t[2]["primary"])
                parsed, direction, ast_hash = best_c["parsed"], best_c["direction"], best_c["ast_hash"]
                with open(code_path, "w", encoding="utf-8") as fh:
                    fh.write(best_c["code"])

                print(f"[loop] iteration {node_num_preview}: winner selected, running seeds 1,2 for confirmation", flush=True)
                r1 = self._run_full(code_path, seed=1, timeout_sec=full_timeout)
                r2 = self._run_full(code_path, seed=2, timeout_sec=full_timeout)
                metrics, _ = self._mean_metrics([best_r0, r1, r2])
                if metrics:
                    print(f"[loop] iteration {node_num_preview}: confirmed primary={metrics['primary']:.4f} (std={metrics['primary_std']:.4f}, n_seeds_ok={metrics['n_seeds_ok']})", flush=True)
                seconds = sum(r.seconds for _, r in quick) + r1.seconds + r2.seconds

                if metrics is None:
                    status = "error"
                    events.append({"type": "full_run_failed", "detail": "confirmation seeds 1,2 both failed after a working seed 0"})
                else:
                    delta = metrics["primary"] - spec.BASELINE_VALID_PRIMARY
                    status = "ok"

                    valid_scores_path = os.path.join(node_dir, "valid_scores.npy")
                    test_scores_path = os.path.join(node_dir, "test_scores.npy")
                    np.save(valid_scores_path, best_r0.valid_scores)
                    np.save(test_scores_path, best_r0.test_scores)

                    incumbent = self.state["best_primary"]
                    gain = metrics["primary"] - incumbent
                    self._record_direction_attempt(direction, gain)
                    self._record_direction_best(direction, metrics["primary"])

                    if self.state["best_node_id"]:
                        inc_path = self.state["nodes"][self.state["best_node_id"]].get("valid_scores_path")
                        if inc_path and os.path.exists(inc_path):
                            inc_scores = np.load(inc_path)
                            comp = compute_comparative_diagnostics(
                                best_r0.valid_scores, inc_scores, self.valid_user_ids, self.valid_labels,
                                dates=self.valid_dates,
                            )
                            comparison_md = render_comparative_diagnostics_markdown(comp)

                    # Statistically honest promotion: only a CONFIRMED
                    # (3-seed mean) gain clearing epsilon becomes "best" --
                    # this is the direct fix for run 1's fluke (n0011 was
                    # promoted on a single seed with a sub-epsilon gain).
                    promoted = metrics["primary"] > incumbent + self.epsilon
                    if promoted:
                        self.state["second_best_node_id"] = self.state["best_node_id"]
                        self.state["second_best_primary"] = incumbent
                        self.state["best_node_id"] = node_id
                        self.state["best_primary"] = metrics["primary"]
                    else:
                        pool = self.state.setdefault("retry_pool", [])
                        pool.append(node_id)
                        self.state["retry_pool"] = pool[-self.retry_pool_max_size:]

                    self.state["scored_iterations"] += 1
                    self.state["scored_primaries"].append(metrics["primary"])
                    self.state["scored_directions"].append(direction)

                    if metrics["primary"] > self.oracle_alarm:
                        events.append({"type": "leak_alarm", "primary": metrics["primary"]})

                    expected_effect = parsed.get("expected_effect")
                    if expected_effect and expected_effect.get("delta") is not None:
                        calibration_error = gain - expected_effect["delta"]

        if ast_hash:
            self.state["known_ast_hashes"].append(ast_hash)

        diff_vs_parent = self._diff_vs_parent(parent_id, code_path) if parsed else None

        node_record = {
            "action": action, "parent_node_id": parent_id, "direction": direction,
            "hypothesis": parsed.get("hypothesis") if parsed else None, "status": status,
            "code_path": code_path, "ast_hash": ast_hash,
            "metrics": metrics, "delta_vs_baseline": delta, "seconds": seconds,
            "est_runtime_sec": parsed.get("est_runtime_sec") if parsed else None,
            "expected_effect": parsed.get("expected_effect") if parsed else None,
            "calibration_error": calibration_error,
            "comparison_vs_incumbent_md": comparison_md,
            "diff_vs_parent": diff_vs_parent,
            "valid_scores_path": os.path.join(node_dir, "valid_scores.npy") if metrics else None,
            "test_scores_path": os.path.join(node_dir, "test_scores.npy") if metrics else None,
            "n_candidates_tried": len(candidates),
        }
        self.state["nodes"][node_id] = node_record
        self.state["order"].append(node_id)

        record = self._make_record(action, parent_id, direction, node_id, status, events, usages, parsed, metrics, delta, seconds)
        record["iteration"] = self.state["total_iterations"]
        record["node_id"] = node_id
        record["n_candidates_tried"] = len(candidates)
        record["wall_seconds_this_iteration"] = time.time() - t_iter0
        record["diff_vs_parent"] = diff_vs_parent
        self._append_log(record)
        self._save_state()
        return record

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
        """Scores the pre-existing n0000 baseline (3-seed mean, same rigor as
        every other node) to seed best_primary. Not LLM-authored, so it does
        not count against total_iterations."""
        code_path = os.path.join(self.solutions_dir, "n0000", "solution.py")
        full_timeout = self.sandbox_cfg["full_timeout_sec"]
        r0 = self._run_full(code_path, seed=0, timeout_sec=full_timeout)
        if r0.status != "ok":
            raise RuntimeError(f"failed to seed node0: {r0.status} {r0.traceback}")
        r1 = self._run_full(code_path, seed=1, timeout_sec=full_timeout)
        r2 = self._run_full(code_path, seed=2, timeout_sec=full_timeout)
        metrics, _ = self._mean_metrics([r0, r1, r2])

        node_dir = os.path.join(self.solutions_dir, "n0000")
        np.save(os.path.join(node_dir, "valid_scores.npy"), r0.valid_scores)
        np.save(os.path.join(node_dir, "test_scores.npy"), r0.test_scores)
        self.state["nodes"]["n0000"] = {
            "action": "root", "parent_node_id": None, "direction": "baseline",
            "hypothesis": "official FM baseline, ported to the fit_predict contract",
            "status": "ok", "code_path": code_path,
            "metrics": metrics,
            "delta_vs_baseline": metrics["primary"] - spec.BASELINE_VALID_PRIMARY,
            "seconds": r0.seconds + r1.seconds + r2.seconds, "est_runtime_sec": 40,
            "valid_scores_path": os.path.join(node_dir, "valid_scores.npy"),
            "test_scores_path": os.path.join(node_dir, "test_scores.npy"),
        }
        self.state["order"].append("n0000")
        self.state["best_node_id"] = "n0000"
        self.state["best_primary"] = metrics["primary"]
        self.state["scored_iterations"] += 1
        self.state["scored_primaries"].append(metrics["primary"])
        self.state["scored_directions"].append("baseline")
        self._save_state()

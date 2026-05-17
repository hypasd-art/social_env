"""Agent interaction visualizer — Flask server.

Usage:
    python visualizer/server.py [--port PORT] [--data-dir RUNS_DIR]

Supports:
  - Browsing past evaluation JSON results
  - Launching new evaluation runs from the UI via CLI subprocess
  - Real-time progress tracking (polling-based)
  - Conversation viewing from execution traces
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

DATA_DIR = Path(os.environ.get("VIS_DATA_DIR", "runs"))

_active_runs: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()


def _scan_runs() -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    if not DATA_DIR.exists():
        return runs

    for f in sorted(DATA_DIR.rglob("*.json"), reverse=True):
        if f.suffix != ".json" or "execution" in f.stem or "agent_episode" in f.stem:
            continue
        try:
            with open(f) as fh:
                data = json.load(fh)
            rows = data.get("rows", [])
            agg = data.get("aggregate_means", {})
            overall = agg.get("overall", agg)
            runs.append({
                "id": f.stem,
                "path": str(f.relative_to(DATA_DIR)),
                "type": "eval_json",
                "timestamp": data.get("run_timestamp", ""),
                "agent_models": data.get("agent_models", []),
                "evaluator_model": data.get("evaluator_model", ""),
                "n_episodes": len(rows),
                "success_rate": overall.get("terminal_success_rate", 0),
                "llm_overall_mean": overall.get("llm_overall_mean", {}),
            })
        except Exception:
            continue

    with _lock:
        now_ts = time.time()
        for rid, state in list(_active_runs.items()):
            # Skip stale runs (older than 2 hours, or failed/done runs older than 10 min)
            age = now_ts - state.get("_last_update", 0)
            if state.get("status") in ("done", "failed") and age > 600:
                continue
            if age > 7200:
                continue
            runs.insert(0, {
                "id": rid,
                "type": "active",
                "timestamp": state.get("started_at", ""),
                "agent_models": state.get("config", {}).get("agent_models", []),
                "evaluator_model": state.get("config", {}).get("evaluator_model", ""),
                "n_episodes": state.get("total", 0),
                "success_rate": 0,
                "status": state.get("status", "pending"),
                "llm_overall_mean": {},
            })

    return runs


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/runs")
def api_runs():
    return jsonify(_scan_runs())


@app.route("/api/scenarios")
def api_scenarios():
    scenarios = []
    manifest_path = os.path.expanduser("~/.sotopia/data/long_term_negotiation_llm_manifest.json")
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path) as fh:
                manifest = json.load(fh)
            for env in manifest.get("environments", []):
                if isinstance(env, dict):
                    scenarios.append({
                        "pk": env.get("pk", ""),
                        "codename": env.get("codename", ""),
                        "num_participants": env.get("num_participants"),
                    })
        except Exception:
            pass
    return jsonify(scenarios)


@app.route("/api/run/<run_id>")
def api_run_detail(run_id: str):
    for f in DATA_DIR.rglob("*.json"):
        if f.stem == run_id:
            with open(f) as fh:
                data = json.load(fh)
            return _json_response(data)
    return jsonify({"error": "not found"}), 404


@app.route("/api/execution/<exec_id>")
def api_execution(exec_id: str):
    for f in DATA_DIR.rglob("*.execution.json"):
        if f.stem.replace(".execution", "") == exec_id:
            with open(f) as fh:
                data = json.load(fh)
            result: dict[str, Any] = {
                "terminal": data.get("terminal", "?"),
                "macro_steps_used": data.get("macro_steps_used", 0),
                "scheduling_log": _format_schedule(data.get("scheduling_log", [])),
                "session_log": data.get("session_log", []),
                "action_log": data.get("action_log", []),
                "event_log": data.get("event_log", []),
                "contracts": data.get("contracts", []),
                "primary_contract_id": data.get("primary_contract_id"),
                "agents": list(data.get("visible_history_by_agent", {}).keys()),
            }
            return _json_response(result)
    return jsonify({"error": "not found"}), 404


@app.route("/api/trace/<exec_id>")
def api_trace(exec_id: str):
    for f in DATA_DIR.rglob("*.execution.json"):
        if f.stem.replace(".execution", "") == exec_id:
            parent = f.parent
            traces: dict[str, list[dict[str, Any]]] = {}
            for trace_file in sorted(parent.glob("*.jsonl")):
                agent_name = trace_file.stem.split("_", 1)[-1] if "_" in trace_file.stem else trace_file.stem
                rows = []
                with open(trace_file) as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                            rows.append({
                                "step": row.get("step_index"),
                                "kind": row.get("step_kind"),
                                "agent": row.get("trace_agent"),
                                "model": row.get("model_name"),
                                "raw_content": row.get("raw_model_content", ""),
                                "parsed": _simplify_parsed(row.get("parsed")),
                            })
                        except Exception:
                            continue
                traces[agent_name] = rows
            return _json_response(traces)
    return jsonify({"error": "not found"}), 404


# ── Run management ──

@app.route("/api/run/start", methods=["POST"])
def start_run():
    config = request.get_json() or {}
    run_id = uuid.uuid4().hex[:12]

    agent_models = config.get("agent_models", ["gpt-5"])
    evaluator_model = config.get("evaluator_model", "gpt-5")
    num_participants = int(config.get("num_participants", 3))
    repeats = int(config.get("repeats", 1))
    batch_size = int(config.get("batch_size", 3))
    api_base = (config.get("api_base") or "").strip()
    api_key = (config.get("api_key") or "").strip()
    scenario_pks = config.get("scenario_pks") or None

    state = {
        "id": run_id,
        "status": "starting",
        "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": config,
        "total": len(agent_models) * repeats,
        "done": 0,
        "message": "Launching...",
        "episodes": [],
        "output_file": None,
    }
    with _lock:
        _active_runs[run_id] = state

    thread = threading.Thread(
        target=_run_eval_subprocess,
        args=(run_id, agent_models, evaluator_model, num_participants,
              repeats, batch_size, api_base, api_key, scenario_pks),
        daemon=True,
    )
    thread.start()

    return jsonify({"run_id": run_id, "status": "started"})


@app.route("/api/run/progress/<run_id>")
def run_progress(run_id: str):
    with _lock:
        state = _active_runs.get(run_id)
    if not state:
        return jsonify({"status": "unknown", "message": "Run not found"})
    result = {
        "status": state["status"], "message": state.get("message", ""),
        "total": state.get("total", 0), "done": state.get("done", 0),
        "episodes": state.get("episodes", []), "output_file": state.get("output_file"),
        "model_outputs": state.get("_model_outputs", [])[-50:],  # last 50 model outputs
    }
    conversations = state.get("_conversations", {})
    if conversations:
        latest_key = sorted(conversations.keys(), key=int)[-1]
        result["latest_conversation"] = conversations[latest_key]
        result["latest_conversation_seq"] = int(latest_key)
    return _json_response(result)


@app.route("/api/run/pause/<run_id>", methods=["POST"])
def pause_run(run_id: str):
    """Pause a running evaluation subprocess."""
    import signal
    with _lock:
        state = _active_runs.get(run_id)
    if not state:
        return jsonify({"error": "run not found"}), 404
    pid = state.get("_subprocess_pid")
    if not pid:
        return jsonify({"error": "no subprocess"}), 400
    try:
        os.kill(pid, signal.SIGSTOP)
        state["status"] = "paused"
        state["message"] = "Paused"
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({"status": "paused"})


@app.route("/api/run/resume/<run_id>", methods=["POST"])
def resume_run(run_id: str):
    """Resume a paused evaluation subprocess."""
    import signal
    with _lock:
        state = _active_runs.get(run_id)
    if not state:
        return jsonify({"error": "run not found"}), 404
    pid = state.get("_subprocess_pid")
    if not pid:
        return jsonify({"error": "no subprocess"}), 400
    try:
        os.kill(pid, signal.SIGCONT)
        state["status"] = "running"
        state["message"] = "Resumed"
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({"status": "resumed"})


@app.route("/api/conversation/<run_id>/<int:seq>")
def api_conversation(run_id: str, seq: int):
    with _lock:
        state = _active_runs.get(run_id)
    if state:
        conv = state.get("_conversations", {}).get(str(seq))
        if conv:
            return _json_response({"status": "ok", "conversation": conv})
    # Look in trace directories — first execution.json, then JSONL fallback
    for search_dir in [DATA_DIR / "live_traces", DATA_DIR]:
        if not search_dir.exists():
            continue
        for ex_file in sorted(search_dir.rglob("*.execution.json"), reverse=True):
            tag = ex_file.stem.replace(".execution", "")
            if f"_{seq}" in tag or tag.endswith(f"_{seq}") or f"_{seq}_" in tag:
                conv = _read_execution_conversation(ex_file.parent, tag)
                if conv:
                    return _json_response({"status": "ok", "conversation": conv})
        # Fallback: parse from JSONL (terminal_evaluator or no_agent traces)
        for jf in sorted(search_dir.rglob("*terminal_evaluator*.jsonl"), reverse=True):
            conv = _read_jsonl_conversation(jf)
            if conv:
                return _json_response({"status": "ok", "conversation": conv})
        for jf in sorted(search_dir.rglob("*no_agent*.jsonl"), reverse=True):
            conv = _read_jsonl_conversation(jf)
            if conv:
                return _json_response({"status": "ok", "conversation": conv})
    return jsonify({"status": "not_found"}), 404


# ── Subprocess-based evaluation ──

def _run_eval_subprocess(
    run_id: str, agent_models: list[str], evaluator_model: str,
    num_participants: int, repeats: int, batch_size: int,
    api_base: str, api_key: str, scenario_pks: list[str] | None,
) -> None:
    """Run evaluation via CLI subprocess (handles imports correctly)."""
    env = os.environ.copy()
    if api_base:
        env["OPENAI_API_BASE"] = api_base
    if api_key:
        env["CUSTOM_API_KEY"] = api_key
    env["SOTOPIA_STORAGE_BACKEND"] = "local"
    env["SOTOPIA_MAX_RENDERED_USER_CHARS"] = "0"

    live_trace_dir = DATA_DIR / "live_traces" / run_id
    live_trace_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "ltr_live"
    out_file = DATA_DIR  # use directory, CLI auto-generates filename

    # Use conda env Python (3.11) to avoid Python 3.9 type annotation incompatibility
    python_exe = os.environ.get("CONDA_PYTHON_EXE", "")
    _conda_candidates = [
        os.path.expanduser("~/.conda/envs/social_env/bin/python"),
        os.path.expanduser("~/.conda/envs/social_env/bin/python3"),
        "/home/yphao/.conda/envs/social_env/bin/python",
        "/home/yphao/.conda/envs/social_env/bin/python3.11",
    ]
    if not python_exe or not os.path.exists(python_exe):
        # Fallback: check sys.executable version
        import sys as _sys
        if _sys.version_info >= (3, 10):
            python_exe = _sys.executable
        else:
            for cand in _conda_candidates:
                if os.path.exists(cand):
                    python_exe = cand
                    break
    if not python_exe:
        python_exe = sys.executable
    # Ensure subprocess also sees the right Python via PATH
    env["CONDA_PYTHON_EXE"] = python_exe
    cmd = [
        python_exe, "-m", "sotopia.cli.benchmark.negotiation_batch",
        "negotiation-batch",
        "--agent-model", agent_models[0],
        "--evaluator-model", evaluator_model,
        "--batch-size", str(batch_size),
        "--repeats", str(repeats),
        "--num-participants", str(num_participants),
        "--tag", tag,
        "--output", str(out_file),
        "--execution-trace-dir", str(live_trace_dir),
        "--trace-flat",
        "--write-execution-record",
    ]
    if scenario_pks:
        for pk in scenario_pks:
            if pk:
                cmd.extend(["--scenario-env-pk", pk])

    print(f"[visualizer] Run {run_id}: subprocess: {' '.join(cmd)}", file=sys.stderr)

    try:
        with _lock:
            s = _active_runs.get(run_id)
            if s:
                s["status"] = "running"
                s["message"] = "Subprocess running..."
                s["_trace_dir"] = str(live_trace_dir)
                s["_out_file"] = str(out_file)

        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env=env, cwd=str(DATA_DIR.parent), text=True,
        )
        with _lock:
            s = _active_runs.get(run_id)
            if s:
                s["_subprocess_pid"] = proc.pid

        # Monitor: poll process, check execution.json + JSONL trace files
        last_check = time.time()
        seen_ex_files: set[str] = set()
        seen_jsonl_lines: dict[str, int] = {}
        while proc.poll() is None:
            time.sleep(2)
            now = time.time()
            if now - last_check >= 3:
                # Check execution.json files
                ex_files = set(str(p) for p in live_trace_dir.rglob("*.execution.json"))
                new_files = ex_files - seen_ex_files
                if new_files:
                    with _lock:
                        state = _active_runs.get(run_id)
                        if state:
                            state["done"] = len(ex_files)
                            state["message"] = f"{len(ex_files)} episodes completed"
                    for exf_path in sorted(new_files):
                        exf = Path(exf_path)
                        tag_match = exf.stem.replace(".execution", "")
                        conv = _read_execution_conversation(exf.parent, tag_match)
                        if conv:
                            parts = tag_match.rsplit("_", 1)
                            try:
                                seq = int(parts[-1])
                            except ValueError:
                                seq = len(state.get("_conversations", {})) if state else 0
                            with _lock:
                                st2 = _active_runs.get(run_id)
                                if st2:
                                    st2.setdefault("_conversations", {})[str(seq)] = conv
                    seen_ex_files = ex_files

                # Monitor JSONL trace files for real-time model outputs
                _watch_trace_files(live_trace_dir, run_id, seen_jsonl_lines)

                last_check = now

        # Process ended - final trace check
        _watch_trace_files(live_trace_dir, run_id, seen_jsonl_lines)

        # Process completed - find output file
        stdout_text = proc.stdout.read() if proc.stdout else ""
        # CLI generates file like: runs/negotiation_eval_{tag}_{ts}.json
        found_files = sorted(DATA_DIR.glob(f"negotiation_eval_{tag}*.json"), reverse=True)
        result_file = found_files[0] if found_files else None
        if proc.returncode == 0 and result_file and result_file.exists():
            with open(result_file) as fh:
                data = json.load(fh)
            rows = data.get("rows", [])
            with _lock:
                state = _active_runs.get(run_id)
                if state:
                    state["status"] = "done"
                    state["message"] = f"Completed: {len(rows)} episodes"
                    state["output_file"] = result_file.name
                    state["done"] = len(rows)
                    for i, row in enumerate(rows):
                        state["episodes"].append({
                            "seq": i,
                            "terminal": row.get("terminal", "?"),
                            "agent_model": row.get("agent_model", ""),
                            "scenario": row.get("scenario_codename", ""),
                        })
        else:
            with _lock:
                state = _active_runs.get(run_id)
                if state:
                    state["status"] = "failed"
                    state["message"] = f"Exit={proc.returncode}: {stdout_text[-400:]}"
        print(f"[visualizer] Run {run_id} done, exit={proc.returncode}", file=sys.stderr)

    except Exception as exc:
        import traceback
        print(f"[visualizer] Run {run_id} FAILED:\n{traceback.format_exc()}", file=sys.stderr)
        with _lock:
            state = _active_runs.get(run_id)
            if state:
                state["status"] = "failed"
                state["message"] = f"Error: {exc}"


def _watch_trace_files(trace_dir, run_id: str, seen: dict[str, int]):
    """Scan JSONL trace files for new model output lines (uses byte position to handle partial writes)."""
    for jf in sorted(trace_dir.rglob("*.jsonl")):
        fname = str(jf)
        prev_pos = seen.get(fname, 0)
        try:
            fsize = jf.stat().st_size
        except Exception:
            continue
        if fsize <= prev_pos:
            continue
        try:
            with open(jf) as fh:
                fh.seek(prev_pos)
                new_data = fh.read()
            new_pos = prev_pos + len(new_data)
        except Exception:
            continue
        new_entries = []
        for line in new_data.split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            raw = d.get("raw_model_content", "")
            parsed = d.get("parsed")
            # Extract prompt from messages
            msgs = d.get("messages", [])
            prompt_text = ""
            for m in msgs:
                if isinstance(m, dict) and m.get("role") == "user":
                    prompt_text = m.get("content", "")
                    break
            entry = {
                "step": d.get("step_index"),
                "agent": d.get("trace_agent", "?"),
                "kind": d.get("step_kind", ""),
                "raw": raw if raw else "",
                "prompt": prompt_text if prompt_text else "",
                "parsed": _simplify_parsed(parsed),
            }
            new_entries.append(entry)
        seen[fname] = new_pos
        if new_entries:
            with _lock:
                state = _active_runs.get(run_id)
                if state:
                    outputs = state.setdefault("_model_outputs", [])
                    outputs.extend(new_entries)
                    state["_last_update"] = time.time()
                    if len(outputs) > 200:
                        state["_model_outputs"] = outputs[-200:]


def _read_jsonl_conversation(jsonl_path) -> dict[str, Any] | None:
    """Parse scheduling/session log from a JSONL trace file (fallback when no execution.json)."""
    try:
        with open(jsonl_path) as fh:
            lines = fh.readlines()
    except Exception:
        return None

    # Collect all content from user messages across all steps
    all_sched = []
    all_sessions = []
    agents_seen = set()
    for line in lines:
        try:
            d = json.loads(line)
        except Exception:
            continue
        msgs = d.get("messages", [])
        for m in msgs:
            if not isinstance(m, dict) or m.get("role") != "user":
                continue
            content = m.get("content", "")
            # Parse scheduling log
            if "# Scheduling" in content:
                idx_sched = content.find("# Scheduling")
                idx_sess = content.find("# Session log", idx_sched) if "# Session log" in content else len(content)
                idx_action = content.find("# Action log", idx_sess) if "# Action log" in content else len(content)
                sched_part = content[idx_sched + len("# Scheduling"):idx_sess].strip()
                sess_part = content[idx_sess + len("# Session log"):idx_action].strip()
                for sline in sched_part.split("\n"):
                    sline = sline.strip()
                    if not sline:
                        continue
                    # Parse "day=X slot=Y | AgentName: ..."
                    import re
                    m2 = re.match(r'day=(\d+)\s+slot=(\d+)\s*\|\s*([^:]+):\s*(.*)', sline)
                    if m2:
                        all_sched.append({
                            "day": int(m2.group(1)),
                            "slot": int(m2.group(2)),
                            "agent": m2.group(3).strip(),
                            "action": m2.group(4).strip(),
                        })
                        agents_seen.add(m2.group(3).strip())
                # Parse session log — each line is a JSON object
                for sline in sess_part.split("\n"):
                    sline = sline.strip()
                    if not sline or sline.startswith("#"):
                        continue
                    try:
                        sobj = json.loads(sline)
                        if isinstance(sobj, dict):
                            sess = {
                                "kind": sobj.get("kind", ""),
                                "day": sobj.get("day"),
                                "slot": sobj.get("slot"),
                                "reason": sobj.get("slot_closure_reason", ""),
                                "transcripts": [],
                            }
                            for t in sobj.get("transcript", []):
                                if isinstance(t, dict):
                                    sess["transcripts"].append(t)
                            for cs in sobj.get("closed_sessions_detail", []):
                                if isinstance(cs, dict):
                                    sp = {"session_id": cs.get("session_id", ""),
                                          "participants": cs.get("participants", []),
                                          "transcript": cs.get("transcript", []),
                                          "contracts": cs.get("negotiation_contracts_snapshot", [])}
                                    if sp["transcript"]:
                                        sess["transcripts"].append(sp)
                            if sess["transcripts"] or sess["reason"]:
                                all_sessions.append(sess)
                    except json.JSONDecodeError:
                        pass

    if not all_sched and not all_sessions:
        return None

    return {
        "agents": sorted(agents_seen),
        "scheduling": all_sched,
        "sessions": all_sessions,
        "contracts": [],
        "terminal": "?",
    }


# ── Conversation parsing ──

def _read_execution_conversation(trace_dir, tag: str) -> dict[str, Any] | None:
    import glob as _glob
    pattern = str(Path(trace_dir) / f"{tag}*.execution.json")
    matches = sorted(_glob.glob(pattern))
    if not matches:
        return None
    try:
        with open(matches[0]) as fh:
            data = json.load(fh)
    except Exception:
        return None

    conv: dict[str, Any] = {
        "agents": list(data.get("visible_history_by_agent", {}).keys()),
        "scheduling": [], "sessions": [], "contracts": [],
        "terminal": data.get("terminal", "?"),
    }
    for entry in data.get("scheduling_log", []):
        if isinstance(entry, (list, tuple)) and len(entry) >= 4:
            conv["scheduling"].append({"day": entry[0], "slot": entry[1], "agent": entry[2], "action": entry[3]})
    for entry in data.get("session_log", []):
        if not isinstance(entry, dict):
            continue
        sess = {"kind": entry.get("kind", ""), "day": entry.get("day"), "slot": entry.get("slot"),
                "reason": entry.get("slot_closure_reason", ""), "transcripts": [], "contracts": []}
        for t in entry.get("transcript", []):
            if isinstance(t, dict):
                sess["transcripts"].append(t)
        for cs in entry.get("closed_sessions_detail", []):
            if isinstance(cs, dict):
                sp = {"session_id": cs.get("session_id", ""), "participants": cs.get("participants", []),
                      "transcript": [], "contracts": cs.get("negotiation_contracts_snapshot", [])}
                for msg in cs.get("transcript", []):
                    if isinstance(msg, dict):
                        sp["transcript"].append(msg)
                if sp["transcript"]:
                    sess["transcripts"].append(sp)
                conv["contracts"].extend(sp["contracts"])
        conv["contracts"].extend(entry.get("negotiation_contracts_snapshot", []))
        if sess["transcripts"] or entry.get("slot_closure_reason"):
            conv["sessions"].append(sess)
    conv["events"] = [ev for ev in data.get("event_log", []) if isinstance(ev, dict)]
    return conv if (conv["scheduling"] or conv["sessions"]) else None


# ── Helpers ──

def _format_schedule(log: list) -> list[dict[str, Any]]:
    out = []
    for entry in log:
        if isinstance(entry, (list, tuple)) and len(entry) >= 4:
            out.append({"day": entry[0], "slot": entry[1], "agent": entry[2], "action": entry[3]})
    return out


def _simplify_parsed(parsed: Any) -> Any:
    if parsed is None:
        return None
    if isinstance(parsed, dict):
        return {k: _simplify_parsed(v) for k, v in list(parsed.items())[:20]}
    if isinstance(parsed, list):
        return [_simplify_parsed(v) for v in parsed[:10]]
    if isinstance(parsed, str):
        return parsed
    return parsed


def _json_response(data: Any):
    return app.response_class(
        response=json.dumps(data, ensure_ascii=False, default=str),
        mimetype="application/json",
    )


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Agent Interaction Visualizer")
    parser.add_argument("--port", type=int, default=18090)
    parser.add_argument("--data-dir", type=str, default="runs")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()
    global DATA_DIR
    DATA_DIR = Path(args.data_dir).resolve()
    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[visualizer] Data dir : {DATA_DIR}")
    print(f"[visualizer] Listening : http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()

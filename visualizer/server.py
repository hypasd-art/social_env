"""Agent interaction visualizer — Flask server.

Usage:
    python visualizer/server.py [--port PORT] [--data-dir RUNS_DIR]
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

# Default data directory
DATA_DIR = Path(os.environ.get("VIS_DATA_DIR", "runs"))


def _scan_runs() -> list[dict[str, Any]]:
    """Scan data directory for evaluation JSON files and trace directories."""
    runs: list[dict[str, Any]] = []
    if not DATA_DIR.exists():
        return runs

    # 1. Find evaluation JSON files (aggregate results)
    for f in sorted(DATA_DIR.rglob("*.json"), reverse=True):
        if f.suffix != ".json" or "execution" in f.stem or "agent_episode" in f.stem:
            continue
        try:
            with open(f) as fh:
                data = json.load(fh)
            rows = data.get("rows", [])
            agg = data.get("aggregate_means", {})
            overall = agg.get("overall", agg)  # handle both flat and nested
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

    # 2. Find execution.json files (individual episode records)
    exec_records: list[dict[str, Any]] = []
    for f in sorted(DATA_DIR.rglob("*.execution.json"), reverse=True):
        try:
            with open(f) as fh:
                data = json.load(fh)
            exec_records.append({
                "id": f.stem.replace(".execution", ""),
                "path": str(f.relative_to(DATA_DIR)),
                "type": "execution",
                "terminal": data.get("terminal", "?"),
                "macro_steps": data.get("macro_steps_used", 0),
                "n_agents": len(data.get("visible_history_by_agent", {})),
                "agents": list(data.get("visible_history_by_agent", {}).keys()),
            })
        except Exception:
            continue

    return runs


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/runs")
def api_runs():
    return jsonify(_scan_runs())


@app.route("/api/run/<run_id>")
def api_run_detail(run_id: str):
    """Load a full evaluation JSON."""
    for f in DATA_DIR.rglob("*.json"):
        if f.stem == run_id:
            with open(f) as fh:
                data = json.load(fh)
            return _json_response(data)
    return jsonify({"error": "not found"}), 404


@app.route("/api/execution/<exec_id>")
def api_execution(exec_id: str):
    """Load an execution.json for detailed episode view."""
    for f in DATA_DIR.rglob("*.execution.json"):
        if f.stem.replace(".execution", "") == exec_id:
            with open(f) as fh:
                data = json.load(fh)
            # Extract key sections for the UI
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
                "llm_model_traces": data.get("llm_model_traces", {}),
            }
            return _json_response(result)
    return jsonify({"error": "not found"}), 404


@app.route("/api/trace/<exec_id>")
def api_trace(exec_id: str):
    """Load JSONL trace files for an execution directory."""
    # Find the directory containing the execution.json
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
                            # Extract just the displayable parts
                            rows.append({
                                "step": row.get("step_index"),
                                "kind": row.get("step_kind"),
                                "agent": row.get("trace_agent"),
                                "model": row.get("model_name"),
                                "raw_content": row.get("raw_model_content", "")[:5000],
                                "parsed": _simplify_parsed(row.get("parsed")),
                            })
                        except Exception:
                            continue
                traces[agent_name] = rows
            return _json_response(traces)
    return jsonify({"error": "not found"}), 404


def _format_schedule(log: list) -> list[dict[str, Any]]:
    """Format scheduling log entries."""
    out = []
    for entry in log:
        if isinstance(entry, (list, tuple)) and len(entry) >= 4:
            out.append({
                "day": entry[0],
                "slot": entry[1],
                "agent": entry[2],
                "action": entry[3],
            })
    return out


def _simplify_parsed(parsed: Any) -> Any:
    """Truncate large parsed structures."""
    if parsed is None:
        return None
    if isinstance(parsed, dict):
        return {k: _simplify_parsed(v) for k, v in list(parsed.items())[:20]}
    if isinstance(parsed, list):
        return [_simplify_parsed(v) for v in parsed[:10]]
    if isinstance(parsed, str) and len(parsed) > 2000:
        return parsed[:2000] + "..."
    return parsed


def _json_response(data: Any):
    """Ensure all values are JSON-serializable."""
    return app.response_class(
        response=json.dumps(data, ensure_ascii=False, default=str),
        mimetype="application/json",
    )


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Agent Interaction Visualizer")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--data-dir", type=str, default="runs")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    global DATA_DIR
    DATA_DIR = Path(args.data_dir).resolve()
    if not DATA_DIR.exists():
        print(f"[visualizer] WARNING: data directory does not exist: {DATA_DIR}")

    print(f"[visualizer] Data dir : {DATA_DIR}")
    print(f"[visualizer] Listening : http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()

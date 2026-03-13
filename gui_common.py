"""
Shared logic for CellVoyager GUI — constants, session state, helper functions.
Imported by gui.py (home) and pages/analysis.py (analysis view).
"""
import base64
import html
import json
import os
import signal
import uuid
import subprocess
import sys
import threading
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import markdown as _markdown
import streamlit as st


def _launch_resume(output_dir: str, analysis_idx: int, run_to_completion: bool = False) -> None:
    """Launch run_v2 in resume mode for the given analysis. Sets session state and reruns.
    If run_to_completion=True (e.g. when resuming from Stop), use high intervene_every so the agent
    runs to completion without pausing for feedback."""
    out_dir = Path(output_dir)
    # Clear stale pause/step/stop files so we don't show wrong UI from previous stopped run
    for f in (_PAUSE_REQUEST_FILE, _PAUSE_RESPONSE_FILE, _STEP_COUNT_FILE, _STOP_REQUEST_FILE):
        (out_dir / f).unlink(missing_ok=True)
    if run_to_completion:
        cfg_path = out_dir / _RUN_CONFIG_FILE
        if cfg_path.exists():
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                cfg["intervene_every_restore"] = cfg.get("intervene_every", 1)
                cfg["intervene_every"] = 999
                cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
            except Exception:
                pass
    cmd = [
        sys.executable, str(ROOT / "run_v2.py"),
        "--resume",
        "--resume-output-dir", output_dir,
        "--resume-analysis-idx", str(analysis_idx),
    ]
    if run_to_completion:
        cmd.extend(["--resume-intervene-every", "999"])
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["CELLVOYAGER_GUI_INTERACTIVE"] = "1"
    proc = subprocess.Popen(
        cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env, start_new_session=True,
    )
    (out_dir / _RUN_PID_FILE).write_text(str(proc.pid), encoding="utf-8")
    log_path = out_dir / _RUN_LOG_FILE
    log_path.write_text("", encoding="utf-8")  # Clear stale output from previous run
    st.session_state.run_proc = proc
    st.session_state.run_pid = proc.pid
    st.session_state.run_output = []
    st.session_state.run_output_dir = output_dir
    st.session_state.run_started = True
    st.session_state.run_interactive_mode = True
    st.session_state.run_thread_started = True
    st.session_state._is_resuming = True  # cleared once the first pause arrives
    t = threading.Thread(target=_read_output, args=(proc, st.session_state.run_output, log_path))
    t.daemon = True
    t.start()

# Paths
ROOT = Path(__file__).resolve().parent
UPLOADS_DIR = ROOT / "gui_uploads"
OUTPUTS_BASE = ROOT / "outputs"
_LAST_RUN_FILE = OUTPUTS_BASE / ".last_run"

# Session state
for key, default in [
    ("run_proc", None),
    ("run_output", []),
    ("run_cmd", None),
    ("run_started", False),
    ("run_thread_started", False),
    ("run_output_dir", None),
    ("run_interactive_mode", False),
    ("run_pid", None),
    ("run_num_analyses", 1),
    ("run_show_interactive", False),
    ("home_paper_text", ""),
    ("home_context_source", "Structured fields"),
    ("home_dataset_summary", ""),
    ("home_past_analyses", ""),
    ("home_focus_directions", ""),
    ("home_bio_background", ""),
    ("home_analysis_name", "covid19"),
    ("home_num_analyses", 1),
    ("home_max_iterations", 8),

    ("home_interactive_mode", False),
    ("home_intervene_every", 1),
    ("home_ding_on_pause", False),
    ("home_use_deepresearch", False),
    ("home_model_name", "o3-mini"),
    ("session_runs", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

_PAUSE_REQUEST_FILE = ".cellvoyager_pause_request"
_PAUSE_RESPONSE_FILE = ".cellvoyager_pause_response"
_EXECUTE_REQUEST_FILE = ".cellvoyager_execute_request"
_STEP_COUNT_FILE = ".cellvoyager_step_count"
_STOP_REQUEST_FILE = ".cellvoyager_stop_request"
_AGENT_SUMMARY_FILE = ".cellvoyager_agent_summary"
_CHAT_REQUEST_FILE = ".cellvoyager_chat_request"
_CHAT_RESPONSE_FILE = ".cellvoyager_chat_response"
_CHAT_HISTORY_PREFIX = ".chat_history"
_RUN_PID_FILE = ".run_pid"
_RUN_LOG_FILE = ".run_log"
_RUN_INTERACTIVE_FILE = ".run_interactive"
_RUN_CONFIG_FILE = ".run_config.json"
_RUN_ERROR_FILE = ".run_error"
_LAST_DISPLAYED_DIR = ".last_displayed"


def _chat_history_file(output_dir: str, analysis_idx: int | None) -> Path | None:
    if not output_dir:
        return None
    suffix = f"_analysis_{analysis_idx}" if analysis_idx is not None else ""
    return Path(output_dir) / f"{_CHAT_HISTORY_PREFIX}{suffix}.json"


def _load_chat_history(output_dir: str, analysis_idx: int | None) -> list:
    path = _chat_history_file(output_dir, analysis_idx)
    if path and path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return []


def _save_chat_history(output_dir: str, analysis_idx: int | None, messages: list) -> None:
    path = _chat_history_file(output_dir, analysis_idx)
    if path:
        try:
            path.write_text(json.dumps(messages), encoding="utf-8")
        except Exception:
            pass


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _parse_run_progress(log_text: str) -> tuple[int, int]:
    """Parse run log for analysis progress. Returns (completed_count, current_analysis).
    completed_count = number of '✅ Completed Analysis N' seen.
    current_analysis = 1 + completed_count, or the one from '🚀 Generated Initial Analysis Plan for Analysis N' if we haven't completed that yet.
    """
    import re
    completed = 0
    current = 1
    for m in re.finditer(r"Completed Analysis (\d+)", log_text):
        completed = max(completed, int(m.group(1)))
    for m in re.finditer(r"Generated Initial Analysis Plan for Analysis (\d+)", log_text):
        current = max(current, int(m.group(1)))
    if completed > 0 and current <= completed:
        current = completed + 1
    return completed, current


def _get_run_log() -> str:
    out_dir = st.session_state.get("run_output_dir")
    if not out_dir:
        return ""
    # Prefer in-memory output when we have it (live capture)
    if st.session_state.get("run_proc") is not None and st.session_state.get("run_output"):
        return "".join(st.session_state.get("run_output", []))
    # Fallback to log file (e.g. when run_output not yet populated, or after process exit)
    log_path = Path(out_dir) / _RUN_LOG_FILE
    if log_path.exists():
        try:
            return log_path.read_text(encoding="utf-8")
        except Exception:
            pass
    return ""


def _has_live_run() -> bool:
    proc = st.session_state.get("run_proc")
    if proc is not None and proc.poll() is None:
        return True
    pid = st.session_state.get("run_pid")
    if pid is None:
        out_dir = st.session_state.get("run_output_dir")
        if out_dir:
            pid_file = Path(out_dir) / _RUN_PID_FILE
            if pid_file.exists():
                try:
                    pid = int(pid_file.read_text(encoding="utf-8").strip())
                    st.session_state.run_pid = pid
                except (ValueError, OSError):
                    pid = None
    if pid is not None and _process_alive(pid):
        return True
    return False


def _extract_agent_summary_from_notebook(nb_path: Path) -> str:
    """Extract a brief summary from the notebook for the pause UI."""
    import re
    try:
        import nbformat as nbf
        nb = nbf.read(str(nb_path), as_version=4)
    except Exception:
        return "Paused by user."
    _FEEDBACK = "## 📝 Your feedback"
    last_interp = None
    last_step = None
    for cell in nb.cells:
        if cell.cell_type != "markdown":
            continue
        src = getattr(cell, "source", "") or ""
        text = "\n".join(src) if isinstance(src, list) else str(src)
        text = text.strip()
        if not text or text.startswith(_FEEDBACK):
            continue
        first = text.split("\n")[0].lower()
        if "interpretation" in text:
            last_interp = text
        elif "step" in first:
            last_step = text
    cand = last_interp or last_step
    if not cand:
        return "Paused by user."
    for pat in (r"\*\*Plan going forward[^*]*\*\*[:\s]*\n?(.+?)(?=\n\n|\n\*\*|$)", r"\*\*What the output shows[^*]*\*\*[:\s]*\n?(.+?)(?=\n\n|\n\*\*|$)"):
        m = re.search(pat, cand, re.DOTALL)
        if m:
            block = m.group(1).strip()
            bullets = [l.strip().lstrip("-*").strip() for l in block.split("\n") if l.strip() and len(l.strip()) > 8][:5]
            if bullets:
                return "\n".join(f"- {b}" for b in bullets)
    return "Paused by user."


def _request_pause() -> bool:
    """Request the agent to pause (Stop button).

    This is a *pause request only*. It writes STOP so the backend pauses at the next
    safe boundary. The backend then writes the pause-request file when it is actually
    blocked and ready for user feedback.
    """
    out_dir_str = st.session_state.get("run_output_dir")
    if not out_dir_str:
        return False
    out_dir = Path(out_dir_str).resolve()
    if not out_dir.exists():
        return False
    # Agent sees this and pauses at next tool boundary.
    (out_dir / _STOP_REQUEST_FILE).write_text("1", encoding="utf-8")
    # Clear stale response from prior pause so backend waits for a fresh Continue/Finish.
    (out_dir / _PAUSE_RESPONSE_FILE).unlink(missing_ok=True)
    # Optional precomputed summary if a notebook already exists.
    notebooks = sorted(out_dir.glob("*.ipynb"), key=lambda p: p.stat().st_mtime, reverse=True)
    if notebooks:
        summary = _extract_agent_summary_from_notebook(notebooks[0])
        (out_dir / _AGENT_SUMMARY_FILE).write_text(summary, encoding="utf-8")
    return True


def _kill_analysis(show_interactive_edit=True):
    """Kill the analysis process immediately (process group). Only used when pause is not possible."""
    proc = st.session_state.get("run_proc")
    pid = st.session_state.get("run_pid")
    out_dir = Path(st.session_state.get("run_output_dir", "")) if st.session_state.get("run_output_dir") else None
    pid_to_kill = (proc.pid if proc is not None else None) or pid
    if pid_to_kill is not None:
        try:
            pgid = os.getpgid(pid_to_kill)
            os.killpg(pgid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            try:
                if proc is not None:
                    proc.terminate()
                    proc.wait(timeout=5)
                elif pid is not None:
                    os.kill(pid, signal.SIGTERM)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    if proc is not None:
                        proc.kill()
                    elif pid is not None:
                        os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass
        time.sleep(0.3)
        if _process_alive(pid_to_kill):
            try:
                pgid = os.getpgid(pid_to_kill)
                os.killpg(pgid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass
    out_dir_str = st.session_state.get("run_output_dir")
    keys_to_drop = [k for k in st.session_state.keys() if isinstance(k, str) and out_dir_str and k.startswith("agent_chat_") and out_dir_str in k]
    for k in keys_to_drop:
        st.session_state.pop(k, None)
    st.session_state.run_proc = None
    st.session_state.run_output = []
    st.session_state.run_pid = None
    st.session_state.run_started = False
    st.session_state.run_thread_started = False
    st.session_state.run_cmd = None
    if out_dir and out_dir.exists():
        (out_dir / _RUN_PID_FILE).unlink(missing_ok=True)
    if show_interactive_edit and out_dir and out_dir.exists():
        st.session_state.run_show_interactive = True
        _restore_last_displayed(out_dir)
        notebooks = sorted(out_dir.glob("*.ipynb"), key=lambda p: p.stat().st_mtime, reverse=True)
        if notebooks:
            (out_dir / _PAUSE_REQUEST_FILE).write_text(str(notebooks[0]), encoding="utf-8")
    else:
        st.session_state.run_output_dir = None
        st.session_state.run_show_interactive = False


def _read_output(proc, output_list, log_path=None):
    for line in proc.stdout:
        output_list.append(line)
        if log_path:
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(line)
            except Exception:
                pass
    proc.stdout.close()


def _collect_notebooks(output_dir_filter=None):
    if output_dir_filter:
        d = Path(output_dir_filter)
        if d.exists() and d.is_dir():
            return [(d.name, str(nb)) for nb in d.glob("*.ipynb")]
    return []


def _collect_notebooks_by_analysis(output_dir_filter=None, num_analyses=None):
    """Return list of (analysis_idx_1based, nb_path or None) for each analysis.
    Notebooks are matched by *_analysis_N.ipynb. If num_analyses given, returns slots 1..N.
    """
    import re
    result = {}  # idx -> path
    if output_dir_filter:
        d = Path(output_dir_filter)
        if d.exists() and d.is_dir():
            for nb in d.glob("*.ipynb"):
                m = re.search(r"_analysis_(\d+)\.ipynb$", nb.name, re.I)
                if m:
                    result[int(m.group(1))] = str(nb)
    n = num_analyses or (max(result.keys()) if result else 1)
    return [(i, result.get(i)) for i in range(1, n + 1)]


def _cell_source_str(cell):
    src = cell.source
    return "\n".join(src) if isinstance(src, list) else (src or "")


def _is_step_summary(cell):
    if cell.cell_type != "markdown":
        return False
    src = _cell_source_str(cell).strip()
    first_line = src.split("\n")[0].strip() if src else ""
    return "summary" in first_line.lower()


def _step_separator(step_label):
    color = "#0d7377"
    st.markdown(
        f'''
        <div style="display: flex; align-items: center; gap: 1rem; margin: 1.25rem 0;">
            <div style="flex: 1; height: 4px; background: linear-gradient(90deg, {color}, #14a3a8); border-radius: 2px;"></div>
            <span style="font-weight: 700; color: {color}; font-size: 3rem; white-space: nowrap;">STEP {step_label}</span>
            <div style="flex: 1; height: 4px; background: linear-gradient(90deg, #14a3a8, {color}); border-radius: 2px;"></div>
        </div>
        ''',
        unsafe_allow_html=True,
    )


# Keep plain-text outputs (including printed tables) in monospace for alignment.
_TEXT_OUTPUT_STYLE = (
    "margin: 0; white-space: pre-wrap; word-break: break-word; "
    "font-family: ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Monaco, Consolas, "
    "'Liberation Mono', 'Courier New', monospace; font-size: 1.25rem; line-height: 1.5;"
)


def _normalize_output_text(text: Any) -> str:
    if isinstance(text, list):
        text = "".join(str(t) for t in text)
    else:
        text = str(text)
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _render_cell_outputs(cell, show_success_toast: bool = False):
    if cell.cell_type != "code":
        return
    outputs = getattr(cell, "outputs", []) or []
    if not outputs:
        return
    parts = []
    for out in outputs:
        ot = out.get("output_type", "")
        if ot == "stream":
            text = _normalize_output_text(out.get("text", ""))
            if text.strip():
                name = out.get("name", "stdout")
                color = "inherit" if name == "stdout" else "#c53030"
                parts.append(
                    f'<div style="{_TEXT_OUTPUT_STYLE} color:{color};">{html.escape(text)}</div>'
                )
        elif ot == "execute_result":
            data = out.get("data", {})
            exec_count = out.get("execution_count")
            prefix = f'<span style="font-style:italic;color:#6b7280;">Out[{exec_count}]:</span> ' if exec_count is not None else ""
            if "text/html" in data:
                html_val = data["text/html"]
                if isinstance(html_val, list):
                    html_val = "".join(str(h) for h in html_val)
                else:
                    html_val = str(html_val)
                parts.append(f'<div class="output_html">{prefix}<div style="margin-top:0.25em;">{html_val}</div></div>')
            elif "text/plain" in data:
                plain = _normalize_output_text(data["text/plain"])
                parts.append(f'<div style="{_TEXT_OUTPUT_STYLE}">{prefix}{html.escape(plain)}</div>')
            if "image/svg+xml" in data:
                svg = data["image/svg+xml"]
                if isinstance(svg, list):
                    svg = "".join(str(s) for s in svg)
                parts.append(f'<div style="margin-top:0.5em;">{svg}</div>')
            if "image/png" in data:
                try:
                    b64 = data["image/png"]
                    if isinstance(b64, list):
                        b64 = "".join(b64)
                    parts.append(f'<div style="margin-top:0.5em;"><img src="data:image/png;base64,{b64}" style="max-width:100%;height:auto;" /></div>')
                except Exception:
                    parts.append('<span style="color:#6b7280;font-size:1.25rem;">[Image output]</span>')
        elif ot == "display_data":
            data = out.get("data", {})
            if "text/html" in data:
                html_val = data["text/html"]
                if isinstance(html_val, list):
                    html_val = "".join(str(h) for h in html_val)
                else:
                    html_val = str(html_val)
                parts.append(f'<div class="output_html"><div style="margin-top:0.25em;">{html_val}</div></div>')
            elif "text/plain" in data:
                plain = _normalize_output_text(data["text/plain"])
                parts.append(f'<div style="{_TEXT_OUTPUT_STYLE}">{html.escape(plain)}</div>')
            if "image/svg+xml" in data:
                svg = data["image/svg+xml"]
                if isinstance(svg, list):
                    svg = "".join(str(s) for s in svg)
                parts.append(f'<div style="margin-top:0.5em;">{svg}</div>')
            if "image/png" in data:
                try:
                    b64 = data["image/png"]
                    if isinstance(b64, list):
                        b64 = "".join(b64)
                    parts.append(f'<div style="margin-top:0.5em;"><img src="data:image/png;base64,{b64}" style="max-width:100%;height:auto;" /></div>')
                except Exception:
                    parts.append('<span style="color:#6b7280;font-size:1.25rem;">[Image output]</span>')
        elif ot == "error":
            tb = out.get("traceback", [])
            if isinstance(tb, list):
                tb = "\n".join(str(t) for t in tb)
            tb = _normalize_output_text(tb)
            parts.append(
                f'<div style="{_TEXT_OUTPUT_STYLE} color:#c53030; background:rgba(254,226,226,0.4); padding:0.5rem; border-radius:4px;">{html.escape(tb)}</div>'
            )
    if not parts:
        return
    inner = "\n".join(parts)
    toast_html = (
        '<div class="cellvoyager-success-toast cellvoyager-success-toast-overlay">Code cell successfully executed</div>'
        if show_success_toast
        else ""
    )
    container_html = (
        '<div class="cv-cell-output" style="position:relative; margin-top:0.35rem; border:1px solid rgba(151,166,175,0.35); '
        'border-radius:6px; background:#ffffff; padding:0.6rem 0.75rem; cursor:pointer;" '
        'title="Click to toggle scrollable view">'
        f"{toast_html}{inner}"
        "</div>"
    )
    st.markdown(
        container_html,
        unsafe_allow_html=True,
    )


def _render_cell_display(cell):
    src = _cell_source_str(cell)
    if cell.cell_type == "markdown":
        st.markdown(src)
    else:
        st.code(src, language="python")
        _render_cell_outputs(cell)


def _render_editable_cell(cell, cell_idx, pause_id, nb_path, use_button=False):
    src = _cell_source_str(cell)
    height = min(300, max(80, 60 + len(src.splitlines()) * 18))
    editor_key = f"pause_cell_{pause_id}_{cell_idx}"
    ace_key = f"ace_{editor_key}"
    editing_key = f"pause_cell_editing_{pause_id}_{cell_idx}"
    saved_key = f"pause_cell_saved_at_{pause_id}_{cell_idx}"

    if editor_key not in st.session_state:
        st.session_state[editor_key] = src
        st.session_state.pop(ace_key, None)
    elif not st.session_state.get(editing_key, False):
        # When not actively editing, sync the editor cache to latest notebook source.
        st.session_state[editor_key] = src
        st.session_state.pop(ace_key, None)

    editing = st.session_state.get(editing_key, False)

    run_success = st.session_state.get("_cell_run_success")
    is_success_this_cell = (
        isinstance(run_success, dict)
        and run_success.get("nb_path") == nb_path
        and run_success.get("pause_id") == pause_id
        and int(run_success.get("cell_index", -1)) == cell_idx
        and (time.time() - float(run_success.get("completed_at", 0.0))) <= 10.0
    )

    with st.container(border=True):
        if cell.cell_type == "code":
            if editing:
                try:
                    from streamlit_ace import st_ace  # Optional syntax-highlighted editor
                    ace_height = min(
                        900,
                        max(
                            170,
                            36 + len((st.session_state.get(editor_key, src) or "").splitlines()) * 22,
                        ),
                    )

                    editor = st_ace(
                        value=st.session_state.get(editor_key, src),
                        language="python",
                        theme="github",
                        height=ace_height,
                        key=ace_key,
                        auto_update=True,
                        wrap=True,
                        show_gutter=False,
                        font_size=15,
                    )
                    if editor is None:
                        editor = st.session_state.get(editor_key, src)
                    st.session_state[editor_key] = editor
                except Exception:
                    editor = st.text_area(
                        "Code source",
                        height=height,
                        key=editor_key,
                        label_visibility="collapsed",
                    )
                _btn = st.button if use_button else st.form_submit_button
                if _btn("Save", key=f"save_cell_{pause_id}_{cell_idx}"):
                    st.session_state["_cell_inline_edit_toggled"] = True
                    st.session_state[saved_key] = time.time()
                    st.session_state[editing_key] = False
                saved_at = st.session_state.get(saved_key)
                if saved_at:
                    st.caption(f"Saved at {time.strftime('%H:%M:%S', time.localtime(saved_at))}")
            else:
                st.code(st.session_state.get(editor_key, src), language="python")
                _btn = st.button if use_button else st.form_submit_button
                if _btn("✏️ Edit cell", key=f"edit_cell_{pause_id}_{cell_idx}"):
                    st.session_state[editing_key] = True
                    st.session_state["_cell_inline_edit_toggled"] = True
                editor = st.session_state.get(editor_key, src)
            _render_cell_outputs(cell, show_success_toast=is_success_this_cell)
        else:
            if editing:
                editor = st.text_area(
                    "Markdown source",
                    height=height,
                    key=editor_key,
                    label_visibility="collapsed",
                )
                _btn = st.button if use_button else st.form_submit_button
                if _btn("Save", key=f"save_cell_{pause_id}_{cell_idx}"):
                    st.session_state["_cell_inline_edit_toggled"] = True
                    st.session_state[saved_key] = time.time()
                    st.session_state[editing_key] = False
                saved_at = st.session_state.get(saved_key)
                if saved_at:
                    st.caption(f"Saved at {time.strftime('%H:%M:%S', time.localtime(saved_at))}")
            else:
                st.markdown(st.session_state.get(editor_key, src))
                _btn = st.button if use_button else st.form_submit_button
                if _btn("✏️ Edit cell", key=f"edit_cell_{pause_id}_{cell_idx}"):
                    st.session_state[editing_key] = True
                    st.session_state["_cell_inline_edit_toggled"] = True
                editor = st.session_state.get(editor_key, src)
    return editor


def _save_last_displayed_snapshot(nb_path: str, nb, output_dir: str | None) -> None:
    """Save a snapshot of the notebook for restore-on-stop. Called when rendering during run."""
    if not output_dir:
        return
    out = Path(output_dir)
    snap_dir = out / _LAST_DISPLAYED_DIR
    snap_dir.mkdir(exist_ok=True)
    nb_name = Path(nb_path).name
    try:
        import nbformat as nbf
        with open(snap_dir / nb_name, "w", encoding="utf-8") as f:
            nbf.write(nb, f)
    except Exception:
        pass


def _restore_last_displayed(output_dir: Path) -> None:
    """Overwrite notebooks with last-displayed snapshots (used when user stops analysis)."""
    snap_dir = output_dir / _LAST_DISPLAYED_DIR
    if not snap_dir.exists():
        return
    for snap in snap_dir.glob("*.ipynb"):
        target = output_dir / snap.name
        if target.exists():
            try:
                import shutil
                shutil.copy2(snap, target)
            except Exception:
                pass


def _render_notebook_jupyter_style(nb_path, editable=False, pause_id=None, standalone_edit=False, save_snapshot=False, output_dir=None, sidebar_actions=False):
    """Render notebook. When standalone_edit=True (completed analyses), show Edit/Save only; no agent feedback."""
    import nbformat as nbf
    try:
        with open(nb_path, encoding="utf-8") as f:
            content = f.read()
        if not content.strip():
            st.warning(f"Notebook is empty or still being written: `{Path(nb_path).name}`")
            return None
        nb = nbf.reads(content, as_version=4)
        if save_snapshot and output_dir:
            _save_last_displayed_snapshot(nb_path, nb, output_dir)
    except (nbf.reader.NotJSONError, ValueError, Exception) as e:
        st.warning(f"Could not load notebook `{Path(nb_path).name}` (empty or invalid): {e}")
        return None
    edit_mode = editable and pause_id and st.session_state.get(f"pause_edit_mode_{pause_id}", False)
    if edit_mode:
        st.markdown(
            """
            <style>
            @keyframes cellvoyager-success-fade {
                0% { opacity: 0; transform: translateY(4px); }
                8% { opacity: 1; transform: translateY(0); }
                80% { opacity: 1; transform: translateY(0); }
                100% { opacity: 0; transform: translateY(-2px); }
            }
            .cellvoyager-success-toast {
                border: 1px solid #86efac;
                background: #ecfdf3;
                color: #166534;
                border-radius: 10px;
                padding: 0.5rem 0.75rem;
                font-size: 1.25rem;
                font-weight: 600;
                animation: cellvoyager-success-fade 10s ease forwards;
                text-align: center;
                white-space: nowrap;
            }
            .cellvoyager-success-toast-overlay {
                position: absolute;
                top: 0.55rem;
                right: 0.6rem;
                z-index: 2;
                box-shadow: 0 3px 10px rgba(22, 101, 52, 0.16);
                pointer-events: none;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
    # Auto-open a newly inserted cell for editing
    open_cell = st.session_state.pop("open_edit_cell", None)
    if open_cell is not None and pause_id:
        st.session_state[f"pause_cell_editing_{pause_id}_{open_cell}"] = True
        # Clear any stale cached content so the new blank cell starts empty
        _ek = f"pause_cell_{pause_id}_{open_cell}"
        st.session_state.pop(_ek, None)
        st.session_state.pop(f"ace_{_ek}", None)

    cell_sources = []
    run_clicked = None
    insert_after_idx = -1
    insert_type = None
    step_display_counter = 0
    run_monitor = st.session_state.get("_cell_run_monitor")
    for i, cell in enumerate(nb.cells):
        if _is_step_summary(cell):
            step_display_counter += 1
            _step_separator(str(step_display_counter))
        with st.container():
            cell_kind = "Code" if cell.cell_type == "code" else "Markdown"
            st.markdown(f"**Cell {i}** · {cell_kind}")
            if edit_mode:
                edited = _render_editable_cell(cell, i, pause_id, nb_path, use_button=sidebar_actions)
                cell_sources.append(edited)
            else:
                _render_cell_display(cell)
                if editable:
                    cell_sources.append(_cell_source_str(cell))
            if edit_mode:
                if cell.cell_type == "code" and not standalone_edit:
                    is_running_this_cell = (
                        isinstance(run_monitor, dict)
                        and run_monitor.get("nb_path") == nb_path
                        and run_monitor.get("pause_id") == pause_id
                        and int(run_monitor.get("cell_index", -1)) == i
                    )
                    if is_running_this_cell:
                        elapsed = time.monotonic() - float(run_monitor.get("started_monotonic", time.monotonic()))
                        st.info(f"Running code cell... waiting for completion ({elapsed:.1f}s)")
                    else:
                        _btn = st.button if sidebar_actions else st.form_submit_button
                        if _btn(
                            "▶ Run Code Cell",
                            key=f"run_cell_{pause_id}_{i}",
                            type="primary",
                            width="stretch",
                        ):
                            run_clicked = i
                ins_cols = st.columns([1, 1, 2])
                with ins_cols[0]:
                    _btn = st.button if sidebar_actions else st.form_submit_button
                    if _btn("Insert Code Below", key=f"ins_code_{pause_id}_{i}"):
                        insert_after_idx = i
                        insert_type = "code"
                with ins_cols[1]:
                    _btn = st.button if sidebar_actions else st.form_submit_button
                    if _btn("Insert Markdown Below", key=f"ins_md_{pause_id}_{i}"):
                        insert_after_idx = i
                        insert_type = "markdown"
            st.divider()
    add_code, add_md, continue_clicked, finish_clicked, edit_clicked, save_clicked = False, False, False, False, False, False
    if editable:
        if sidebar_actions:
            # Feedback textarea lives in the sidebar; just read the value here
            feedback = st.session_state.get(f"pause_feedback_{pause_id}", "")
            # Consume sidebar button flags set this rerun
            if st.session_state.pop("_sb_continue", False):
                continue_clicked = True
            if st.session_state.pop("_sb_edit", False):
                edit_clicked = True
            if st.session_state.pop("_sb_finish", False):
                finish_clicked = True
        elif not standalone_edit:
            st.markdown(
                '<div class="feedback-box-header">💬 <strong>Feedback for the agent (gets passed to the Agent once you hit Continue Analysis)</strong></div>',
                unsafe_allow_html=True,
            )
            feedback = st.text_area(
                "Feedback",
                placeholder="e.g., focus more on cluster 3, or skip the next visualization...",
                height=100, key=f"pause_feedback_{pause_id}", label_visibility="collapsed",
            )
            st.caption("Click Edit to modify cells, or Continue/Finish to proceed.")
        else:
            feedback = ""
            st.caption("Click Edit to modify cells, then Save changes to persist edits.")
        if not sidebar_actions:
            try:
                from streamlit_extras.stylable_container import stylable_container
                button_container = stylable_container(
                    key=f"pause_buttons_{pause_id}",
                    css_styles="""
                        button {
                            padding: 0.5rem 2rem !important;
                            font-size: 1.75rem !important;
                            font-weight: 700 !important;
                            min-height: 48px !important;
                        }
                    """,
                )
            except ImportError:
                button_container = st.container()
            with button_container:
                if standalone_edit:
                    btn_cols = st.columns(2)
                    with btn_cols[0]:
                        edit_clicked = st.form_submit_button("✏️ Edit", help="Show edit options for cells", type="primary", width="stretch")
                    with btn_cols[1]:
                        save_clicked = st.form_submit_button("💾 Save changes", help="Save notebook edits", type="primary", width="stretch")
                else:
                    _, c1, c2, c3, _ = st.columns([0.4, 1, 1, 1, 0.4])
                    with c1:
                        edit_clicked = st.form_submit_button("Edit Analysis", help="Show edit options for cells", type="primary", width="stretch")
                    with c2:
                        continue_clicked = st.form_submit_button("Continue Analysis", help="Send feedback and continue analysis", type="primary", width="stretch")
                    with c3:
                        finish_clicked = st.form_submit_button("Finish Analysis", help="Tell agent to finish the analysis", type="primary", width="stretch")
        if edit_mode:
            st.caption(
                "Use per-cell controls to run and insert below, or add new cells at the end."
                + ("" if standalone_edit else " Run code cells, then Continue or Finish.")
            )
            add_row = st.columns(2)
            _btn = st.button if sidebar_actions else st.form_submit_button
            with add_row[0]:
                add_code = _btn("+ Code cell (at end)")
            with add_row[1]:
                add_md = _btn("+ Markdown (at end)")
        if add_code and insert_type is None:
            insert_after_idx = len(nb.cells) - 1
            insert_type = "code"
        elif add_md and insert_type is None:
            insert_after_idx = len(nb.cells) - 1
            insert_type = "markdown"
        return cell_sources, feedback, nb, run_clicked, insert_after_idx, insert_type, continue_clicked, finish_clicked, edit_clicked, save_clicked
    return None


def _render_notebook(nb_path):
    try:
        from nbconvert import HTMLExporter
        import nbformat
        with open(nb_path, encoding="utf-8") as f:
            content = f.read()
        if not content.strip():
            st.warning(f"Notebook is empty or still being written: `{Path(nb_path).name}`")
            return
        nb = nbformat.reads(content, as_version=4)
        exporter = HTMLExporter()
        exporter.template_name = "classic"
        body, _ = exporter.from_notebook_node(nb)
        scroll_script = """
        <script>
        (function() {
            var scrollToBottom = function() { window.scrollTo(0, document.body.scrollHeight); };
            if (document.readyState === 'loading') {
                document.addEventListener('DOMContentLoaded', scrollToBottom);
            } else { scrollToBottom(); }
        })();
        </script>
        """
        body = body + scroll_script
        st.components.v1.html(body, height=600, scrolling=True)
    except Exception as e:
        st.error(f"Could not render notebook: {e}")
        st.code(f"Open in Jupyter: {nb_path}")


def _pause_request_path():
    out = st.session_state.get("run_output_dir")
    if not out:
        return None
    return Path(out) / _PAUSE_REQUEST_FILE


def _pause_response_path():
    out = st.session_state.get("run_output_dir")
    if not out:
        return None
    return Path(out) / _PAUSE_RESPONSE_FILE


def _pause_execute_path():
    out = st.session_state.get("run_output_dir")
    if not out:
        return None
    return Path(out) / _EXECUTE_REQUEST_FILE


def _chat_request_path():
    out = st.session_state.get("run_output_dir")
    if not out:
        return None
    return Path(out) / _CHAT_REQUEST_FILE


def _chat_response_path():
    out = st.session_state.get("run_output_dir")
    if not out:
        return None
    return Path(out) / _CHAT_RESPONSE_FILE


def _chat_via_api(messages: list, output_dir: str, analysis_idx: int | None = None) -> str | None:
    try:
        import nbformat as nbf
    except ImportError:
        return None
    out = Path(output_dir)
    if analysis_idx is not None:
        notebooks = sorted(out.glob(f"*_analysis_{analysis_idx}.ipynb"))
    else:
        notebooks = list(out.glob("*.ipynb"))
    context_parts = []
    for nb_path in notebooks[:3]:
        try:
            with open(nb_path, encoding="utf-8") as f:
                nb = nbf.read(f, as_version=4)
            for cell in nb.cells[:50]:
                src = "\n".join(cell.source) if isinstance(cell.source, list) else (cell.source or "")
                if src.strip():
                    context_parts.append(f"[{cell.cell_type}]\n{src[:2000]}")
        except Exception:
            pass
    context = "\n\n---\n\n".join(context_parts)[:15000]
    system = (
        "You are the agent that performed this single-cell transcriptomics analysis. "
        "The user is viewing the Jupyter notebook with your results. Answer their questions concisely based on the notebook content below.\n\n"
        "Notebook content:\n" + (context or "(no notebook loaded)")
    )
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=anthropic_key)
            api_messages = [{"role": m["role"], "content": m["content"]} for m in messages]
            resp = client.messages.create(
                model="claude-sonnet-4-5", max_tokens=1024, system=system, messages=api_messages,
            )
            if resp.content and len(resp.content) > 0:
                return resp.content[0].text.strip()
            return None
        except Exception:
            pass
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            api_messages = [{"role": "system", "content": system}]
            for m in messages:
                api_messages.append({"role": m["role"], "content": m["content"]})
            resp = client.chat.completions.create(
                model="gpt-4o-mini", messages=api_messages, max_tokens=1024,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception:
            pass
    return None


def _render_chat_box(
    output_dir: str | None,
    analysis_idx: int | None = None,
    floating: bool = False,
    num_total: int = 1,
    sel_key: str | None = None,
):
    """Lightweight chat panel. Renders a compact analysis selector, message history, and input bar.
    When floating=True, pins the parent column to the viewport."""
    if not output_dir:
        return

    # Compact "Analysis: [selector]" row
    if num_total > 1 and sel_key:
        if sel_key not in st.session_state:
            st.session_state[sel_key] = 1
        # Only show analyses that have an actual notebook on disk
        _existing = _collect_notebooks_by_analysis(output_dir, num_total)
        _created = [i for i, nb in _existing if nb is not None]
        if not _created:
            _created = [1]
        # Clamp the stored selection to the created range
        if st.session_state[sel_key] not in _created:
            st.session_state[sel_key] = _created[-1]
        _lbl, _sel = st.columns([0.28, 0.72])
        with _lbl:
            st.markdown(
                '<p style="margin:0;font-size:0.8rem;'
                'color:#64748b;font-weight:600">Analysis:</p>',
                unsafe_allow_html=True,
            )
        with _sel:
            analysis_idx = st.selectbox(
                "Analysis",
                options=_created,
                format_func=lambda x: f"Analysis {x}",
                key=sel_key,
                label_visibility="collapsed",
            )

    suffix = f"_analysis_{analysis_idx}" if analysis_idx is not None else ""
    key = f"agent_chat_{output_dir}{suffix}"
    pending_key = f"agent_chat_pending_{output_dir}{suffix}"
    pending_reply_key = f"agent_chat_pending_reply_{output_dir}{suffix}"
    if key not in st.session_state:
        st.session_state[key] = _load_chat_history(output_dir, analysis_idx)
    messages = st.session_state[key]
    pending_prompt = st.session_state.pop(pending_key, None)
    if pending_prompt:
        messages.append({"role": "user", "content": pending_prompt})
        _save_chat_history(output_dir, analysis_idx, messages)
        st.session_state[pending_reply_key] = True

    # Header with expand/collapse toggle
    expanded = st.session_state.get("cv_chat_expanded", False)
    _h_col, _btn_col = st.columns([2.5, 1])
    with _h_col:
        st.markdown(
            '<p style="margin:0 0 0.4rem 0;font-size:0.95rem;font-weight:700;color:#1e293b">💬 Chat with Agent</p>',
            unsafe_allow_html=True,
        )
    with _btn_col:
        if st.button(
            "✕ Close" if expanded else "⤢ Expand",
            key="cv_chat_expand_btn",
            help="Close overlay" if expanded else "Expand to full overlay",
        ):
            st.session_state["cv_chat_expanded"] = not expanded
            st.rerun()

    # CSS — base styles + optional overlay when expanded
    _overlay_css = (
        # Backdrop — covers everything including sidebar; pointer-events blocks interaction with background
        "body::before { content:''; position:fixed; inset:0; background:rgba(0,0,0,0.52); z-index:9998; pointer-events:all; }"
        # Float chat column as centered modal above the backdrop
        "[data-testid='stColumn']:has([data-testid='stForm']) {"
        "  position:fixed !important; top:5vh !important; left:50% !important;"
        "  transform:translateX(-50%) !important; width:70vw !important;"
        "  max-height:90vh !important; overflow-y:auto !important;"
        "  z-index:9999 !important; background:#fff !important;"
        "  border-radius:16px !important; box-shadow:0 32px 80px rgba(0,0,0,0.35) !important;"
        "  padding:28px !important; pointer-events:all !important; }"
        # Pin close button to top-right corner of the modal
        # Target the header row specifically: has a <p> but NOT an <input> (unlike the form's input row)
        "[data-testid='stColumn']:has([data-testid='stForm']) [data-testid='stHorizontalBlock']:has(p):not(:has(input)) [data-testid='stColumn']:last-child {"
        "  position:absolute !important; top:16px !important; right:20px !important; width:auto !important; }"
        "[data-testid='stColumn']:has([data-testid='stForm']) [data-testid='stHorizontalBlock']:has(p):not(:has(input)) [data-testid='stColumn']:last-child button {"
        "  width:auto !important; padding:3px 12px !important; font-size:0.8rem !important; }"
    ) if expanded else ""

    st.markdown(
        '<div class="cv-chat-top-anchor"></div>'
        '<style>'
        '[data-testid="InputInstructions"] { display:none !important; }'
        # Fix gap above expand button: vertically align header row
        '[data-testid="stColumn"]:has([data-testid="stForm"]) [data-testid="stHorizontalBlock"]:first-of-type { align-items:center !important; }'
        '[data-testid="stColumn"]:has([data-testid="stForm"]) [data-testid="stHorizontalBlock"]:first-of-type [data-testid="stColumn"]:last-child > div { margin-top:0 !important; padding-top:0 !important; }'
        # Form styles
        '[data-testid="stForm"] { border:none !important; padding:0 !important; margin:0 !important; }'
        '[data-testid="stForm"] > div { padding:0 !important; margin:0 !important; }'
        '[data-testid="stForm"] [data-testid="stHorizontalBlock"] { gap:6px !important; padding:0 !important; }'
        '[data-testid="stForm"] [data-testid="stColumn"] { padding:0 !important; min-width:0 !important; }'
        '[data-testid="stForm"] button { width:100% !important; padding:4px 6px !important; min-height:0 !important; font-size:0.85rem !important; }'
        # Expand/Close button — smaller font so it stays on one line at laptop screen widths
        '[data-testid="stColumn"]:has([data-testid="stForm"]) [data-testid="stHorizontalBlock"]:first-of-type [data-testid="stColumn"]:last-child button { font-size:0.72rem !important; padding:3px 8px !important; white-space:nowrap !important; overflow:visible !important; }'
        # Analysis label + selector row — align items vertically
        '[data-testid="stColumn"]:has([data-testid="stForm"]) [data-testid="stHorizontalBlock"]:has(select) { align-items:center !important; }'
        '[data-testid="stForm"] input { background:#fff !important; border:1px solid #cbd5e1 !important; border-radius:6px !important; }'
        + _overlay_css +
        '</style>',
        unsafe_allow_html=True,
    )
    st.components.v1.html(
        """<script>
        (function() {
          const p = window.parent;
          const setInputAttrs = () => {
            p.document.querySelectorAll(
              '[data-testid="stChatInputTextArea"] textarea, .stTextInput input'
            ).forEach(el => {
              el.setAttribute('autocomplete', 'off');
              el.setAttribute('autocorrect', 'off');
              el.setAttribute('autocapitalize', 'off');
              el.setAttribute('spellcheck', 'false');
            });
          };
          const killGaps = () => {
            p.document.querySelectorAll('[data-testid="stForm"]').forEach(form => {
              let el = form;
              for (let i = 0; i < 12; i++) {
                el.style.setProperty('margin-bottom', '0', 'important');
                el.style.setProperty('padding-bottom', '0', 'important');
                el.style.setProperty('margin-right', '0', 'important');
                el.style.setProperty('padding-right', '0', 'important');
                if (!el.parentElement) break;
                if (el.matches('[data-testid="stColumn"]')) { el = el.parentElement; break; }
                el = el.parentElement;
              }
            });
            p.document.querySelectorAll('[data-testid="stColumn"]').forEach(col => {
              if (col.querySelector('[data-testid="stForm"]')) {
                col.style.setProperty('padding-bottom', '0', 'important');
                col.style.setProperty('padding-right', '0', 'important');
              }
            });
          };
          setInputAttrs(); killGaps();
          [300, 800, 1500].forEach(t => setTimeout(() => { setInputAttrs(); killGaps(); }, t));
          const obs = new MutationObserver(killGaps);
          obs.observe(p.document.body, { childList: true, subtree: true });
          setTimeout(() => obs.disconnect(), 6000);
        })();
        </script>""",
        height=0,
    )

    # Scrollable message history — user bubbles right, assistant left
    _history_height = 520 if expanded else 280
    with st.container(height=_history_height, border=False):
        for msg in messages:
            if msg["role"] == "user":
                safe = html.escape(msg["content"]).replace("\n", "<br>")
                st.markdown(
                    f'<div style="display:flex;justify-content:flex-end;margin:4px 8px 4px 0">'
                    f'<div style="background:#2563eb;color:#fff;padding:8px 12px;'
                    f'border-radius:16px 16px 4px 16px;max-width:85%;font-size:1.25rem;line-height:1.45">'
                    f'{safe}</div></div>',
                    unsafe_allow_html=True,
                )
            else:
                rendered = _markdown.markdown(msg["content"], extensions=["nl2br", "tables", "fenced_code"])
                st.markdown(
                    f'<div style="display:flex;justify-content:flex-start;margin:4px 0">'
                    f'<div style="background:#f1f5f9;color:#1e293b;padding:8px 12px;'
                    f'border-radius:16px 16px 16px 4px;max-width:85%;font-size:1.25rem;line-height:1.45">'
                    f'{rendered}</div></div>',
                    unsafe_allow_html=True,
                )
        if st.session_state.get(pending_reply_key, False):
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    reply = None
                    req_path = _chat_request_path()
                    resp_path = _chat_response_path()
                    if req_path and resp_path:
                        try:
                            resp_path.unlink(missing_ok=True)
                            req_path.write_text(
                                json.dumps({"message": messages[-1]["content"], "conversation": messages[:-1]}),
                                encoding="utf-8",
                            )
                            for _ in range(120):
                                time.sleep(0.05)
                                if resp_path.exists():
                                    data = json.loads(resp_path.read_text(encoding="utf-8"))
                                    reply = data.get("response", "")
                                    resp_path.unlink(missing_ok=True)
                                    break
                        except Exception:
                            pass
                    if reply is None:
                        reply = _chat_via_api(messages, output_dir, analysis_idx=analysis_idx)
                    messages.append({
                        "role": "assistant",
                        "content": reply or "*(Could not get a response. Your message was saved.)*",
                    })
                    _save_chat_history(output_dir, analysis_idx, messages)
                    # Clear AFTER reply is received so any rerun mid-spinner retries correctly
                    st.session_state.pop(pending_reply_key, None)
            st.rerun()

    # Input bar
    with st.form(key=f"chat_form_{key}", clear_on_submit=True):
        cols = st.columns([10, 1])
        with cols[0]:
            user_input = st.text_input(
                "Message", placeholder="Ask about the results…", label_visibility="collapsed"
            )
        with cols[1]:
            submitted = st.form_submit_button("→", use_container_width=True)
    if submitted and user_input.strip():
        messages.append({"role": "user", "content": user_input.strip()})
        _save_chat_history(output_dir, analysis_idx, messages)
        st.session_state[pending_reply_key] = True
        st.rerun()

    # Bottom sentinel
    st.markdown('<div class="cv-chat-anchor"></div>', unsafe_allow_html=True)
    if floating:
        st.components.v1.html(
            """
            <script>
            (function () {
              try {
                const p = window.parent;
                const BOX_ID = 'cv-floating-chat';
                const applyFloat = () => {
                  const anchor = p.document.querySelector('.cv-chat-top-anchor');
                  if (!anchor) return false;
                  // Walk up to the stColumn ancestor
                  let col = anchor.parentElement;
                  while (col && col.getAttribute && col.getAttribute('data-testid') !== 'stColumn') {
                    col = col.parentElement;
                  }
                  if (!col) return false;
                  col.id = BOX_ID;
                  Object.assign(col.style, {
                    position:    'sticky',
                    top:         '3.5rem',
                    alignSelf:   'flex-start',
                    maxHeight:   'calc(100vh - 5rem)',
                    overflowY:   'auto',
                    zIndex:      '998',
                    borderRadius:'10px',
                    boxShadow:   '0 4px 20px rgba(30,64,175,0.18)',
                    padding:     '0.5rem',
                  });
                  return true;
                };
                // Retry until the anchor is in DOM
                let tries = 0;
                const retry = () => { if (!applyFloat() && tries++ < 20) setTimeout(retry, 150); };
                retry();
                // Re-apply on resize
                if (!p._cvChatResizeWired) {
                  p._cvChatResizeWired = true;
                  p.addEventListener('resize', applyFloat, { passive: true });
                }
                // Re-apply whenever DOM changes (Streamlit reruns may recreate the column)
                if (!p._cvChatObserver) {
                  p._cvChatObserver = new p.MutationObserver(() => applyFloat());
                  p._cvChatObserver.observe(p.document.body, { childList: true, subtree: false });
                }
              } catch (e) {}
            })();
            </script>
            """,
            height=0,
        )


def _render_edit_mode_banner_sticky() -> None:
    """Render the edit mode banner fixed at center-right of the viewport, following the user as they scroll."""
    has_ace = True
    try:
        import streamlit_ace  # noqa: F401
    except Exception:
        has_ace = False
    tip = "" if has_ace else '<p style="font-size:1.1rem;color:#6b7280;margin-top:0.4rem;">Tip: install <code>streamlit-ace</code> for syntax highlighting.</p>'
    st.markdown(
        f"""
        <div style="position:fixed;top:50%;right:1.5rem;transform:translateY(-50%);
            width:22%;max-width:280px;z-index:1000;
            background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;
            padding:1rem;font-size:1.25rem;color:#1e40af;
            box-shadow:0 2px 10px rgba(30,64,175,0.15);">
            <strong>Edit mode is ON.</strong> Click <strong>Edit cell</strong> on any cell to modify it,
            then <strong>Save</strong> to apply changes. To <strong>run</strong> a code cell, click
            <strong>Run Code Cell</strong>.
            {tip}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _should_show_chat():
    if st.session_state.get("run_started") and _has_live_run():
        proc = st.session_state.get("run_proc")
        in_pause = (
            st.session_state.get("run_interactive_mode")
            and _pause_request_path()
            and _pause_request_path().exists()
        )
        if proc is not None:
            return in_pause or proc.poll() is not None
        return in_pause or True
    return bool(st.session_state.get("run_output_dir"))

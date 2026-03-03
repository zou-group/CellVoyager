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
    st.session_state.run_output = []
    st.session_state.run_output_dir = output_dir
    st.session_state.run_started = True
    st.session_state.run_interactive_mode = True
    st.session_state.run_thread_started = True
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
    ("home_analysis_name", "covid19"),
    ("home_num_analyses", 1),
    ("home_max_iterations", 8),
    ("home_execution_mode", "claude"),
    ("home_interactive_mode", False),
    ("home_intervene_every", 1),
    ("home_use_deepresearch", False),
    ("home_model_name", "o3-mini"),
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
_RUN_PID_FILE = ".run_pid"
_RUN_LOG_FILE = ".run_log"
_RUN_INTERACTIVE_FILE = ".run_interactive"
_RUN_CONFIG_FILE = ".run_config.json"
_LAST_DISPLAYED_DIR = ".last_displayed"


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
    """Request the agent to pause (Stop button). Creates STOP file (agent checks before each tool call)
    and pause request file (for GUI). Agent will call pause_for_user_review at next check. Does NOT kill.
    Returns True if pause files were created, False otherwise (caller may fall back to kill)."""
    out_dir_str = st.session_state.get("run_output_dir")
    if not out_dir_str:
        return False
    out_dir = Path(out_dir_str).resolve()
    if not out_dir.exists():
        return False
    notebooks = sorted(out_dir.glob("*.ipynb"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not notebooks:
        # Try config to get expected notebook path (e.g. before agent creates it)
        try:
            cfg_path = out_dir / _RUN_CONFIG_FILE
            if cfg_path.exists():
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                name = cfg.get("analysis_name", "analysis")
                nb_path = out_dir / f"{name}_analysis_1.ipynb"
                if nb_path.exists():
                    notebooks = [nb_path]
        except Exception:
            pass
    if not notebooks:
        return False
    nb_path = notebooks[0]
    (out_dir / _STOP_REQUEST_FILE).write_text("1", encoding="utf-8")  # Agent sees this, returns pause_requested
    (out_dir / _PAUSE_REQUEST_FILE).write_text(str(nb_path.resolve()), encoding="utf-8")
    (out_dir / _PAUSE_RESPONSE_FILE).unlink(missing_ok=True)  # Clear stale response
    summary = _extract_agent_summary_from_notebook(nb_path)
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


# Jupyter-like output styling: monospace font, pre-wrap, matches notebook output area
_JUPYTER_OUTPUT_STYLE = (
    "font-family: ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Monaco, Consolas, "
    "'Liberation Mono', 'Courier New', monospace; font-size: 13px; line-height: 1.4; "
    "margin: 0; white-space: pre-wrap; word-break: break-word;"
)


def _render_cell_outputs(cell):
    if cell.cell_type != "code":
        return
    outputs = getattr(cell, "outputs", []) or []
    if not outputs:
        return
    parts = []
    for out in outputs:
        ot = out.get("output_type", "")
        if ot == "stream":
            text = out.get("text", "")
            if isinstance(text, list):
                text = "".join(str(t) for t in text)
            else:
                text = str(text)
            if text.strip():
                name = out.get("name", "stdout")
                color = "inherit" if name == "stdout" else "#c53030"
                parts.append(
                    f'<pre style="{_JUPYTER_OUTPUT_STYLE} color:{color};">{html.escape(text)}</pre>'
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
                plain = data["text/plain"]
                if isinstance(plain, list):
                    plain = "".join(str(p) for p in plain)
                else:
                    plain = str(plain)
                parts.append(f'<pre style="{_JUPYTER_OUTPUT_STYLE}">{prefix}{html.escape(plain)}</pre>')
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
                    parts.append('<span style="color:#6b7280;font-size:0.75rem;">[Image output]</span>')
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
                plain = data["text/plain"]
                if isinstance(plain, list):
                    plain = "".join(str(p) for p in plain)
                else:
                    plain = str(plain)
                parts.append(f'<pre style="{_JUPYTER_OUTPUT_STYLE}">{html.escape(plain)}</pre>')
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
                    parts.append('<span style="color:#6b7280;font-size:0.75rem;">[Image output]</span>')
        elif ot == "error":
            tb = out.get("traceback", [])
            if isinstance(tb, list):
                tb = "\n".join(tb)
            parts.append(
                f'<pre style="{_JUPYTER_OUTPUT_STYLE} color:#c53030; background:rgba(254,226,226,0.4); padding:0.5rem; border-radius:4px;">{html.escape(tb)}</pre>'
            )
    if not parts:
        return
    inner = "\n".join(parts)
    st.markdown(
        f"""
        <div style="margin-top:0.35rem; border:1px solid rgba(151,166,175,0.35); border-radius:6px; background:rgba(250,251,252,0.9); padding:0.6rem 0.75rem; font-family: ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Monaco, Consolas, monospace; font-size: 13px;">
            {inner}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_cell_display(cell):
    src = _cell_source_str(cell)
    if cell.cell_type == "markdown":
        st.markdown(src)
    else:
        st.code(src, language="python")
        _render_cell_outputs(cell)


def _render_editable_cell(cell, cell_idx, pause_id, expand_edit=False):
    src = _cell_source_str(cell)
    height = min(300, max(80, 60 + len(src.splitlines()) * 18))
    if cell.cell_type == "markdown":
        st.markdown(src)
        with st.expander("Edit", expanded=expand_edit):
            editor = st.text_area(
                "Source", value=src, height=height,
                key=f"pause_cell_{pause_id}_{cell_idx}", label_visibility="collapsed",
            )
    else:
        st.code(src, language="python")
        _render_cell_outputs(cell)
        with st.expander("Edit", expanded=expand_edit):
            editor = st.text_area(
                "Source", value=src, height=height,
                key=f"pause_cell_{pause_id}_{cell_idx}", label_visibility="collapsed",
            )
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


def _render_notebook_jupyter_style(nb_path, editable=False, pause_id=None, standalone_edit=False, save_snapshot=False, output_dir=None):
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
    cell_sources = []
    run_clicked = None
    insert_after_idx = -1
    insert_type = None
    step_display_counter = 0
    for i, cell in enumerate(nb.cells):
        if _is_step_summary(cell):
            step_display_counter += 1
            _step_separator(str(step_display_counter))
        with st.container():
            st.markdown(f"**Cell {i}** ({cell.cell_type})")
            if edit_mode:
                expand_edit = i == st.session_state.get("open_edit_cell")
                if expand_edit:
                    st.session_state.pop("open_edit_cell", None)
                edited = _render_editable_cell(cell, i, pause_id, expand_edit=expand_edit)
                cell_sources.append(edited)
            else:
                _render_cell_display(cell)
                if editable:
                    cell_sources.append(_cell_source_str(cell))
            if edit_mode:
                if cell.cell_type == "code" and not standalone_edit:
                    if st.form_submit_button("▶ Run", key=f"run_cell_{pause_id}_{i}", type="primary", use_container_width=True):
                        run_clicked = i
                ins_cols = st.columns([1, 1, 2])
                with ins_cols[0]:
                    if st.form_submit_button("Insert code", key=f"ins_code_{pause_id}_{i}"):
                        insert_after_idx = i
                        insert_type = "code"
                with ins_cols[1]:
                    if st.form_submit_button("Insert markdown", key=f"ins_md_{pause_id}_{i}"):
                        insert_after_idx = i
                        insert_type = "markdown"
            st.divider()
    add_code, add_md, continue_clicked, finish_clicked, edit_clicked, save_clicked = False, False, False, False, False, False
    if editable:
        if not standalone_edit:
            st.markdown(
                '<div class="feedback-box-header">💬 <strong>Feedback for the agent</strong></div>',
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
                    edit_clicked = st.form_submit_button("✏️ Edit", help="Show edit options for cells", type="primary")
                with btn_cols[1]:
                    save_clicked = st.form_submit_button("💾 Save changes", help="Save notebook edits", type="primary")
            else:
                btn_cols = st.columns(3)
                with btn_cols[0]:
                    edit_clicked = st.form_submit_button("✏️ Edit", help="Show edit options for cells", type="primary")
                with btn_cols[1]:
                    continue_clicked = st.form_submit_button("▶ Continue", help="Send feedback and continue analysis", type="primary")
                with btn_cols[2]:
                    finish_clicked = st.form_submit_button("🏠 Finish", help="Tell agent to finish the analysis", type="primary")
        if edit_mode:
            st.caption(
                "Insert cells after a specific cell above, or add at end."
                + ("" if standalone_edit else " Run code cells, then Continue or Finish.")
            )
            add_row = st.columns(2)
            with add_row[0]:
                add_code = st.form_submit_button("+ Code cell (at end)")
            with add_row[1]:
                add_md = st.form_submit_button("+ Markdown (at end)")
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


def _render_chat_box(output_dir: str | None, analysis_idx: int | None = None):
    """Render chat box. When analysis_idx is set, uses separate history and context for that analysis."""
    if not output_dir:
        return
    suffix = f"_analysis_{analysis_idx}" if analysis_idx is not None else ""
    key = f"agent_chat_{output_dir}{suffix}"
    pending_key = f"agent_chat_pending_{output_dir}{suffix}"
    pending_reply_key = f"agent_chat_pending_reply_{output_dir}{suffix}"
    if key not in st.session_state:
        st.session_state[key] = []
    messages = st.session_state[key]
    pending_prompt = st.session_state.pop(pending_key, None)
    if pending_prompt:
        messages.append({"role": "user", "content": pending_prompt})
        st.session_state[pending_reply_key] = True
    with st.container():
        st.markdown(
            '<div class="chat-box-outer">'
            '<div class="chat-box-header">💬 <strong>Chat with agent</strong></div>'
            '<p class="chat-box-caption">Ask about the analysis results (separate from feedback)</p></div>',
            unsafe_allow_html=True,
        )
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    if st.session_state.pop(pending_reply_key, False):
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
        st.rerun()
    prompt = st.chat_input("Ask about the results...")
    if prompt:
        st.session_state[pending_key] = prompt
        st.rerun()


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

"""
Analysis page — running/pause/notebook viewer. Reached via st.switch_page when Run is clicked.
No home content (Settings, Paper) is ever rendered here.
"""
import html
import json
import time
from pathlib import Path

import streamlit as st

import gui_common as g

if not st.session_state.get("run_output_dir"):
    # Restore from last run file if session was reset (e.g. after completion transition)
    if g._LAST_RUN_FILE.exists():
        try:
            last_dir = g._LAST_RUN_FILE.read_text(encoding="utf-8").strip()
            if last_dir and Path(last_dir).exists():
                st.session_state.run_output_dir = last_dir
                st.session_state.run_started = False
        except Exception:
            pass
if not st.session_state.get("run_output_dir"):
    st.switch_page("gui.py")

LOGO_PATH = g.ROOT / "images" / "symbol.jpeg"
st.set_page_config(
    page_title="CellVoyager — Analysis",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else "📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    h1 { font-weight: 600 !important; letter-spacing: -0.02em !important; }
    div[data-testid="stVerticalBlock"] > div { gap: 0.5rem !important; }
    div[data-testid="stExpander"] { border-radius: 10px !important; box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important; }
    div[data-testid="stAlert"] { border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)

if LOGO_PATH.exists():
    col_logo, col_title = st.columns([0.15, 1])
    with col_logo:
        st.image(str(LOGO_PATH), width="stretch")
    with col_title:
        st.title("CellVoyager")
else:
    st.title("CellVoyager")

st.divider()

# Sidebar: Stop / Finish only
with st.sidebar:
    st.markdown("### ▶ Analysis")
    if g._has_live_run():
        if st.button("⏹ Stop Analysis", type="primary", use_container_width=True, help="Pause the analysis so you can edit. Click Continue to resume — no reload."):
            if not g._request_pause():
                g._kill_analysis(show_interactive_edit=True)
            st.rerun()
    else:
        if st.button("🏠 Finish", type="primary", use_container_width=True, help="Return to home and start a new analysis"):
            st.session_state.run_output_dir = None
            st.switch_page("gui.py")

# Main: running / pause / notebook viewer
# If run was started but process is dead (e.g. finished, or server restarted), transition to completed view
if st.session_state.run_started and not g._has_live_run():
    st.session_state.run_proc = None
    st.session_state.run_pid = None
    st.session_state.run_started = False
    st.session_state.run_thread_started = False
    if st.session_state.run_output_dir:
        out_dir = Path(st.session_state.run_output_dir)
        if out_dir.exists():
            (out_dir / g._RUN_PID_FILE).unlink(missing_ok=True)
            # Restore intervene_every if we had overridden it for run-to-completion
            cfg_path = out_dir / g._RUN_CONFIG_FILE
            if cfg_path.exists():
                try:
                    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                    if "intervene_every_restore" in cfg:
                        cfg["intervene_every"] = cfg.pop("intervene_every_restore", 1)
                        cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
                except Exception:
                    pass
    st.rerun()

if st.session_state.get("run_started") and g._has_live_run():
    proc = st.session_state.get("run_proc")
    output_text = g._get_run_log()
    request_path = g._pause_request_path()
    response_path = g._pause_response_path()
    # Show pause UI when request file exists (from agent's natural pause OR from Stop button)
    in_pause_ui = bool(request_path and response_path and request_path.exists())

    if in_pause_ui:
        show_chat = g._should_show_chat()
        if show_chat:
            main_col, chat_col = st.columns([0.72, 0.28])
        else:
            main_col = st.container()
            chat_col = None

        result = None
        nb_path = ""
        try:
            nb_path = request_path.read_text(encoding="utf-8").strip()
        except Exception:
            pass
        num_total = st.session_state.get("run_num_analyses", 1)
        paused_analysis_idx = None
        if nb_path:
            import re
            m = re.search(r"_analysis_(\d+)\.ipynb", Path(nb_path).name, re.I)
            if m:
                paused_analysis_idx = int(m.group(1))

        with main_col:
            st.markdown("---")
            st.markdown("### ⏸ Agent waiting for feedback")
            st.caption("Edit cells, insert new ones, run code. Click **Continue** when ready for the agent to proceed.")
            if num_total > 1 and paused_analysis_idx is not None:
                analyses = g._collect_notebooks_by_analysis(st.session_state.run_output_dir, num_total)
                tab_labels = [f"Analysis {i}" + (" ⏸" if i == paused_analysis_idx else "") for i, _ in analyses]
                tabs = st.tabs(tab_labels)
                for i, (aidx, _) in enumerate(analyses):
                    with tabs[i]:
                        if aidx == paused_analysis_idx and nb_path and Path(nb_path).exists():
                            import nbformat as nbf
                            from nbformat.v4 import new_code_cell, new_markdown_cell
                            pause_id = f"{request_path.stat().st_mtime}" if request_path.exists() else "0"
                            with st.form("interactive_feedback_form"):
                                st.caption("Edit cells, add new ones, run code to see outputs. Click Continue when ready for the agent.")
                                result = g._render_notebook_jupyter_style(nb_path, editable=True, pause_id=pause_id)
                            summary_path = Path(st.session_state.run_output_dir) / g._AGENT_SUMMARY_FILE if st.session_state.run_output_dir else None
                            if summary_path and summary_path.exists():
                                try:
                                    summary_text = summary_path.read_text(encoding="utf-8").strip()
                                    if summary_text:
                                        st.markdown("---")
                                        st.markdown("#### 🤖 What the agent has done so far")
                                        escaped = html.escape(summary_text).replace("\n", "<br>")
                                        st.markdown(
                                            f'<div class="agent-summary-box"><div class="agent-summary-content">{escaped}</div></div>',
                                            unsafe_allow_html=True,
                                        )
                                except Exception:
                                    pass
                        else:
                            other_nb_path = analyses[i][1] if i < len(analyses) else None
                            if other_nb_path:
                                g._render_notebook_jupyter_style(other_nb_path, editable=False)
                            else:
                                st.info("Pending")
            elif nb_path and Path(nb_path).exists():
                import nbformat as nbf
                from nbformat.v4 import new_code_cell, new_markdown_cell
                pause_id = f"{request_path.stat().st_mtime}" if request_path.exists() else "0"
                with st.form("interactive_feedback_form"):
                    st.caption("Edit cells, add new ones, run code to see outputs. Click Continue when ready for the agent.")
                    result = g._render_notebook_jupyter_style(nb_path, editable=True, pause_id=pause_id)
                summary_path = Path(st.session_state.run_output_dir) / g._AGENT_SUMMARY_FILE if st.session_state.run_output_dir else None
                if summary_path and summary_path.exists():
                    try:
                        summary_text = summary_path.read_text(encoding="utf-8").strip()
                        if summary_text:
                            st.markdown("---")
                            st.markdown("#### 🤖 What the agent has done so far")
                            escaped = html.escape(summary_text).replace("\n", "<br>")
                            st.markdown(
                                f'<div class="agent-summary-box"><div class="agent-summary-content">{escaped}</div></div>',
                                unsafe_allow_html=True,
                            )
                    except Exception:
                        pass
        if show_chat and chat_col is not None:
            with chat_col:
                num_total = st.session_state.get("run_num_analyses", 1)
                if num_total > 1:
                    sel_key = "chat_analysis_selection_pause"
                    if sel_key not in st.session_state:
                        st.session_state[sel_key] = 1
                    chosen = st.selectbox(
                        "Chat about",
                        options=list(range(1, num_total + 1)),
                        format_func=lambda x: f"Analysis {x}",
                        key=sel_key,
                    )
                    g._render_chat_box(st.session_state.run_output_dir, analysis_idx=chosen)
                else:
                    g._render_chat_box(st.session_state.run_output_dir, analysis_idx=1)
        if nb_path and Path(nb_path).exists() and result:
            out = result
            cell_sources, feedback, nb, run_clicked, insert_after_idx, insert_type, continue_clicked, finish_clicked, edit_clicked = out[:9]
            save_clicked = out[9] if len(out) > 9 else False
            for i, src in enumerate(cell_sources):
                nb.cells[i].source = src
            with open(nb_path, "w", encoding="utf-8") as f:
                nbf.write(nb, f)
            if run_clicked is not None:
                exec_path = g._pause_execute_path()
                if exec_path:
                    exec_path.write_text(json.dumps({"cell_index": run_clicked}), encoding="utf-8")
                    with st.spinner("Executing cell..."):
                        for _ in range(300):
                            time.sleep(0.05)
                            if not exec_path.exists():
                                break
                st.rerun()
            elif insert_type is not None:
                new_cell = new_code_cell("") if insert_type == "code" else new_markdown_cell("")
                new_cell["cell_type"] = insert_type
                import uuid
                new_cell["id"] = f"gui_{uuid.uuid4().hex[:12]}"
                insert_at = insert_after_idx + 1
                nb.cells.insert(insert_at, new_cell)
                with open(nb_path, "w", encoding="utf-8") as f:
                    nbf.write(nb, f)
                st.session_state["open_edit_cell"] = insert_at
                st.rerun()
            elif edit_clicked:
                st.session_state[f"pause_edit_mode_{pause_id}"] = True
                st.session_state["open_edit_cell"] = 0
                st.rerun()
            elif continue_clicked:
                response_path.write_text(feedback or "", encoding="utf-8")
                request_path.unlink(missing_ok=True)
                st.rerun()
            elif finish_clicked:
                response_path.write_text("finish the analysis", encoding="utf-8")
                request_path.unlink(missing_ok=True)
                st.rerun()
        else:
            st.warning("Notebook path not found. The agent may have advanced.")
            if st.button("Continue (send empty feedback)"):
                response_path.write_text("", encoding="utf-8")
                request_path.unlink(missing_ok=True)
                st.rerun()
    else:
        show_chat = g._should_show_chat()
        if show_chat:
            main_col, chat_col = st.columns([0.72, 0.28])
        else:
            main_col = st.container()
            chat_col = None

        with main_col:
            st.markdown("---")
            run_col_title, run_col_status = st.columns([1, 0.4])
            with run_col_title:
                st.markdown("### ▶ Analysis running")
                num_total = st.session_state.get("run_num_analyses", 1)
                if num_total > 1:
                    completed, current = g._parse_run_progress(output_text)
                    progress_label = f"Analysis {current} of {num_total} running"
                    if completed > 0:
                        progress_label = f"Analysis {completed} of {num_total} complete · {progress_label}"
                    st.caption(progress_label)
            with run_col_status:
                st.markdown(
                    """
                    <div style="display:inline-flex;align-items:center;gap:0.5rem;padding:0.65rem 1.25rem;border:3px solid #0d7377;border-radius:12px;background:linear-gradient(135deg,#e8f6f7,#d4efef);font-weight:700;font-size:1rem;color:#0d7377;box-shadow:0 4px 12px rgba(13,115,119,0.25);">
                        <span style="display:inline-block;width:1rem;height:1rem;border:2px solid rgba(13,115,119,0.3);border-top-color:#0d7377;border-radius:50%;animation:status-spin 0.8s linear infinite;"></span>
                        <span>Status: Running</span>
                    </div>
                    <style>@keyframes status-spin{to{transform:rotate(360deg);}}</style>
                    """,
                    unsafe_allow_html=True,
                )
            with st.expander("📋 Output log", expanded=True):
                st.text_area("Log", value=output_text, height=200, disabled=True, label_visibility="collapsed")
            st.markdown("#### Notebook")
            num_total = st.session_state.get("run_num_analyses", 1)
            completed, current = g._parse_run_progress(output_text) if num_total > 1 else (0, 1)
            if num_total > 1:
                analyses = g._collect_notebooks_by_analysis(st.session_state.run_output_dir, num_total)
                tab_labels = [f"Analysis {i}" + (" ✓" if nb and i <= completed else " ●" if nb and i == current else "") for i, nb in analyses]
                tabs = st.tabs(tab_labels)
                for i, (analysis_idx, nb_path) in enumerate(analyses):
                    with tabs[i]:
                        if nb_path:
                            g._render_notebook_jupyter_style(
                                nb_path, editable=False,
                                save_snapshot=True, output_dir=st.session_state.run_output_dir,
                            )
                        else:
                            st.info("Pending" if analysis_idx > current else "In progress...")
            else:
                notebooks = g._collect_notebooks(st.session_state.run_output_dir)
                if notebooks:
                    for run_name, nb_path in notebooks:
                        st.markdown(f"**📓 {run_name} / {Path(nb_path).name}**")
                        g._render_notebook_jupyter_style(
                            nb_path, editable=False,
                            save_snapshot=True, output_dir=st.session_state.run_output_dir,
                        )
                else:
                    st.info("Notebooks will appear here as they are created.")

        if show_chat and chat_col is not None:
            with chat_col:
                num_total = st.session_state.get("run_num_analyses", 1)
                if num_total > 1:
                    sel_key = "chat_analysis_selection_run"
                    if sel_key not in st.session_state:
                        st.session_state[sel_key] = 1
                    chosen = st.selectbox(
                        "Chat about",
                        options=list(range(1, num_total + 1)),
                        format_func=lambda x: f"Analysis {x}",
                        key=sel_key,
                    )
                    g._render_chat_box(st.session_state.run_output_dir, analysis_idx=chosen)
                else:
                    g._render_chat_box(st.session_state.run_output_dir, analysis_idx=1)

        proc_done = proc is not None and proc.poll() is not None
        pid_done = (
            st.session_state.run_pid is not None
            and not g._process_alive(st.session_state.run_pid)
        )
        if proc_done or pid_done:
            out_dir = Path(st.session_state.run_output_dir) if st.session_state.run_output_dir else None
            if out_dir:
                (out_dir / g._RUN_PID_FILE).unlink(missing_ok=True)
            st.session_state.run_proc = None
            st.session_state.run_pid = None
            st.session_state.run_started = False
            st.session_state.run_thread_started = False
            if proc_done:
                if proc.returncode == 0:
                    num_total = st.session_state.get("run_num_analyses", 1)
                    msg = f"✅ All {num_total} analyses complete!" if num_total > 1 else "✅ Analysis complete!"
                    st.success(msg)
                else:
                    st.error(f"Analysis exited with code {proc.returncode}")
            else:
                num_total = st.session_state.get("run_num_analyses", 1)
                msg = f"✅ All {num_total} analyses complete!" if num_total > 1 else "✅ Analysis complete!"
                st.success(msg)
            # Transition to completed view (notebooks + Finish) — stay on analysis page
            st.rerun()
        else:
            time.sleep(2)
            st.rerun()

# Interactive edit screen (when analysis was stopped — no agent, but full edit UI)
elif st.session_state.run_output_dir and not st.session_state.run_started and st.session_state.get("run_show_interactive"):
    request_path = g._pause_request_path()
    response_path = g._pause_response_path()
    nb_path = ""
    if request_path and request_path.exists():
        try:
            nb_path = request_path.read_text(encoding="utf-8").strip()
        except Exception:
            pass
    if nb_path and Path(nb_path).exists():
        show_chat = g._should_show_chat()
        if show_chat:
            main_col, chat_col = st.columns([0.72, 0.28])
        else:
            main_col = st.container()
            chat_col = None
        with main_col:
            st.markdown("---")
            st.markdown("### ⏹ Analysis stopped")
            st.caption("Edit the notebook below. Click **Continue** to resume the analysis, or **Finish** to return home.")
            import nbformat as nbf
            from nbformat.v4 import new_code_cell, new_markdown_cell
            pause_id = "killed"
            with st.form("killed_interactive_form"):
                st.caption("Edit cells, add new ones. Click Continue to resume the analysis, or Finish to go home.")
                result = g._render_notebook_jupyter_style(nb_path, editable=True, pause_id=pause_id)
            if result:
                cell_sources, feedback, nb, run_clicked, insert_after_idx, insert_type, continue_clicked, finish_clicked, edit_clicked = result[:9]
                for i, src in enumerate(cell_sources):
                    nb.cells[i].source = src
                with open(nb_path, "w", encoding="utf-8") as f:
                    nbf.write(nb, f)
                if insert_type is not None and insert_after_idx >= 0:
                    new_cell = new_code_cell("") if insert_type == "code" else new_markdown_cell("")
                    new_cell["cell_type"] = insert_type
                    import uuid
                    new_cell["id"] = f"gui_{uuid.uuid4().hex[:12]}"
                    nb.cells.insert(insert_after_idx + 1, new_cell)
                    with open(nb_path, "w", encoding="utf-8") as f:
                        nbf.write(nb, f)
                if edit_clicked:
                    st.session_state[f"pause_edit_mode_{pause_id}"] = True
                    st.session_state["open_edit_cell"] = 0
                    st.rerun()
                elif continue_clicked:
                    st.session_state.run_show_interactive = False
                    if request_path and request_path.exists():
                        request_path.unlink(missing_ok=True)
                    m = __import__("re").search(r"_analysis_(\d+)\.ipynb", Path(nb_path).name, __import__("re").I)
                    analysis_idx = int(m.group(1)) if m else 1
                    g._launch_resume(st.session_state.run_output_dir, analysis_idx, run_to_completion=True)
                    st.rerun()
                elif finish_clicked:
                    st.session_state.run_output_dir = None
                    st.session_state.run_show_interactive = False
                    if request_path and request_path.exists():
                        request_path.unlink(missing_ok=True)
                    st.switch_page("gui.py")
        if show_chat and chat_col is not None:
            with chat_col:
                analyses = g._collect_notebooks_by_analysis(st.session_state.run_output_dir, st.session_state.get("run_num_analyses", 1))
                if len(analyses) > 1:
                    sel_key = "chat_analysis_selection_killed"
                    if sel_key not in st.session_state:
                        m = __import__("re").search(r"_analysis_(\d+)\.ipynb", Path(nb_path).name, __import__("re").I)
                        st.session_state[sel_key] = int(m.group(1)) if m else 1
                    st.selectbox("Chat about", options=list(range(1, len(analyses) + 1)), format_func=lambda x: f"Analysis {x}", key=sel_key)
                    g._render_chat_box(st.session_state.run_output_dir, analysis_idx=st.session_state.get(sel_key, 1))
                else:
                    g._render_chat_box(st.session_state.run_output_dir, analysis_idx=1)
    else:
        st.warning("No notebook found. Click Finish to return home.")
        if st.button("🏠 Finish", type="primary"):
            st.session_state.run_output_dir = None
            st.session_state.run_show_interactive = False
            if request_path and request_path.exists():
                request_path.unlink(missing_ok=True)
            st.switch_page("gui.py")

# Notebook viewer (when analysis completed)
elif st.session_state.run_output_dir and not st.session_state.run_started:
    # Ensure we have a valid output dir (restore from file if needed)
    out_dir = st.session_state.run_output_dir
    if not out_dir or not Path(out_dir).exists():
        if g._LAST_RUN_FILE.exists():
            try:
                last_dir = g._LAST_RUN_FILE.read_text(encoding="utf-8").strip()
                if last_dir and Path(last_dir).exists():
                    st.session_state.run_output_dir = last_dir
                    st.rerun()
            except Exception:
                pass
    show_chat = g._should_show_chat()
    if show_chat:
        main_col, chat_col = st.columns([0.72, 0.28])
    else:
        main_col = st.container()
        chat_col = None

    with main_col:
        st.markdown("---")
        st.markdown("### 📓 Notebooks")
        st.caption("From your last run.")

        num_total = st.session_state.get("run_num_analyses", 1)
        save_to_path = None
        save_nb = None

        if num_total > 1:
            analyses = g._collect_notebooks_by_analysis(st.session_state.run_output_dir, num_total)
            tab_labels = [f"Analysis {i}" for i, _ in analyses]
            tabs = st.tabs(tab_labels)
            for i, (analysis_idx, nb_path) in enumerate(analyses):
                with tabs[i]:
                    if nb_path:
                        with st.form(f"analysis_completed_{i}"):
                            result = g._render_notebook_jupyter_style(
                                nb_path, editable=True, pause_id=f"done_{i}", standalone_edit=True
                            )
                            if result:
                                edit_clicked = result[7]
                                save_clicked = result[9] if len(result) > 9 else False
                                if edit_clicked:
                                    st.session_state[f"pause_edit_mode_done_{i}"] = True
                                    st.session_state["open_edit_cell"] = 0
                                    st.rerun()
                                elif save_clicked:
                                    save_to_path = nb_path
                                    save_nb = result[2]
                                    cell_sources, _, _, _, insert_after_idx, insert_type = result[0], result[1], result[2], result[3], result[4], result[5]
                                    for j, src in enumerate(cell_sources):
                                        save_nb.cells[j].source = src
                                    if insert_type is not None and insert_after_idx >= 0:
                                        from nbformat.v4 import new_code_cell, new_markdown_cell
                                        import uuid
                                        new_cell = new_code_cell("") if insert_type == "code" else new_markdown_cell("")
                                        new_cell["cell_type"] = insert_type
                                        new_cell["id"] = f"gui_{uuid.uuid4().hex[:12]}"
                                        save_nb.cells.insert(insert_after_idx + 1, new_cell)
                        config_path = Path(st.session_state.run_output_dir) / g._RUN_CONFIG_FILE
                        continue_pending = st.session_state.get("continue_further_pending")
                        if continue_pending == (st.session_state.run_output_dir, analysis_idx):
                            st.warning(
                                "The agent will re-run this analysis to restore kernel state. "
                                "You can then run cells, edit, and give feedback interactively."
                            )
                            col_a, col_b = st.columns(2)
                            with col_a:
                                if st.button("Start", key=f"continue_confirm_{i}", type="primary"):
                                    st.session_state.pop("continue_further_pending", None)
                                    g._launch_resume(st.session_state.run_output_dir, analysis_idx)
                                    st.rerun()
                            with col_b:
                                if st.button("Cancel", key=f"continue_cancel_{i}"):
                                    st.session_state.pop("continue_further_pending", None)
                                    st.rerun()
                        elif config_path.exists() and st.button("▶ Continue further", key=f"continue_further_{i}", help="Re-run to restore state, then interact (run, edit, feedback)"):
                            st.session_state.continue_further_pending = (st.session_state.run_output_dir, analysis_idx)
                            st.rerun()
                    else:
                        st.info("No notebook for this analysis.")
        else:
            notebooks = g._collect_notebooks(st.session_state.run_output_dir)
            if not notebooks:
                st.info("Run an analysis to see notebooks here.")
            else:
                for run_name, nb_path in notebooks:
                    nb_label = Path(nb_path).name
                    with st.expander(f"📓 {run_name} / {nb_label}"):
                        g._render_notebook(nb_path)
                analyses_single = g._collect_notebooks_by_analysis(st.session_state.run_output_dir, 1)
                if analyses_single and analyses_single[0][1]:
                    _, nb_path = analyses_single[0]
                    analysis_idx = 1
                    config_path = Path(st.session_state.run_output_dir) / g._RUN_CONFIG_FILE
                    continue_pending = st.session_state.get("continue_further_pending")
                    if continue_pending == (st.session_state.run_output_dir, analysis_idx):
                        st.warning(
                            "The agent will re-run this analysis to restore kernel state. "
                            "You can then run cells, edit, and give feedback interactively."
                        )
                        col_a, col_b = st.columns(2)
                        with col_a:
                            if st.button("Start", key="continue_confirm_single", type="primary"):
                                st.session_state.pop("continue_further_pending", None)
                                g._launch_resume(st.session_state.run_output_dir, analysis_idx)
                                st.rerun()
                        with col_b:
                            if st.button("Cancel", key="continue_cancel_single"):
                                st.session_state.pop("continue_further_pending", None)
                                st.rerun()
                    elif config_path.exists() and st.button("▶ Continue further", key="continue_further_single", help="Re-run to restore state, then interact (run, edit, feedback)"):
                        st.session_state.continue_further_pending = (st.session_state.run_output_dir, analysis_idx)
                        st.rerun()

        if save_to_path and save_nb is not None:
            import nbformat as nbf
            with open(save_to_path, "w", encoding="utf-8") as f:
                nbf.write(save_nb, f)
            st.rerun()

        if st.session_state.run_output_dir:
            if st.button("Finish", type="primary", help="Return to home"):
                st.session_state.run_output_dir = None
                st.switch_page("gui.py")

    if show_chat and chat_col is not None and st.session_state.run_output_dir:
        with chat_col:
            num_total = st.session_state.get("run_num_analyses", 1)
            if num_total > 1:
                sel_key = "chat_analysis_selection"
                if sel_key not in st.session_state:
                    st.session_state[sel_key] = 1
                chosen = st.selectbox(
                    "Chat about",
                    options=list(range(1, num_total + 1)),
                    format_func=lambda x: f"Analysis {x}",
                    key=sel_key,
                )
                g._render_chat_box(st.session_state.run_output_dir, analysis_idx=chosen)
            else:
                g._render_chat_box(st.session_state.run_output_dir, analysis_idx=1)

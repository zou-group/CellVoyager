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
        st.caption("Analysis in progress — notebooks, feedback, chat")
else:
    st.title("CellVoyager")
    st.caption("Analysis in progress")

st.divider()

# Sidebar: Kill / Finish only
with st.sidebar:
    st.markdown("### ▶ Analysis")
    if g._has_live_run():
        if st.button("⏹ Kill analysis", type="primary", use_container_width=True, help="Stop the analysis and return to home"):
            g._kill_analysis()
            st.switch_page("gui.py")
    else:
        if st.button("🏠 Finish", type="primary", use_container_width=True, help="Return to home and start a new analysis"):
            st.session_state.run_output_dir = None
            st.switch_page("gui.py")

# Main: running / pause / notebook viewer
if st.session_state.run_started and g._has_live_run():
    proc = st.session_state.run_proc
    output_text = g._get_run_log()
    request_path = g._pause_request_path()
    response_path = g._pause_response_path()
    in_pause_ui = (
        st.session_state.run_interactive_mode
        and request_path
        and response_path
        and request_path.exists()
    )

    if in_pause_ui:
        show_chat = g._should_show_chat()
        if show_chat:
            main_col, chat_col = st.columns([0.72, 0.28])
        else:
            main_col = st.container()
            chat_col = None

        result = None
        nb_path = ""
        with main_col:
            st.markdown("---")
            st.markdown("### ⏸ Agent waiting for feedback")
            st.caption("Edit cells, insert new ones, run code. Click **Continue** when ready for the agent to proceed.")
            try:
                nb_path = request_path.read_text(encoding="utf-8").strip()
            except Exception:
                pass
            if nb_path and Path(nb_path).exists():
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
                g._render_chat_box(st.session_state.run_output_dir)
        if nb_path and Path(nb_path).exists() and result:
            cell_sources, feedback, nb, run_clicked, insert_after_idx, insert_type, continue_clicked, finish_clicked, edit_clicked = result
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
            with st.expander("📋 Output log", expanded=False):
                st.text_area("Log", value=output_text, height=200, disabled=True, label_visibility="collapsed")
            st.markdown("#### Notebook")
            notebooks = g._collect_notebooks(st.session_state.run_output_dir)
            if notebooks:
                for run_name, nb_path in notebooks:
                    nb_label = Path(nb_path).name
                    st.markdown(f"**📓 {run_name} / {nb_label}**")
                    g._render_notebook_jupyter_style(nb_path, editable=False)
            else:
                st.info("Notebooks will appear here as they are created.")

        if show_chat and chat_col is not None:
            with chat_col:
                g._render_chat_box(st.session_state.run_output_dir)

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
                    st.success("✅ Analysis complete!")
                else:
                    st.error(f"Analysis exited with code {proc.returncode}")
            else:
                st.success("✅ Analysis complete!")
            # Transition to completed view (notebooks + Finish) — stay on analysis page
            st.rerun()
        else:
            time.sleep(2)
            st.rerun()

# Notebook viewer (when analysis completed)
elif st.session_state.run_output_dir and not st.session_state.run_started:
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

        notebooks = g._collect_notebooks(st.session_state.run_output_dir)
        if not notebooks:
            st.info("Run an analysis to see notebooks here.")
        else:
            for run_name, nb_path in notebooks:
                nb_label = Path(nb_path).name
                with st.expander(f"📓 {run_name} / {nb_label}"):
                    g._render_notebook(nb_path)
        if st.session_state.run_output_dir:
            if st.button("Finish", type="primary", help="Return to home"):
                st.session_state.run_output_dir = None
                st.switch_page("gui.py")

    if show_chat and chat_col is not None and st.session_state.run_output_dir:
        with chat_col:
            g._render_chat_box(st.session_state.run_output_dir)

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
        except Exception:
            pass
if not st.session_state.get("run_output_dir"):
    st.switch_page("gui.py")

# Recover running state after page refresh/session reset by checking persisted PID.
run_output_dir = st.session_state.get("run_output_dir")
if run_output_dir:
    run_pid = st.session_state.get("run_pid")
    if run_pid is None:
        pid_file = Path(run_output_dir) / g._RUN_PID_FILE
        if pid_file.exists():
            try:
                run_pid = int(pid_file.read_text(encoding="utf-8").strip())
                st.session_state.run_pid = run_pid
            except (ValueError, OSError):
                st.session_state.run_pid = None
                run_pid = None
    if run_pid is not None and g._process_alive(run_pid):
        st.session_state.run_started = True

LOGO_PATH = g.ROOT / "images" / "symbol.jpeg"
st.set_page_config(
    page_title="CellVoyager — Analysis",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else "📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Checked immediately after set_page_config so it fires regardless of any rerun
# triggered elsewhere on the page (transition block, fragment, etc.)
if st.session_state.pop("_go_home", False):
    st.switch_page("gui.py")

st.markdown("""
<style>
    /* Hide Streamlit toolbar, footer, and top decoration line */
    header[data-testid="stHeader"] {
        background: transparent !important;
        border-bottom: none !important;
    }
    #MainMenu { display: none !important; }
    footer { display: none !important; }
    [data-testid="stDecoration"] { display: none !important; }
    section[data-testid="stSidebar"]::before { display: none !important; }
    section[data-testid="stSidebar"] > div { border-top: none !important; }

    /* Disable Streamlit's element fade-in animation in the sidebar to prevent flicker on reruns */
    section[data-testid="stSidebar"] * {
        animation-duration: 0s !important;
        animation-delay: 0s !important;
        transition-duration: 0s !important;
    }
    .stApp > div:first-child { border-top: none !important; }

    /* Typography — minimum 1.25rem */
    h1 { font-size: 2rem !important; font-weight: 600 !important; letter-spacing: -0.02em !important; }
    h3 { font-size: 1.5rem !important; color: #0f172a !important; font-weight: 650 !important; margin-top: 0.5rem !important; }
    h4 { font-size: 1.35rem !important; color: #0f172a !important; font-weight: 650 !important; }

    .stMarkdown p, .stText { font-size: 1.25rem !important; line-height: 1.65 !important; }
    .stCaption, [data-testid="stCaptionContainer"] p { font-size: 1.25rem !important; }

    /* Widget labels */
    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] p,
    label, label p { font-size: 1.25rem !important; }

    /* Inputs */
    .stTextInput input, .stNumberInput input { font-size: 1.25rem !important; }
    .stTextArea textarea { font-size: 1.25rem !important; }
    .stSelectbox > div > div { font-size: 1.25rem !important; }

    /* Expander labels */
    div[data-testid="stExpander"] summary p { font-size: 1.25rem !important; font-weight: 600 !important; }

    /* Alerts */
    div[data-testid="stAlert"] { border-radius: 8px !important; font-size: 1.25rem !important; }
    div[data-testid="stAlert"] p { font-size: 1.25rem !important; }

    /* Tabs */
    div[data-testid="stTabs"] button[role="tab"] {
        font-weight: 600 !important;
        font-size: 1.5rem !important;
        padding: 0.6rem 1.2rem !important;
    }

    /* Layout */
    div[data-testid="stVerticalBlock"] > div { gap: 0.5rem !important; }

    /* Sidebar */
    section[data-testid="stSidebar"] > div {
        background: #f7fbff;
        border-right: 1px solid #dbe4f0;
        width: 22rem !important;
    }
    section[data-testid="stSidebar"] { width: 22rem !important; min-width: 22rem !important; }
    section[data-testid="stSidebar"] .stMarkdown h3 {
        font-size: 1rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
        color: #64748b !important;
        font-weight: 700 !important;
        margin-top: 1.4rem !important;
        margin-bottom: 0.4rem !important;
    }
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] label p,
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stCaption { font-size: 1.25rem !important; }

    /* Primary buttons */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0f766e, #0a6a63) !important;
        border: 0 !important;
        color: #fff !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 1.25rem !important;
        box-shadow: 0 2px 10px rgba(15, 118, 110, 0.22);
        transition: transform 0.12s ease, box-shadow 0.12s ease;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 5px 16px rgba(15, 118, 110, 0.30);
    }

    /* Form submit buttons (Edit, Continue, Finish, Run Code Cell, etc.) */
    div[data-testid="stFormSubmitButton"] > button {
        min-height: 44px !important;
        font-size: 1.25rem !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
    }
    div[data-testid="stFormSubmitButton"] > button[kind="primary"] {
        min-height: 52px !important;
        font-size: 1.25rem !important;
        font-weight: 700 !important;
    }

    /* Expanders */
    div[data-testid="stExpander"] {
        border-radius: 10px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
        border: 1px solid #dbe4f0 !important;
        overflow: hidden;
    }

    /* Running spinner keyframes */
    @keyframes status-spin { to { transform: rotate(360deg); } }

    /* Agent summary box — shown when agent pauses */
    .agent-summary-box {
        background: linear-gradient(135deg, #f0fdf4, #f7fef9);
        border: 1px solid #bbf7d0;
        border-left: 4px solid #16a34a;
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        margin: 0.75rem 0;
    }
    .agent-summary-content {
        color: #14532d;
        font-size: 1.25rem;
        line-height: 1.7;
    }

    /* Chat panel */
    .chat-box-header {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0 0 0.15rem;
    }
    .chat-box-caption {
        font-size: 1.1rem;
        color: #64748b;
        margin: 0 0 0.5rem;
        line-height: 1.4;
    }
    /* Strip the chat input form's own border so it blends into the outer container */
    [data-testid="column"]:nth-child(2) [data-testid="stForm"] {
        border: none !important;
        background: transparent !important;
        padding: 0.25rem 0 0 !important;
        border-top: 1px solid #e5e7eb !important;
        border-radius: 0 !important;
        margin-top: 0.1rem !important;
    }
    /* Send button inside the chat form */
    [data-testid="column"]:nth-child(2) [data-testid="stFormSubmitButton"] > button {
        min-height: 38px !important;
        font-size: 1.25rem !important;
        padding: 0 0.6rem !important;
    }

    /* Feedback section */
    .feedback-box-header {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.35rem;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid #e2e8f0;
    }
    /* Feedback textarea — stand out from background */
    .stTextArea textarea {
        background: #ffffff !important;
        border: 1.5px solid #c8d6e8 !important;
    }

    /* Notebook code blocks */
    [data-testid="stCode"] pre,
    [data-testid="stCode"] code {
        font-size: 1.25rem !important;
        line-height: 1.55 !important;
    }

    /* Notebook markdown cell body text (not headers) */
    [data-testid="stMarkdown"] p,
    [data-testid="stMarkdownContainer"] p {
        font-size: 1.25rem !important;
        line-height: 1.65 !important;
    }
    [data-testid="stMarkdown"] li,
    [data-testid="stMarkdownContainer"] li {
        font-size: 1.25rem !important;
    }

    /* Notebook cell output — click to toggle scrollable */
    .cv-cell-output {
        transition: max-height 0.25s ease;
    }
    .cv-cell-output.cv-scrollable {
        height: 260px;
        min-height: 80px;
        overflow-y: auto;
        resize: vertical;
        border-color: #94a3b8 !important;
    }

    /* Notebook HTML output containers */
    .output_html {
        overflow-x: auto;
        max-width: 100%;
    }
    .output_html table {
        border-collapse: collapse;
        font-size: 1.25rem;
    }
    .output_html th, .output_html td {
        border: 1px solid #e2e8f0;
        padding: 0.35rem 0.75rem;
    }
    .output_html th {
        background: #f8fafc;
        font-weight: 600;
        color: #374151;
    }
    .output_html tr:nth-child(even) td { background: #fafafa; }
</style>
""", unsafe_allow_html=True)


def _install_continue_scroll_restore() -> None:
    """Preserve scroll position when Continue triggers a rerun."""
    st.components.v1.html(
        """
        <script>
        (function () {
          try {
            const p = window.parent;
            if (!p) return;
            const KEY_Y = "cellvoyager_scroll_y";
            const KEY_RESTORE = "cellvoyager_scroll_restore";

            if (p.sessionStorage.getItem(KEY_RESTORE) === "1") {
              const y = Number(p.sessionStorage.getItem(KEY_Y) || "0");
              p.requestAnimationFrame(() => p.scrollTo(0, y));
              setTimeout(() => p.scrollTo(0, y), 120);
              p.sessionStorage.removeItem(KEY_RESTORE);
            }

            const wireButtons = () => {
              const buttons = p.document.querySelectorAll("button");
              buttons.forEach((btn) => {
                if (btn.dataset.cvContinueScrollHooked === "1") return;
                btn.dataset.cvContinueScrollHooked = "1";
                btn.addEventListener("click", () => {
                  const label = (btn.innerText || "").trim();
                  if (label.includes("Continue")) {
                    const y = p.scrollY || p.pageYOffset || 0;
                    p.sessionStorage.setItem(KEY_Y, String(y));
                    p.sessionStorage.setItem(KEY_RESTORE, "1");
                  }
                }, true);
              });
            };

            wireButtons();
            const root = p.document.querySelector("section.main") || p.document.body;
            if (!root) return;
            const observer = new p.MutationObserver(() => wireButtons());
            observer.observe(root, { childList: true, subtree: true });
          } catch (e) {}
        })();
        </script>
        """,
        height=0,
    )


_install_continue_scroll_restore()

# Wire click-to-collapse on cell outputs (event delegation, runs once)
st.components.v1.html("""
<script>
(function () {
  try {
    const p = window.parent;
    if (p._cvOutputClickWired) return;
    p._cvOutputClickWired = true;
    let _mdX = 0, _mdY = 0;
    p.document.addEventListener('mousedown', function (e) {
      _mdX = e.clientX; _mdY = e.clientY;
    });
    p.document.addEventListener('click', function (e) {
      // Ignore if the mouse moved (i.e. a resize drag, not a real click)
      if (Math.abs(e.clientX - _mdX) > 4 || Math.abs(e.clientY - _mdY) > 4) return;
      const output = e.target.closest('.cv-cell-output');
      if (output) output.classList.add('cv-scrollable');
    });
  } catch (e) {}
})();
</script>
""", height=0)

run_output_dir = st.session_state.get("run_output_dir")
pause_request_path = g._pause_request_path()
pause_response_path = g._pause_response_path()
stop_request_path = (Path(run_output_dir) / g._STOP_REQUEST_FILE) if run_output_dir else None
_transitioning = st.session_state.pop("_transitioning_from_pause", False)
is_paused_now = bool(pause_request_path and pause_request_path.exists()) and not _transitioning
is_stop_pending = bool(
    stop_request_path
    and stop_request_path.exists()
    and not is_paused_now
)
# Compute pause_id here so the sidebar can share the feedback widget key with the main area
pause_id = (
    f"paused_{pause_request_path.stat().st_mtime}"
    if pause_request_path and pause_request_path.exists()
    else "no_pause"
)

# Determine whether the current pause is a between-analysis pause (agent summary starts with ✅)
_is_between_pause = False
if is_paused_now and run_output_dir:
    _sbp = Path(run_output_dir) / g._AGENT_SUMMARY_FILE
    try:
        if _sbp.exists():
            _is_between_pause = _sbp.read_text(encoding="utf-8").startswith("✅")
    except Exception:
        pass

# Sidebar: logo + Stop / Finish only
SIDEBAR_LOGO = g.ROOT / "images" / "logo.jpeg"
with st.sidebar:
    if SIDEBAR_LOGO.exists():
        st.image(str(SIDEBAR_LOGO), width="stretch")
    st.divider()
    if st.button("← Home", width="stretch", help="Return to home screen. The analysis will be stopped if running."):
        if g._has_live_run():
            g._kill_analysis(show_interactive_edit=False)
        st.session_state.run_started = False
        st.session_state.run_pid = None
        st.session_state.run_proc = None
        st.session_state.run_output_dir = None
        st.session_state.pop("run_error", None)
        st.switch_page("gui.py")
    st.divider()
    if g._has_live_run():
        st.session_state.pop("_resume_restoring", None)  # clear transitional restoring state
        if is_paused_now:
            _action_pending = any(st.session_state.get(k) for k in ["_sb_continue", "_sb_edit", "_sb_finish", "_sb_next_analysis"])
            if _action_pending:
                st.caption("Processing...")
            else:
                if _is_between_pause:
                    if st.button("Continue to Next Analysis", type="primary", width="stretch"):
                        st.session_state["_sb_next_analysis"] = True
                    st.caption("Proceed to the next analysis.")
                    st.markdown('<div style="margin-bottom:0.6rem"></div>', unsafe_allow_html=True)
                    st.markdown("**💬 Feedback for the agent**")
                    st.text_area(
                        "Feedback",
                        placeholder="e.g., focus more on cluster 3, or skip the next visualization...",
                        height=120,
                        key=f"pause_feedback_{pause_id}",
                        label_visibility="collapsed",
                    )
                    st.markdown('<div style="margin-bottom:0.4rem"></div>', unsafe_allow_html=True)
                    if st.button("Continue Analysis", width="stretch"):
                        st.session_state["_sb_continue"] = True
                    st.caption("Keep extending the current analysis with the provided feedback")
                else:
                    st.markdown("**💬 Feedback for the agent**")
                    st.text_area(
                        "Feedback",
                        placeholder="e.g., focus more on cluster 3, or skip the next visualization...",
                        height=120,
                        key=f"pause_feedback_{pause_id}",
                        label_visibility="collapsed",
                    )
                    st.markdown('<div style="margin-bottom:0.4rem"></div>', unsafe_allow_html=True)
                    if st.button("Continue Analysis", type="primary", width="stretch"):
                        st.session_state["_sb_continue"] = True
                    st.caption("Continues the analysis with the provided feedback (if any)")
                st.markdown('<div style="margin-bottom:0.6rem"></div>', unsafe_allow_html=True)
                if st.button("Edit Analysis", width="stretch"):
                    st.session_state["_sb_edit"] = True
                st.caption("Lets you add and run your own code as well as edit any existing code/text")
                st.markdown('<div style="margin-bottom:0.6rem"></div>', unsafe_allow_html=True)
                if st.button("Finish Analysis", width="stretch"):
                    st.session_state["_sb_finish"] = True
                st.caption("Tells the agent to finish the analysis in the next step, then stop")
        elif is_stop_pending:
            st.info("Stop requested. Finishing current sub-step, then pausing for feedback...")
        else:
            if st.button("⏹ Stop Analysis", type="primary", width="stretch", help="Pause the analysis at the next safe sub-step boundary, then continue from there."):
                if not g._request_pause():
                    # Fallback only when pause files cannot be created.
                    g._kill_analysis(show_interactive_edit=True)
                st.rerun()
    else:
        # Completed view sidebar: per-analysis Continue further + Finish
        _completed_out_dir = st.session_state.run_output_dir
        _completed_num = st.session_state.get("run_num_analyses", 1)
        if _completed_out_dir:
            _completed_analyses = g._collect_notebooks_by_analysis(_completed_out_dir, _completed_num)
            _cfg_path_done = Path(_completed_out_dir) / g._RUN_CONFIG_FILE
            _resume_restoring = st.session_state.get("_resume_restoring")
            for _aidx, _nb in _completed_analyses:
                if _nb:
                    if _resume_restoring == (_completed_out_dir, _aidx):
                        st.info(f"Restoring Analysis {_aidx} (re-running the notebook)...")
                    elif _cfg_path_done.exists():
                        _lbl = f"▶ Continue Analysis {_aidx}" if _completed_num > 1 else "▶ Continue further"
                        if st.button(_lbl, key=f"sb_continue_further_{_aidx}", width="stretch"):
                            st.session_state._resume_restoring = (_completed_out_dir, _aidx)
                            g._launch_resume(_completed_out_dir, _aidx)
                            st.rerun()
            st.markdown('<div style="margin-bottom:0.5rem"></div>', unsafe_allow_html=True)
        if st.button("Finish Analysis", type="primary", width="stretch", help="Return to home and start a new analysis"):
            st.session_state.run_output_dir = None
            st.session_state.pop("run_error", None)
            st.switch_page("gui.py")

# Main: running / pause / notebook viewer
# If run was started but process is dead (e.g. finished, or server restarted), transition to completed view
if st.session_state.run_started and not g._has_live_run():
    st.session_state.run_proc = None
    st.session_state.run_pid = None
    st.session_state.run_started = False
    st.session_state.run_thread_started = False
    st.session_state.pop("_active_run_tab", None)
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
            # Restore error state if we lost it (e.g. page refresh after crash)
            error_file = out_dir / g._RUN_ERROR_FILE
            if error_file.exists() and not st.session_state.get("run_error"):
                try:
                    st.session_state["run_error"] = error_file.read_text(encoding="utf-8").strip()
                except Exception:
                    pass
    st.rerun()

@st.fragment(run_every=2)
def _running_view():
    """Polls every 2 s without re-rendering the sidebar."""
    run_output_dir = st.session_state.get("run_output_dir")
    proc = st.session_state.get("run_proc")
    output_text = g._get_run_log()

    # If agent has paused, trigger a full rerun so the pause UI can show
    _req = g._pause_request_path()
    _res = g._pause_response_path()
    if _req and _res and _req.exists():
        st.rerun()
        return

    stop_request_path = (Path(run_output_dir) / g._STOP_REQUEST_FILE) if run_output_dir else None
    _is_stop_pending = bool(stop_request_path and stop_request_path.exists())

    show_chat = g._should_show_chat()
    if show_chat:
        main_col, _gap, chat_col = st.columns([0.831, 0.02, 0.236])
    else:
        main_col = st.container()
        chat_col = None

    with main_col:
        st.markdown("---")
        if st.session_state.get("_is_resuming"):
            st.warning(
                "**Restoring kernel state** — The agent is re-running existing notebook cells to restore "
                "the Python kernel. This may take a moment. Once complete, the analysis will pause and "
                "let you continue interactively."
            )
        else:
            st.info(
                "**Analysis Running** — The agent is generating and executing analysis steps. "
                "The notebook below updates in real time as each step completes. "
                "Click **Stop Analysis** in the sidebar to pause at any safe checkpoint."
            )
        if _is_stop_pending:
            st.warning("Stop requested. Finishing current sub-step, then pausing for feedback...")
        with st.expander("📋 Output log", expanded=False):
            st.text_area("Log", value=output_text, height=200, disabled=True, label_visibility="collapsed")
        st.markdown("#### Notebook")
        num_total = st.session_state.get("run_num_analyses", 1)
        completed, current = g._parse_run_progress(output_text) if num_total > 1 else (0, 1)
        if num_total > 1:
            analyses = g._collect_notebooks_by_analysis(run_output_dir, num_total)
            tab_labels = [f"Analysis {i}" + (" ✓" if nb and i <= completed else " ●" if nb and i == current else "") for i, nb in analyses]
            tabs = st.tabs(tab_labels)
            for i, (analysis_idx, nb_path) in enumerate(analyses):
                with tabs[i]:
                    if nb_path:
                        g._render_notebook_jupyter_style(
                            nb_path, editable=False,
                            save_snapshot=True, output_dir=run_output_dir,
                        )
                    else:
                        st.info("Pending" if analysis_idx > current else "In progress...")
        else:
            notebooks = g._collect_notebooks(run_output_dir)
            if notebooks:
                for run_name, nb_path in notebooks:
                    st.markdown(f"**📓 {run_name} / {Path(nb_path).name}**")
                    g._render_notebook_jupyter_style(
                        nb_path, editable=False,
                        save_snapshot=True, output_dir=run_output_dir,
                    )
            else:
                st.info("Notebooks will appear here as they are created.")

    if show_chat and chat_col is not None:
        with chat_col:
            num_total = st.session_state.get("run_num_analyses", 1)
            g._render_chat_box(
                run_output_dir, analysis_idx=1,
                num_total=num_total,
                sel_key="chat_analysis_selection_run" if num_total > 1 else None,
            )

    proc_done = proc is not None and proc.poll() is not None
    run_pid = st.session_state.get("run_pid")
    pid_done = run_pid is not None and not g._process_alive(run_pid)
    if proc_done or pid_done:
        out_dir = Path(run_output_dir) if run_output_dir else None
        if out_dir:
            (out_dir / g._RUN_PID_FILE).unlink(missing_ok=True)
        st.session_state.run_proc = None
        st.session_state.run_pid = None
        st.session_state.run_started = False
        st.session_state.run_thread_started = False
        if proc_done and proc.returncode != 0:
            # Persist error so the completed view can show it
            error_msg = f"Process exited with code {proc.returncode}"
            st.session_state["run_error"] = error_msg
            if out_dir:
                try:
                    (out_dir / g._RUN_ERROR_FILE).write_text(error_msg, encoding="utf-8")
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
        st.session_state.pop("_is_resuming", None)  # restore phase complete
        # Play a ding sound once per new pause to alert the user (if enabled in run config)
        _ding_enabled = False
        if run_output_dir:
            _cfg_path = Path(run_output_dir) / g._RUN_CONFIG_FILE
            if _cfg_path.exists():
                try:
                    _ding_enabled = bool(json.loads(_cfg_path.read_text(encoding="utf-8")).get("ding_on_pause", False))
                except Exception:
                    pass
        if _ding_enabled and pause_id != st.session_state.get("_last_ding_pause_id"):
            st.session_state["_last_ding_pause_id"] = pause_id
            st.components.v1.html("""
<script>
(function () {
  try {
    var p = window.parent || window;
    var ACtx = p.AudioContext || p.webkitAudioContext;
    if (!ACtx) return;
    var ctx = new ACtx();
    // Two-tone ding: fundamental + harmonic
    [880, 1108].forEach(function (freq, i) {
      var osc  = ctx.createOscillator();
      var gain = ctx.createGain();
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.type = 'sine';
      osc.frequency.value = freq;
      var t0 = ctx.currentTime + i * 0.06;
      gain.gain.setValueAtTime(i === 0 ? 0.35 : 0.20, t0);
      gain.gain.exponentialRampToValueAtTime(0.0001, t0 + 1.2);
      osc.start(t0);
      osc.stop(t0 + 1.2);
    });
  } catch (e) {}
})();
</script>
""", height=0)
        result = None
        nb_path = ""
        try:
            nb_path = request_path.read_text(encoding="utf-8").strip()
        except Exception:
            pass
        req_mtime = f"{request_path.stat().st_mtime}" if request_path and request_path.exists() else "0"
        pause_id = f"paused_{req_mtime}"
        edit_mode = st.session_state.get(f"pause_edit_mode_{pause_id}", False)
        show_chat = g._should_show_chat()
        main_col, _gap, chat_col = st.columns([0.831, 0.02, 0.236])
        num_total = st.session_state.get("run_num_analyses", 1)
        paused_analysis_idx = None
        if nb_path:
            import re
            m = re.search(r"_analysis_(\d+)\.ipynb", Path(nb_path).name, re.I)
            if m:
                paused_analysis_idx = int(m.group(1))

        with main_col:
            if _is_between_pause:
                st.markdown("### ✅ Analysis complete — review before continuing")
            else:
                st.markdown("### ⏸ Agent waiting for feedback")
            if num_total > 1 and paused_analysis_idx is not None:
                analyses = g._collect_notebooks_by_analysis(st.session_state.run_output_dir, num_total)
                tab_labels = [f"Analysis {i}" + (" ⏸" if i == paused_analysis_idx else "") for i, _ in analyses]
                tabs = st.tabs(tab_labels)
                # Always auto-select the paused analysis tab on every render (Streamlit resets tabs to 0 on each rerun)
                if paused_analysis_idx > 1:
                    _target = paused_analysis_idx - 1  # 0-based
                    st.components.v1.html(f"""<script>
(function() {{
  var idx = {_target};
  var p = window.parent || window;
  function clickTab() {{
    var tabs = p.document.querySelectorAll('[data-testid="stTabs"] button[role="tab"]');
    if (tabs.length > idx) {{ tabs[idx].click(); return true; }}
    return false;
  }}
  var tries = 0;
  function attempt() {{ if (tries++ > 30) return; if (!clickTab()) setTimeout(attempt, 100); }}
  attempt();
}})();
</script>""", height=0)
                for i, (aidx, _) in enumerate(analyses):
                    with tabs[i]:
                        if aidx == paused_analysis_idx and nb_path and Path(nb_path).exists():
                            import nbformat as nbf
                            from nbformat.v4 import new_code_cell, new_markdown_cell
                            result = g._render_notebook_jupyter_style(nb_path, editable=True, pause_id=pause_id, sidebar_actions=True)
                            summary_path = Path(st.session_state.run_output_dir) / g._AGENT_SUMMARY_FILE if st.session_state.run_output_dir else None
                            if summary_path and summary_path.exists():
                                try:
                                    summary_text = summary_path.read_text(encoding="utf-8").strip()
                                    if summary_text:
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
                result = g._render_notebook_jupyter_style(nb_path, editable=True, pause_id=pause_id, sidebar_actions=True)
                summary_path = Path(st.session_state.run_output_dir) / g._AGENT_SUMMARY_FILE if st.session_state.run_output_dir else None
                if summary_path and summary_path.exists():
                    try:
                        summary_text = summary_path.read_text(encoding="utf-8").strip()
                        if summary_text:
                            st.markdown("#### 🤖 What the agent has done so far")
                            escaped = html.escape(summary_text).replace("\n", "<br>")
                            st.markdown(
                                f'<div class="agent-summary-box"><div class="agent-summary-content">{escaped}</div></div>',
                                unsafe_allow_html=True,
                            )
                    except Exception:
                        pass
        if chat_col is not None:
            with chat_col:
                if show_chat:
                    num_total = st.session_state.get("run_num_analyses", 1)
                    g._render_chat_box(
                        st.session_state.run_output_dir, analysis_idx=1, floating=True,
                        num_total=num_total,
                        sel_key="chat_analysis_selection_pause" if num_total > 1 else None,
                    )
        if nb_path and Path(nb_path).exists() and result:
            out = result
            cell_sources, feedback, nb, run_clicked, insert_after_idx, insert_type, continue_clicked, finish_clicked, edit_clicked = out[:9]
            save_clicked = out[9] if len(out) > 9 else False
            exec_path = g._pause_execute_path()
            run_monitor = st.session_state.get("_cell_run_monitor")
            if run_monitor and run_monitor.get("nb_path") != nb_path:
                st.session_state.pop("_cell_run_monitor", None)
                st.rerun()
            if run_monitor and run_monitor.get("nb_path") == nb_path:
                elapsed = time.monotonic() - float(run_monitor.get("started_monotonic", time.monotonic()))
                req_consumed = (exec_path is None) or (not exec_path.exists())
                if req_consumed:
                    st.session_state["_cell_run_success"] = {
                        "nb_path": nb_path,
                        "pause_id": run_monitor.get("pause_id"),
                        "cell_index": run_monitor.get("cell_index"),
                        "completed_at": time.time(),
                    }
                    st.session_state.pop("_cell_run_monitor", None)
                    st.rerun()
                elif elapsed > 120:
                    st.session_state.pop("_cell_run_monitor", None)
                    st.warning("Code cell is taking longer than expected. You can wait or try running again.")
                else:
                    time.sleep(0.1)
                    st.rerun()
            notebook_dirty = False
            for i, src in enumerate(cell_sources):
                if nb.cells[i].source != src:
                    nb.cells[i].source = src
                    notebook_dirty = True
            if notebook_dirty:
                with open(nb_path, "w", encoding="utf-8") as f:
                    nbf.write(nb, f)
            if run_clicked is not None:
                if exec_path:
                    nb_mtime_before = Path(nb_path).stat().st_mtime_ns if Path(nb_path).exists() else 0
                    exec_path.unlink(missing_ok=True)  # Clear stale request from a prior click.
                    req = {"cell_index": run_clicked, "request_id": str(time.time_ns())}
                    exec_path.write_text(json.dumps(req), encoding="utf-8")
                    st.session_state["_cell_run_monitor"] = {
                        "nb_path": nb_path,
                        "pause_id": pause_id,
                        "cell_index": run_clicked,
                        "nb_mtime_before": nb_mtime_before,
                        "started_monotonic": time.monotonic(),
                    }
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
                if _is_between_pause:
                    # Extend the current analysis further with the user's feedback
                    payload = f"__CONTINUE_CURRENT__:{feedback}" if feedback else "__CONTINUE_CURRENT__"
                    response_path.write_text(payload, encoding="utf-8")
                else:
                    response_path.write_text(feedback or "", encoding="utf-8")
                request_path.unlink(missing_ok=True)
                st.session_state["_transitioning_from_pause"] = True
                st.session_state["_pending_scroll_bottom"] = True
                # Keep the running view on the same analysis tab that was paused
                if paused_analysis_idx is not None and paused_analysis_idx > 1:
                    st.session_state["_active_run_tab"] = paused_analysis_idx - 1
                else:
                    st.session_state.pop("_active_run_tab", None)
                st.rerun()
            elif st.session_state.pop("_sb_next_analysis", False):
                # Proceed to the next analysis, passing any feedback along
                _next_feedback = st.session_state.get(f"pause_feedback_{pause_id}", "") or ""
                response_path.write_text(_next_feedback, encoding="utf-8")
                request_path.unlink(missing_ok=True)
                st.session_state["_transitioning_from_pause"] = True
                st.session_state["_pending_scroll_bottom"] = True
                # Schedule a tab jump to the next analysis tab (0-based index)
                if paused_analysis_idx is not None:
                    st.session_state["_pending_tab_jump"] = paused_analysis_idx  # tab index = analysis_idx (1-based paused → next tab)
                st.rerun()
            elif finish_clicked:
                response_path.write_text("__FINISH__", encoding="utf-8")
                request_path.unlink(missing_ok=True)
                st.session_state["_transitioning_from_pause"] = True
                st.rerun()
            elif st.session_state.pop("_cell_inline_edit_toggled", False):
                st.rerun()
        else:
            st.warning("Notebook path not found. The agent may have advanced.")
            if st.button("Continue (send empty feedback)"):
                response_path.write_text("", encoding="utf-8")
                request_path.unlink(missing_ok=True)
                st.rerun()
    else:
        # Update active run tab if a transition just happened
        _pending_tab = st.session_state.pop("_pending_tab_jump", None)
        if _pending_tab is not None:
            st.session_state["_active_run_tab"] = _pending_tab
        # Always re-inject JS on every full rerun while running, so tab survives reruns
        _active_run_tab = st.session_state.get("_active_run_tab")
        if _active_run_tab:
            st.components.v1.html(f"""<script>
(function() {{
  var idx = {_active_run_tab};
  var p = window.parent || window;
  function clickTab() {{
    var tabs = p.document.querySelectorAll('[data-testid="stTabs"] button[role="tab"]');
    if (tabs.length > idx) {{ tabs[idx].click(); return true; }}
    return false;
  }}
  var tries = 0;
  function attempt() {{ if (tries++ > 30) return; if (!clickTab()) setTimeout(attempt, 100); }}
  attempt();
}})();
</script>""", height=0)
        if st.session_state.pop("_pending_scroll_bottom", False):
            st.components.v1.html("""<script>
(function() {
  var p = window.parent || window;
  function scrollToBottom() {
    p.document.documentElement.scrollTop = p.document.body.scrollHeight;
    p.window.scrollTo({ top: p.document.body.scrollHeight, behavior: 'smooth' });
  }
  [100, 600, 1400, 2500].forEach(function(t) { setTimeout(scrollToBottom, t); });
})();
</script>""", height=0)
        _running_view()

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
        pause_id = "killed"
        edit_mode = st.session_state.get(f"pause_edit_mode_{pause_id}", False)
        show_chat = g._should_show_chat()
        if show_chat or edit_mode:
            main_col, _gap, chat_col = st.columns([0.831, 0.02, 0.236])
        else:
            main_col = st.container()
            chat_col = None
        with main_col:
            st.markdown("---")
            st.markdown("### ⏹ Analysis stopped")
            st.info(
                "**Analysis Paused** — Review and edit the notebook below.\n\n"
                "- Click **Edit** to modify notebook cells\n"
                "- Click **Continue** to resume the analysis from where it left off\n"
                "- Click **Finish** to end the analysis and return home"
            )
            import nbformat as nbf
            from nbformat.v4 import new_code_cell, new_markdown_cell
            with st.form("killed_interactive_form"):
                st.caption("Edit cells, add new ones. Click Continue to resume the analysis, or Finish to go home.")
                result = g._render_notebook_jupyter_style(nb_path, editable=True, pause_id=pause_id)
            if result:
                cell_sources, feedback, nb, run_clicked, insert_after_idx, insert_type, continue_clicked, finish_clicked, edit_clicked = result[:9]
                notebook_dirty = False
                for i, src in enumerate(cell_sources):
                    if nb.cells[i].source != src:
                        nb.cells[i].source = src
                        notebook_dirty = True
                if notebook_dirty:
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
                elif st.session_state.pop("_cell_inline_edit_toggled", False):
                    st.rerun()
        if chat_col is not None:
            with chat_col:
                if show_chat:
                    analyses = g._collect_notebooks_by_analysis(st.session_state.run_output_dir, st.session_state.get("run_num_analyses", 1))
                    num_total_killed = len(analyses)
                    g._render_chat_box(
                        st.session_state.run_output_dir, analysis_idx=1,
                        num_total=num_total_killed,
                        sel_key="chat_analysis_selection_killed" if num_total_killed > 1 else None,
                    )
    else:
        st.warning("No notebook found. Click Finish to return home.")
        if st.button("Finish Analysis", type="primary"):
            st.session_state.run_output_dir = None
            st.session_state.run_show_interactive = False
            if request_path and request_path.exists():
                request_path.unlink(missing_ok=True)
            st.switch_page("gui.py")

# Notebook viewer (when analysis completed)
elif st.session_state.run_output_dir and not st.session_state.run_started:
    # Show restoring message while resume process is starting up
    _resume_restoring_main = st.session_state.get("_resume_restoring")
    if _resume_restoring_main and _resume_restoring_main[0] == st.session_state.run_output_dir:
        _restoring_aidx = _resume_restoring_main[1]
        st.info(f"**Restoring Analysis {_restoring_aidx}** (re-running the notebook)...")
        st.stop()

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
        main_col, _gap, chat_col = st.columns([0.831, 0.02, 0.236])
    else:
        main_col = st.container()
        chat_col = None

    with main_col:
        st.markdown("---")
        st.markdown("### 📓 Notebooks")
        st.caption("From your last run.")

        _run_error = st.session_state.get("run_error")
        # Also try reading from file if session state was lost
        if not _run_error and st.session_state.run_output_dir:
            _err_file = Path(st.session_state.run_output_dir) / g._RUN_ERROR_FILE
            if _err_file.exists():
                try:
                    _run_error = _err_file.read_text(encoding="utf-8").strip()
                except Exception:
                    pass

        if _run_error:
            st.error(f"**Analysis crashed** — {_run_error}")
            _log_text = g._get_run_log()
            if _log_text.strip():
                with st.expander("📋 Full output log", expanded=True):
                    st.code(_log_text, language=None)
        else:
            st.info(
                "**Analysis Complete** — Your results are ready below. "
                "Use **Continue Analysis** in the sidebar to extend any analysis further, "
                "or **Finish Analysis** to return home."
            )

        num_total = st.session_state.get("run_num_analyses", 1)

        if num_total > 1:
            analyses = g._collect_notebooks_by_analysis(st.session_state.run_output_dir, num_total)
            tab_labels = [f"Analysis {i}" for i, _ in analyses]
            tabs = st.tabs(tab_labels)
            for i, (analysis_idx, nb_path) in enumerate(analyses):
                with tabs[i]:
                    if nb_path:
                        g._render_notebook_jupyter_style(nb_path, editable=False, save_snapshot=True, output_dir=st.session_state.run_output_dir)
                    else:
                        st.info("No notebook for this analysis.")
        else:
            analyses_single = g._collect_notebooks_by_analysis(st.session_state.run_output_dir, 1)
            if analyses_single and analyses_single[0][1]:
                _, nb_path = analyses_single[0]
                g._render_notebook_jupyter_style(nb_path, editable=False, save_snapshot=True, output_dir=st.session_state.run_output_dir)
            else:
                st.info("Run an analysis to see notebooks here.")

    if show_chat and chat_col is not None and st.session_state.run_output_dir:
        with chat_col:
            num_total = st.session_state.get("run_num_analyses", 1)
            g._render_chat_box(
                st.session_state.run_output_dir, analysis_idx=1,
                num_total=num_total,
                sel_key="chat_analysis_selection" if num_total > 1 else None,
            )

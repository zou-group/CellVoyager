"""
CellVoyager GUI — Upload dataset, enter paper summary, run analysis, view notebooks.
Run with: streamlit run gui.py
"""
import base64
import html
import json
import datetime
import re
import uuid
import os
import subprocess
import sys
import threading
import time
from io import BytesIO
from pathlib import Path

import streamlit as st

# Paths
ROOT = Path(__file__).resolve().parent
UPLOADS_DIR = ROOT / "gui_uploads"
OUTPUTS_BASE = ROOT / "outputs"
_LAST_RUN_FILE = OUTPUTS_BASE / ".last_run"

# Session state for live run
if "run_proc" not in st.session_state:
    st.session_state.run_proc = None
if "run_output" not in st.session_state:
    st.session_state.run_output = []
if "run_cmd" not in st.session_state:
    st.session_state.run_cmd = None
if "run_started" not in st.session_state:
    st.session_state.run_started = False
if "run_thread_started" not in st.session_state:
    st.session_state.run_thread_started = False
if "run_output_dir" not in st.session_state:
    st.session_state.run_output_dir = None
if "run_interactive_mode" not in st.session_state:
    st.session_state.run_interactive_mode = False
if "run_pid" not in st.session_state:
    st.session_state.run_pid = None

_PAUSE_REQUEST_FILE = ".cellvoyager_pause_request"
_PAUSE_RESPONSE_FILE = ".cellvoyager_pause_response"
_EXECUTE_REQUEST_FILE = ".cellvoyager_execute_request"
_AGENT_SUMMARY_FILE = ".cellvoyager_agent_summary"
_CHAT_REQUEST_FILE = ".cellvoyager_chat_request"
_CHAT_RESPONSE_FILE = ".cellvoyager_chat_response"
_RUN_PID_FILE = ".run_pid"
_RUN_LOG_FILE = ".run_log"
_RUN_INTERACTIVE_FILE = ".run_interactive"


def _process_alive(pid: int) -> bool:
    """Check if process with given PID is still running."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _get_run_log() -> str:
    """Get run output log: from memory if live, else from .run_log file."""
    if st.session_state.run_proc is not None and st.session_state.run_output:
        return "".join(st.session_state.run_output)
    out_dir = st.session_state.run_output_dir
    if not out_dir:
        return ""
    log_path = Path(out_dir) / _RUN_LOG_FILE
    if log_path.exists():
        try:
            return log_path.read_text(encoding="utf-8")
        except Exception:
            pass
    return ""


def _has_live_run() -> bool:
    """True if a run is in progress (live proc or recovered PID)."""
    proc = st.session_state.run_proc
    if proc is not None and proc.poll() is None:
        return True
    pid = st.session_state.get("run_pid")
    if pid is not None and _process_alive(pid):
        return True
    return False


def _restore_last_run():
    """Restore last run from disk when session is fresh (e.g. after page reload)."""
    if st.session_state.run_output_dir:
        return
    if not _LAST_RUN_FILE.exists():
        return
    try:
        saved = _LAST_RUN_FILE.read_text(encoding="utf-8").strip()
        if not saved or not Path(saved).exists():
            return
        st.session_state.run_output_dir = saved
        out_dir = Path(saved)
        pid_file = out_dir / _RUN_PID_FILE
        if pid_file.exists():
            try:
                pid = int(pid_file.read_text().strip())
                if _process_alive(pid):
                    st.session_state.run_started = True
                    st.session_state.run_pid = pid
                    inter_file = out_dir / _RUN_INTERACTIVE_FILE
                    st.session_state.run_interactive_mode = (
                        inter_file.exists() and inter_file.read_text().strip() == "1"
                    )
            except (ValueError, OSError):
                pid_file.unlink(missing_ok=True)
    except Exception:
        pass


_restore_last_run()


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

LOGO_PATH = ROOT / "images" / "symbol.jpeg"
st.set_page_config(
    page_title="CellVoyager",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else "📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Pin tab title to avoid flickering between "CellVoyager" and "Streamlit" on reruns
if hasattr(st, "html"):
    st.html('<div style="display:none"><script>document.title="CellVoyager";</script></div>', unsafe_allow_javascript=True)
else:
    import streamlit.components.v1 as components
    components.html('<script>try{window.parent.document.title="CellVoyager"}catch(e){}</script>', height=0)

# Inject custom styles for polish
st.markdown("""
<style>
    /* Hero and typography */
    h1 { font-weight: 600 !important; letter-spacing: -0.02em !important; }
    .stMarkdown p { color: #495057 !important; }
    /* Tighter section spacing */
    div[data-testid="stVerticalBlock"] > div { gap: 0.5rem !important; }
    /* Card-style containers for notebook cells */
    div[data-testid="stExpander"] { border-radius: 10px !important; box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important; }
    /* Info/success boxes */
    div[data-testid="stAlert"] { border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)

# Hero header
if LOGO_PATH.exists():
    col_logo, col_title = st.columns([0.15, 1])
    with col_logo:
        st.image(str(LOGO_PATH), width='stretch')
    with col_title:
        st.title("CellVoyager")
        st.caption("Single-cell transcriptomics analysis with AI — hypothesis generation, live notebooks, interactive feedback")
else:
    st.title("CellVoyager")
    st.caption("Single-cell transcriptomics analysis with AI — hypothesis generation, live notebooks, interactive feedback")

st.divider()

# Ensure uploads dir exists
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# Sidebar: inputs
with st.sidebar:
    st.markdown("### 📁 Data & context")
    h5ad_file = st.file_uploader(
        "Dataset (.h5ad)",
        type=["h5ad"],
        help="Single-cell dataset in AnnData format",
    )
    paper_source = st.radio("Paper summary source", ["Type or paste", "Upload file"], horizontal=True)
    if paper_source == "Upload file":
        paper_file = st.file_uploader("Paper file (.txt, .md)", type=["txt", "md"], help="Summary or abstract")
        paper_text = paper_file.read().decode() if paper_file else ""
    else:
        paper_file = None
        paper_text = ""

    st.divider()
    st.markdown("### ⚙️ Settings")

    analysis_name = st.text_input("Analysis name", value="covid19", help="Output folders and logs")
    num_analyses = st.number_input("Analyses", min_value=1, max_value=20, value=1, help="Plans per run")
    max_iterations = st.number_input("Max iterations", min_value=1, max_value=50, value=8)
    execution_mode = st.selectbox("Execution", ["claude", "legacy"], help="claude = Agent + Jupyter")
    interactive_mode = st.checkbox("Interactive mode", value=False, help="Pause for feedback and edits (claude)")
    intervene_every = st.number_input(
        "Intervene every N steps",
        min_value=1,
        max_value=20,
        value=1,
        disabled=not interactive_mode,
        help="Show edit screen every N interpretation steps when interactive (1 = after each step)",
    )
    use_deepresearch = st.checkbox("DeepResearch", value=True, help="Paper-based background")
    model_name = st.text_input("Model", value="o3-mini")

    st.divider()
    api_keys_ok = True
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY not set")
        api_keys_ok = False
    if execution_mode == "claude" and not os.getenv("ANTHROPIC_API_KEY"):
        st.error("ANTHROPIC_API_KEY not set (needed for claude)")
        api_keys_ok = False

    run_clicked = st.button("▶ Run analysis", type="primary", use_container_width=True)

    # Run selector: pick which run's notebooks to view (persists across reloads)
    if not st.session_state.run_started:
        st.divider()
        st.markdown("### 📂 View run")
        available_runs = []
        if OUTPUTS_BASE.exists():
            for d in sorted(OUTPUTS_BASE.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
                if d.is_dir() and not d.name.startswith(".") and list(d.glob("*.ipynb")):
                    available_runs.append((d.name, str(d)))
        if available_runs:
            all_paths = [None] + [path for _, path in available_runs]
            current = st.session_state.run_output_dir
            idx = all_paths.index(current) if current in all_paths else 0
            choice = st.selectbox(
                "Show notebooks from",
                all_paths,
                format_func=lambda x: "(none)" if x is None else Path(x).name,
                index=idx,
                key="run_selector",
            )
            if choice != st.session_state.run_output_dir:
                st.session_state.run_output_dir = choice
                st.rerun()
        else:
            st.caption("No completed runs yet. Run an analysis to see notebooks here.")

# Main area: paper text (when typing)
if paper_source == "Type or paste":
    st.markdown("#### 📄 Paper summary")
    paper_text = st.text_area(
        "Paste the paper summary or biological context below",
        value=paper_text,
        height=200,
        placeholder="Paste paper abstract, methods, or key findings...",
        label_visibility="collapsed",
    )

# Run analysis: start on click, then poll when running
def _collect_notebooks(output_dir_filter=None):
    """Collect notebooks. Only from output_dir_filter if set; otherwise none."""
    if output_dir_filter:
        d = Path(output_dir_filter)
        if d.exists() and d.is_dir():
            return [(d.name, str(nb)) for nb in d.glob("*.ipynb")]
    return []


def _cell_source_str(cell):
    """Get cell source as a single string (nbformat can use list or str)."""
    src = cell.source
    return "\n".join(src) if isinstance(src, list) else (src or "")


def _is_step_summary(cell):
    """True if markdown cell is a step summary (e.g. ## Step 1 summary).
    Step headers are placed ONLY before these cells."""
    if cell.cell_type != "markdown":
        return False
    src = _cell_source_str(cell).strip()
    first_line = src.split("\n")[0].strip() if src else ""
    return "summary" in first_line.lower()


def _step_separator(step_label):
    """Two bars with centered bold STEP [NUM] text in between."""
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


def _render_cell_outputs(cell):
    """Render notebook cell outputs (stream, display_data, execute_result, error)."""
    if cell.cell_type != "code":
        return
    outputs = getattr(cell, "outputs", []) or []
    for out in outputs:
        ot = out.get("output_type", "")
        if ot == "stream":
            text = out.get("text", "")
            if isinstance(text, list):
                text = "".join(text)
            if text.strip():
                st.text(text)
        elif ot == "execute_result":
            data = out.get("data", {})
            if "text/plain" in data:
                plain = data["text/plain"]
                if isinstance(plain, list):
                    plain = "".join(plain)
                st.code(plain, language=None)
            if "image/png" in data:
                try:
                    b64 = data["image/png"]
                    if isinstance(b64, list):
                        b64 = "".join(b64)
                    img_bytes = base64.b64decode(b64)
                    st.image(BytesIO(img_bytes))
                except Exception:
                    st.caption("[Image output]")
        elif ot == "display_data":
            data = out.get("data", {})
            if "image/png" in data:
                try:
                    b64 = data["image/png"]
                    if isinstance(b64, list):
                        b64 = "".join(b64)
                    img_bytes = base64.b64decode(b64)
                    st.image(BytesIO(img_bytes))
                except Exception:
                    st.caption("[Image output]")
            elif "text/plain" in data:
                plain = data["text/plain"]
                if isinstance(plain, list):
                    plain = "".join(plain)
                st.text(plain)
        elif ot == "error":
            tb = out.get("traceback", [])
            if isinstance(tb, list):
                tb = "\n".join(tb)
            st.error(tb)


def _render_cell_display(cell):
    """Render a notebook cell Jupyter-style: markdown or code + outputs (read-only)."""
    src = _cell_source_str(cell)
    is_md = cell.cell_type == "markdown"
    if is_md:
        st.markdown(src)
    else:
        st.code(src, language="python")
        _render_cell_outputs(cell)


def _render_editable_cell(cell, cell_idx, pause_id, expand_edit=False):
    """
    Show rendered content with an Edit expander below. Expand to edit the cell source.
    expand_edit=True opens the Edit expander (e.g. for newly inserted cells).
    Returns the text_area/code_editor value for form collection.
    """
    src = _cell_source_str(cell)
    height = min(300, max(80, 60 + len(src.splitlines()) * 18))
    if cell.cell_type == "markdown":
        st.markdown(src)
        with st.expander("Edit", expanded=expand_edit):
            editor = st.text_area(
                "Source",
                value=src,
                height=height,
                key=f"pause_cell_{pause_id}_{cell_idx}",
                label_visibility="collapsed",
            )
    else:
        st.code(src, language="python")
        _render_cell_outputs(cell)
        with st.expander("Edit", expanded=expand_edit):
            editor = st.text_area(
                "Source",
                value=src,
                height=height,
                key=f"pause_cell_{pause_id}_{cell_idx}",
                label_visibility="collapsed",
            )
    return editor


def _render_notebook_jupyter_style(nb_path, editable=False, pause_id=None):
    """
    Render a notebook cell-by-cell Jupyter-style (markdown, code, outputs).
    If editable=True, add edit expanders, Run buttons, Insert/Add cell buttons.
    Returns (cell_sources, feedback, nb, run_cell_idx, insert_after_idx, insert_type, continue_clicked).
    insert_after_idx: index after which to insert (0=after cell 0, len-1=after last); -1 = add at end.
    insert_type: "code" or "markdown" or None
    """
    import nbformat as nbf
    with open(nb_path, encoding="utf-8") as f:
        nb = nbf.read(f, as_version=4)
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
            if editable and pause_id is not None:
                expand_edit = i == st.session_state.get("open_edit_cell")
                if expand_edit:
                    st.session_state.pop("open_edit_cell", None)
                edited = _render_editable_cell(cell, i, pause_id, expand_edit=expand_edit)
                cell_sources.append(edited)
            else:
                _render_cell_display(cell)
            if editable:
                if cell.cell_type == "code":
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
    add_code, add_md, continue_clicked = False, False, False
    if editable:
        st.markdown(
            '<div class="feedback-box-header">💬 <strong>Feedback for the agent</strong></div>',
            unsafe_allow_html=True,
        )
        feedback = st.text_area(
            "Feedback",
            placeholder="e.g., focus more on cluster 3, or skip the next visualization...",
            height=100,
            key=f"pause_feedback_{pause_id}",
            label_visibility="collapsed",
        )
        st.caption("Insert cells after a specific cell above, or add at end. Run code cells, then Continue.")
        btn_cols = st.columns(4)
        with btn_cols[0]:
            add_code = st.form_submit_button("+ Code cell (at end)")
        with btn_cols[1]:
            add_md = st.form_submit_button("+ Markdown (at end)")
        with btn_cols[2]:
            continue_clicked = st.form_submit_button("Continue")
        if add_code and insert_type is None:
            insert_after_idx = len(nb.cells) - 1
            insert_type = "code"
        elif add_md and insert_type is None:
            insert_after_idx = len(nb.cells) - 1
            insert_type = "markdown"
        return cell_sources, feedback, nb, run_clicked, insert_after_idx, insert_type, continue_clicked
    return None


def _render_notebook(nb_path):
    try:
        from nbconvert import HTMLExporter
        import nbformat

        with open(nb_path, encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        exporter = HTMLExporter()
        exporter.template_name = "classic"
        body, _ = exporter.from_notebook_node(nb)
        # Scroll to bottom on load so new cells stay in view when page refreshes
        scroll_script = """
        <script>
        (function() {
            var scrollToBottom = function() {
                window.scrollTo(0, document.body.scrollHeight);
            };
            if (document.readyState === 'loading') {
                document.addEventListener('DOMContentLoaded', scrollToBottom);
            } else {
                scrollToBottom();
            }
        })();
        </script>
        """
        body = body + scroll_script
        st.components.v1.html(body, height=600, scrolling=True)
    except Exception as e:
        st.error(f"Could not render notebook: {e}")
        st.code(f"Open in Jupyter: {nb_path}")


if run_clicked and not st.session_state.run_started:
    if not api_keys_ok:
        st.error("Set API keys in your environment and restart.")
    elif not h5ad_file and not (ROOT / "example" / "covid19.h5ad").exists():
        st.error("Upload an .h5ad file or ensure example/covid19.h5ad exists.")
    elif not paper_text.strip() and not paper_file:
        st.error("Provide paper summary: type it above or upload a file.")
    else:
        if h5ad_file:
            h5ad_path = UPLOADS_DIR / h5ad_file.name
            with open(h5ad_path, "wb") as f:
                f.write(h5ad_file.getvalue())
        else:
            h5ad_path = ROOT / "example" / "covid19.h5ad"

        paper_path = UPLOADS_DIR / f"{analysis_name}_paper.txt"
        paper_path.write_text(paper_text, encoding="utf-8")

        run_output_dir = OUTPUTS_BASE / f"{analysis_name}_gui_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_output_dir.mkdir(parents=True, exist_ok=True)
        st.session_state.run_output_dir = str(run_output_dir)
        OUTPUTS_BASE.mkdir(parents=True, exist_ok=True)
        _LAST_RUN_FILE.write_text(st.session_state.run_output_dir, encoding="utf-8")
        (run_output_dir / _RUN_INTERACTIVE_FILE).write_text("1" if interactive_mode else "0", encoding="utf-8")

        cmd = [
            sys.executable,
            str(ROOT / "run_v2.py"),
            "--h5ad-path", str(h5ad_path),
            "--paper-path", str(paper_path),
            "--analysis-name", analysis_name,
            "--num-analyses", str(num_analyses),
            "--max-iterations", str(int(max_iterations)),
            "--execution-mode", execution_mode,
            "--model-name", model_name,
            "--output-home", str(ROOT),
            "--log-home", str(ROOT / "logs"),
            "--output-dir", st.session_state.run_output_dir,
        ]
        if interactive_mode:
            cmd.append("--interactive")
            cmd.extend(["--intervene-every", str(int(intervene_every))])
        if not use_deepresearch:
            cmd.append("--no-deepresearch")

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        if interactive_mode and execution_mode == "claude":
            env["CELLVOYAGER_GUI_INTERACTIVE"] = "1"
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        (run_output_dir / _RUN_PID_FILE).write_text(str(proc.pid), encoding="utf-8")
        st.session_state.run_proc = proc
        st.session_state.run_output = []
        st.session_state.run_cmd = cmd
        st.session_state.run_started = True
        st.session_state.run_interactive_mode = interactive_mode
        log_path = run_output_dir / _RUN_LOG_FILE
        t = threading.Thread(target=_read_output, args=(proc, st.session_state.run_output, log_path))
        t.daemon = True
        t.start()
        st.session_state.run_thread_started = True
        st.rerun()

def _pause_request_path():
    if not st.session_state.run_output_dir:
        return None
    return Path(st.session_state.run_output_dir) / _PAUSE_REQUEST_FILE

def _pause_response_path():
    if not st.session_state.run_output_dir:
        return None
    return Path(st.session_state.run_output_dir) / _PAUSE_RESPONSE_FILE

def _pause_execute_path():
    if not st.session_state.run_output_dir:
        return None
    return Path(st.session_state.run_output_dir) / _EXECUTE_REQUEST_FILE


def _chat_request_path():
    if not st.session_state.run_output_dir:
        return None
    return Path(st.session_state.run_output_dir) / _CHAT_REQUEST_FILE


def _chat_response_path():
    if not st.session_state.run_output_dir:
        return None
    return Path(st.session_state.run_output_dir) / _CHAT_RESPONSE_FILE


def _chat_via_api(messages: list, output_dir: str) -> str | None:
    """Call the model directly with notebook context. Prefers Claude when available, falls back to OpenAI."""
    try:
        import nbformat as nbf
    except ImportError:
        return None
    out = Path(output_dir)
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

    # Try Claude first (when Anthropic key exists), then OpenAI
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=anthropic_key)
            api_messages = [{"role": m["role"], "content": m["content"]} for m in messages]
            resp = client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=1024,
                system=system,
                messages=api_messages,
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
                model="gpt-4o-mini",
                messages=api_messages,
                max_tokens=1024,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception:
            pass
    return None


def _render_chat_box(output_dir: str | None):
    """Render the chat-with-agent box on the side. Messages stored in session state."""
    if not output_dir:
        return
    key = f"agent_chat_{output_dir}"
    pending_key = f"agent_chat_pending_{output_dir}"
    pending_reply_key = f"agent_chat_pending_reply_{output_dir}"
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

    # Handle "waiting for reply" BEFORE chat_input so spinner appears in message flow, not below input
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
                    reply = _chat_via_api(messages, output_dir)
                messages.append({
                    "role": "assistant",
                    "content": reply or "*(Could not get a response. Your message was saved.)*",
                })
        st.rerun()

    # Chat input at end — stays visible
    prompt = st.chat_input("Ask about the results...")
    if prompt:
        st.session_state[pending_key] = prompt
        st.rerun()


def _should_show_chat():
    """Chat appears during interactive pause, when run completes, or when viewing last run's notebooks."""
    if st.session_state.run_started and _has_live_run():
        proc = st.session_state.run_proc
        in_pause = (
            st.session_state.run_interactive_mode
            and _pause_request_path()
            and _pause_request_path().exists()
        )
        if proc is not None:
            return in_pause or proc.poll() is not None
        return in_pause or True  # recovered run: show chat
    return bool(st.session_state.run_output_dir)


if st.session_state.run_started and _has_live_run():
    proc = st.session_state.run_proc
    output_text = _get_run_log()
    request_path = _pause_request_path()
    response_path = _pause_response_path()
    in_pause_ui = (
        st.session_state.run_interactive_mode
        and request_path
        and response_path
        and request_path.exists()
    )

    if in_pause_ui:
        show_chat = _should_show_chat()
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
                    result = _render_notebook_jupyter_style(nb_path, editable=True, pause_id=pause_id)
                # Agent summary at bottom (not part of notebook)
                summary_path = Path(st.session_state.run_output_dir) / _AGENT_SUMMARY_FILE if st.session_state.run_output_dir else None
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
                _render_chat_box(st.session_state.run_output_dir)
        if nb_path and Path(nb_path).exists() and result:
                cell_sources, feedback, nb, run_clicked, insert_after_idx, insert_type, continue_clicked = result
                # Apply form edits to notebook
                for i, src in enumerate(cell_sources):
                    nb.cells[i].source = src
                with open(nb_path, "w", encoding="utf-8") as f:
                    nbf.write(nb, f)
                if run_clicked is not None:
                    exec_path = _pause_execute_path()
                    if exec_path:
                        exec_path.write_text(json.dumps({"cell_index": run_clicked}), encoding="utf-8")
                        with st.spinner("Executing cell..."):
                            for _ in range(300):  # 300 * 0.05 = 15 sec max
                                time.sleep(0.05)
                                if not exec_path.exists():
                                    break
                    st.rerun()
                elif insert_type is not None:
                    new_cell = new_code_cell("") if insert_type == "code" else new_markdown_cell("")
                    new_cell["cell_type"] = insert_type
                    new_cell["id"] = f"gui_{uuid.uuid4().hex[:12]}"
                    insert_at = insert_after_idx + 1
                    nb.cells.insert(insert_at, new_cell)
                    with open(nb_path, "w", encoding="utf-8") as f:
                        nbf.write(nb, f)
                    st.session_state["open_edit_cell"] = insert_at
                    st.rerun()
                elif continue_clicked:
                    response_path.write_text(feedback or "", encoding="utf-8")
                    request_path.unlink(missing_ok=True)
                    st.rerun()
        else:
            st.warning("Notebook path not found. The agent may have advanced.")
            if st.button("Continue (send empty feedback)"):
                response_path.write_text("", encoding="utf-8")
                request_path.unlink(missing_ok=True)
                st.rerun()
    else:
        show_chat = _should_show_chat()
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
            notebooks = _collect_notebooks(st.session_state.run_output_dir)
            if notebooks:
                for run_name, nb_path in notebooks:
                    nb_label = Path(nb_path).name
                    st.markdown(f"**📓 {run_name} / {nb_label}**")
                    _render_notebook_jupyter_style(nb_path, editable=False)
            else:
                st.info("Notebooks will appear here as they are created.")

        if show_chat and chat_col is not None:
            with chat_col:
                _render_chat_box(st.session_state.run_output_dir)

        proc_done = proc is not None and proc.poll() is not None
        pid_done = (
            st.session_state.run_pid is not None
            and not _process_alive(st.session_state.run_pid)
        )
        if proc_done or pid_done:
            out_dir = Path(st.session_state.run_output_dir) if st.session_state.run_output_dir else None
            if out_dir:
                (out_dir / _RUN_PID_FILE).unlink(missing_ok=True)
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
        else:
            time.sleep(2)
            st.rerun()

# Notebook viewer (when not running)
if not st.session_state.run_started:
    show_chat = _should_show_chat()
    if show_chat:
        main_col, chat_col = st.columns([0.72, 0.28])
    else:
        main_col = st.container()
        chat_col = None

    with main_col:
        st.markdown("---")
        st.markdown("### 📓 Notebooks")
        st.caption("From your last run." if st.session_state.run_output_dir else "Run an analysis to see notebooks here.")

        notebooks = _collect_notebooks(st.session_state.run_output_dir)
        if not notebooks:
            st.info("Run an analysis to see notebooks here.")
        else:
            for run_name, nb_path in notebooks:
                nb_label = Path(nb_path).name
                with st.expander(f"📓 {run_name} / {nb_label}"):
                    _render_notebook(nb_path)
        if st.session_state.run_output_dir:
            if st.button("Finish", type="primary", help="Return to home"):
                st.session_state.run_output_dir = None
                st.rerun()

    if show_chat and chat_col is not None and st.session_state.run_output_dir:
        with chat_col:
            _render_chat_box(st.session_state.run_output_dir)

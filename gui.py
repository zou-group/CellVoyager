"""
CellVoyager GUI — Home: dataset, settings, paper summary, run analysis.
Run with: streamlit run gui.py
Analysis runs on pages/analysis.py (separate page, no home content).
"""
import datetime
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import streamlit as st

import gui_common as g

ROOT = g.ROOT
UPLOADS_DIR = g.UPLOADS_DIR
OUTPUTS_BASE = g.OUTPUTS_BASE
_LAST_RUN_FILE = g._LAST_RUN_FILE
_RUN_INTERACTIVE_FILE = g._RUN_INTERACTIVE_FILE
_RUN_LOG_FILE = g._RUN_LOG_FILE
_RUN_PID_FILE = g._RUN_PID_FILE

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

# Hero header (always shown in both modes)
if LOGO_PATH.exists():
    col_logo, col_title = st.columns([0.15, 1])
    with col_logo:
        st.image(str(LOGO_PATH), width='stretch')
    with col_title:
        st.title("CellVoyager")
else:
    st.title("CellVoyager")

st.divider()
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# If a run is active, go straight to analysis page (home content never renders)
if st.session_state.get("run_output_dir"):
    st.switch_page("pages/analysis.py")

# ========== HOME MODE ==========
_run_validation_error = None
_run_clicked = False  # Set in sidebar below

# Sidebar: home inputs only
with st.sidebar:
    st.markdown("### 📁 Data & context")
    h5ad_file = st.file_uploader(
        "Dataset (.h5ad)",
        type=["h5ad"],
        help="Single-cell dataset in AnnData format",
        key="home_h5ad_upload",
    )
    if h5ad_file:
        save_path = UPLOADS_DIR / h5ad_file.name
        with open(save_path, "wb") as f:
            f.write(h5ad_file.getvalue())
        st.session_state.home_h5ad_path = str(save_path)
    else:
        st.session_state.home_h5ad_path = None
    paper_source = st.radio("Paper summary source", ["Type or paste", "Upload file"], horizontal=True, key="home_paper_source")
    if paper_source == "Upload file":
        paper_file = st.file_uploader("Paper file (.txt, .md)", type=["txt", "md"], help="Summary or abstract", key="home_paper_upload")
        if paper_file:
            txt = paper_file.read().decode()
            fp = UPLOADS_DIR / f"{st.session_state.home_analysis_name}_paper.txt"
            fp.write_text(txt, encoding="utf-8")
            st.session_state.home_paper_file_path = str(fp)
            paper_text = txt
        else:
            st.session_state.home_paper_file_path = None
            paper_text = st.session_state.get("home_paper_text", "")
        st.session_state.home_paper_text = paper_text
    else:
        paper_file = None
        paper_text = st.session_state.get("home_paper_text", "")

    _run_clicked = st.button("▶ Run analysis", type="primary", key="run_btn_sidebar", use_container_width=True)

    st.divider()
    st.markdown("### ⚙️ Settings")

    analysis_name = st.text_input("Analysis name", key="home_analysis_name", help="Output folders and logs")
    num_analyses = st.number_input("Analyses", min_value=1, max_value=20, key="home_num_analyses", help="Plans per run")
    max_iterations = st.number_input("Max iterations", min_value=1, max_value=50, key="home_max_iterations")
    execution_mode = st.selectbox("Execution", ["claude", "legacy"], key="home_execution_mode", help="claude = Agent + Jupyter")
    interactive_mode = st.checkbox("Interactive mode", key="home_interactive_mode", help="Pause for feedback and edits (claude)")
    intervene_every = st.number_input(
        "Intervene every N steps",
        min_value=1,
        max_value=20,
        disabled=not interactive_mode,
        key="home_intervene_every",
        help="Show edit screen every N interpretation steps when interactive (1 = after each step)",
    )
    use_deepresearch = st.checkbox("DeepResearch", key="home_use_deepresearch", help="Paper-based background")
    model_name = st.text_input("Model", key="home_model_name")

    st.divider()
    api_keys_ok = True
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY not set")
        api_keys_ok = False
    if execution_mode == "claude" and not os.getenv("ANTHROPIC_API_KEY"):
        st.error("ANTHROPIC_API_KEY not set (needed for claude)")
        api_keys_ok = False

    st.caption("Click **Run analysis** to start.")

if _run_clicked and not st.session_state.get("run_started"):
    _paper = st.session_state.home_paper_text or ""
    _paper_file_path = st.session_state.get("home_paper_file_path")
    if not _paper.strip() and _paper_file_path and Path(_paper_file_path).exists():
        try:
            _paper = Path(_paper_file_path).read_text(encoding="utf-8")
        except Exception:
            pass
    _analysis_name = st.session_state.home_analysis_name
    _num_analyses = st.session_state.home_num_analyses
    _max_iterations = st.session_state.home_max_iterations
    _execution_mode = st.session_state.home_execution_mode
    _interactive_mode = st.session_state.home_interactive_mode
    _intervene_every = st.session_state.home_intervene_every
    _use_deepresearch = st.session_state.home_use_deepresearch
    _model_name = st.session_state.home_model_name
    _h5ad_path = st.session_state.get("home_h5ad_path")
    _api_ok = bool(os.getenv("OPENAI_API_KEY")) and (
        _execution_mode != "claude" or bool(os.getenv("ANTHROPIC_API_KEY"))
    )
    _has_h5ad = _h5ad_path and Path(_h5ad_path).exists()
    _has_paper = bool(_paper.strip()) or (_paper_file_path and Path(_paper_file_path).exists())
    if not _api_ok:
        _run_validation_error = "Set API keys in your environment and restart."
    elif not _has_h5ad:
        _run_validation_error = "Upload a dataset (.h5ad file) to run the analysis."
    elif not _has_paper:
        _run_validation_error = "Provide paper summary: type it above or upload a file."
    else:
        h5ad_path = Path(_h5ad_path)
        paper_path = UPLOADS_DIR / f"{_analysis_name}_paper.txt"
        paper_path.write_text(_paper, encoding="utf-8")
        run_output_dir = OUTPUTS_BASE / f"{_analysis_name}_gui_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_output_dir.mkdir(parents=True, exist_ok=True)
        st.session_state.run_output_dir = str(run_output_dir)
        st.session_state.run_num_analyses = int(_num_analyses)
        OUTPUTS_BASE.mkdir(parents=True, exist_ok=True)
        _LAST_RUN_FILE.write_text(st.session_state.run_output_dir, encoding="utf-8")
        (run_output_dir / _RUN_INTERACTIVE_FILE).write_text("1" if _interactive_mode else "0", encoding="utf-8")
        run_config = {
            "h5ad_path": str(h5ad_path),
            "paper_path": str(paper_path),
            "analysis_name": _analysis_name,
            "execution_mode": _execution_mode,
            "max_iterations": int(_max_iterations),
            "model_name": _model_name,
            "use_deepresearch": _use_deepresearch,
            "intervene_every": int(_intervene_every),
        }
        (run_output_dir / g._RUN_CONFIG_FILE).write_text(json.dumps(run_config), encoding="utf-8")
        cmd = [
            sys.executable, str(ROOT / "run_v2.py"),
            "--h5ad-path", str(h5ad_path),
            "--paper-path", str(paper_path),
            "--analysis-name", _analysis_name,
            "--num-analyses", str(_num_analyses),
            "--max-iterations", str(int(_max_iterations)),
            "--execution-mode", _execution_mode,
            "--model-name", _model_name,
            "--output-home", str(ROOT),
            "--log-home", str(ROOT / "logs"),
            "--output-dir", st.session_state.run_output_dir,
        ]
        if _execution_mode == "claude":
            # Always enable interactive plumbing for GUI-driven pause/continue.
            # If step-by-step mode is off, use a very large intervene interval so pauses
            # only happen when the user clicks Stop.
            intervene = int(_intervene_every) if _interactive_mode else 999999
            cmd.extend(["--interactive", "--intervene-every", str(intervene)])
        elif _interactive_mode:
            cmd.extend(["--interactive", "--intervene-every", str(int(_intervene_every))])
        if _use_deepresearch:
            cmd.append("--deepresearch")
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        if _execution_mode == "claude":
            env["CELLVOYAGER_GUI_INTERACTIVE"] = "1"
        proc = subprocess.Popen(
            cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=env, start_new_session=True,
        )
        (run_output_dir / _RUN_PID_FILE).write_text(str(proc.pid), encoding="utf-8")
        st.session_state.run_proc = proc
        st.session_state.run_output = []
        st.session_state.run_cmd = cmd
        st.session_state.run_started = True
        st.session_state.run_interactive_mode = (_interactive_mode or _execution_mode == "claude")
        st.session_state.run_thread_started = True
        log_path = run_output_dir / _RUN_LOG_FILE
        t = threading.Thread(target=g._read_output, args=(proc, st.session_state.run_output, log_path))
        t.daemon = True
        t.start()
        st.switch_page("pages/analysis.py")

# Main area: paper input
if st.session_state.get("home_paper_source") == "Type or paste":
    st.markdown("#### 📄 Paper summary")
    st.text_area(
        "Paste the paper summary or biological context below",
        height=200,
        placeholder="Paste paper abstract, methods, or key findings...",
        label_visibility="collapsed",
        key="home_paper_text",
    )
if _run_validation_error:
    st.error(_run_validation_error)

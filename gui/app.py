"""
CellVoyager GUI — Home: dataset, settings, paper summary, run analysis.
Run with: streamlit run gui/app.py
Analysis runs on gui/pages/analysis.py (separate page, no home content).
"""
import datetime
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

# Ensure project root is on sys.path so `gui` resolves as a package
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st

import gui.common as g

ROOT = g.ROOT
UPLOADS_DIR = g.UPLOADS_DIR
OUTPUTS_BASE = g.OUTPUTS_BASE
_LAST_RUN_FILE = g._LAST_RUN_FILE
_RUN_INTERACTIVE_FILE = g._RUN_INTERACTIVE_FILE
_RUN_LOG_FILE = g._RUN_LOG_FILE
_RUN_PID_FILE = g._RUN_PID_FILE

DEMO_MODE = os.getenv("CELLVOYAGER_DEMO_MODE", "0") == "1"
FIXED_H5AD_PATH = Path(
    os.getenv("CELLVOYAGER_FIXED_H5AD_PATH", str(ROOT / "example" / "covid19.h5ad"))
).resolve()
FIXED_PAPER_PATH = Path(
    os.getenv("CELLVOYAGER_FIXED_PAPER_PATH", str(ROOT / "example" / "covid19_summary.txt"))
).resolve()

LOGO_PATH = ROOT / "gui" / "assets" / "logo.jpeg"
ICON_PATH = ROOT / "gui" / "assets" / "symbol.jpeg"
st.set_page_config(
    page_title="CellVoyager",
    page_icon=str(ICON_PATH) if ICON_PATH.exists() else "📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

import streamlit.components.v1 as components
components.html('<script>try{window.parent.document.title="CellVoyager"}catch(e){}</script>', height=0)

# Auto-expand the sidebar if it is currently collapsed
components.html("""
<script>
(function() {
  var doc = (window.parent || window).document;
  var tries = 0;
  function expand() {
    tries++;
    if (tries > 50) return;
    // Click the collapsed-control expand button if present
    var btn = doc.querySelector('[data-testid="collapsedControl"] button')
           || doc.querySelector('[data-testid="collapsedControl"]');
    if (btn) { try { btn.click(); } catch(e) {} return; }
    setTimeout(expand, 150);
  }
  expand();
})();
</script>
""", height=0)

# Inject custom styles
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
    :root {
        --cv-bg:            #07111f;
        --cv-bg-mid:        #0b1829;
        --cv-surface:       rgba(255,255,255,0.055);
        --cv-surface-hi:    rgba(255,255,255,0.09);
        --cv-border:        rgba(255,255,255,0.08);
        --cv-border-bright: rgba(255,255,255,0.16);
        --cv-primary:       #2563eb;
        --cv-primary-dark:  #1d4ed8;
        --cv-title:         #f1f5f9;
        --cv-text:          #94a3b8;
        --cv-text-bright:   #cbd5e1;
        --cv-muted:         #475569;
        --cv-eyebrow:       #93c5fd;
    }

    /* Scale all rem-based sizes proportionally with viewport width.
       Calibrated so 1rem = 16px at ~2257px effective viewport (67% zoom on 14" MacBook). */
    html { font-size: clamp(10px, 0.709vw, 22px) !important; }
    body { font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif !important; }

    /* Hide Streamlit chrome while keeping sidebar toggle accessible */
    header[data-testid="stHeader"] {
        background: transparent !important;
        border-bottom: none !important;
    }
    #MainMenu, footer, [data-testid="stDecoration"],
    [data-testid="stDeployButton"],
    [data-testid="stStatusWidget"] { display: none !important; }
    [data-testid="stToolbar"] { visibility: hidden !important; }
    [data-testid="stToolbar"] button { visibility: visible !important; }
    [data-testid="stSidebarCollapseButton"] { display: none !important; }

    .stApp {
        background:
            radial-gradient(circle, rgba(255,255,255,0.055) 1px, transparent 1px),
            radial-gradient(ellipse 160% 90% at -10% -20%, #1e3a8a 0%, #0f2060 30%, transparent 65%),
            var(--cv-bg) !important;
        background-size: 28px 28px, 100% 100%, 100% 100% !important;
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 1.8rem;
    }

    h1 {
        color: var(--cv-title) !important;
        font-weight: 700 !important;
        letter-spacing: -0.03em !important;
        margin-bottom: 0.1rem !important;
    }

    h3, .stMarkdown h3 { font-size: 1.4rem !important; color: var(--cv-text-bright) !important; }
    h4, .stMarkdown h4 {
        font-size: 1.5rem !important;
        color: var(--cv-title) !important;
        font-weight: 650 !important;
        margin-top: 0.35rem !important;
    }

    .stMarkdown p {
        font-size: 1rem !important;
        color: var(--cv-text) !important;
        line-height: 1.65 !important;
    }

    .stCaption, .stCaption *,
    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] p,
    [data-testid="stCaptionContainer"] span,
    [data-testid="stCaptionContainer"] small,
    div[data-testid="stMarkdownContainer"] small {
        font-size: 1.25rem !important;
        color: #e2e8f0 !important;
    }

    div[data-testid="stVerticalBlock"] > div {
        gap: 0.55rem !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        width: 28rem !important;
        min-width: 28rem !important;
    }
    section[data-testid="stSidebar"] > div {
        background: var(--cv-bg-mid) !important;
        border-right: 1px solid var(--cv-border) !important;
        width: 28rem !important;
    }

    section[data-testid="stSidebar"] .stMarkdown h3 {
        font-size: 1.625rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
        color: var(--cv-eyebrow) !important;
        font-weight: 700 !important;
        margin-top: 1.4rem !important;
        margin-bottom: 0.4rem !important;
    }

    /* Field sub-labels in sidebar */
    .cv-field-label {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        color: var(--cv-eyebrow) !important;
        margin-top: 1rem !important;
        margin-bottom: 0.25rem !important;
        border-bottom: 1.5px solid var(--cv-border);
        padding-bottom: 0.2rem;
    }

    /* Sidebar widget labels, captions, and placeholders — unified color */
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] label p,
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stCaption,
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
        font-size: 1.25rem !important;
        color: #e2e8f0 !important;
    }

    /* Sidebar input text size + uniform background */
    section[data-testid="stSidebar"] .stTextInput input,
    section[data-testid="stSidebar"] .stNumberInput input,
    section[data-testid="stSidebar"] .stTextArea textarea {
        font-size: 1.33rem !important;
        background: var(--cv-surface) !important;
    }
    section[data-testid="stSidebar"] .stSelectbox > div > div {
        font-size: 1.3rem !important;
        background: var(--cv-surface) !important;
        min-height: 2.6rem !important;
        height: auto !important;
        overflow: visible !important;
        padding: 0.4rem 0.75rem !important;
    }

    /* Selectbox dropdown options */
    [data-testid="stSelectboxVirtualDropdown"] li,
    [data-testid="stSelectboxVirtualDropdown"] li span,
    ul[data-testid="stVirtualDropdown"] li,
    ul[data-testid="stVirtualDropdown"] li span,
    div[data-baseweb="popover"] li,
    div[data-baseweb="popover"] li span {
        font-size: 1.3rem !important;
    }

    /* Primary button */
    .stButton > button[kind="primary"],
    .stButton > button[kind="primaryFormSubmit"],
    .stButton button[data-testid="stBaseButton-primary"],
    button[data-testid="stBaseButton-primary"] {
        background: linear-gradient(135deg, var(--cv-primary), var(--cv-primary-dark)) !important;
        border: 0 !important;
        color: #fff !important;
        border-radius: 10px !important;
        font-size: 1.3rem !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 14px rgba(37,99,235,0.32);
        transition: transform 0.12s ease, box-shadow 0.12s ease;
    }
    .stButton > button[kind="primary"]:hover,
    button[data-testid="stBaseButton-primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 22px rgba(37,99,235,0.44);
    }

    /* Secondary / default buttons */
    .stButton > button:not([kind="primary"]) {
        background: var(--cv-surface) !important;
        border: 1px solid var(--cv-border-bright) !important;
        color: var(--cv-text-bright) !important;
        border-radius: 8px !important;
    }
    .stButton > button:not([kind="primary"]):hover {
        background: var(--cv-surface-hi) !important;
        border-color: rgba(255,255,255,0.26) !important;
    }

    /* Inputs */
    .stTextArea textarea {
        background: var(--cv-surface) !important;
        border-radius: 10px !important;
        border: 2px solid #60a5fa !important;
        color: #ffffff !important;
        font-size: 1.25rem !important;
        -webkit-box-shadow: 0 0 15px 5px #60a5fa !important;
        -moz-box-shadow: 0 0 15px 5px #60a5fa !important;
        box-shadow: 0 0 15px 5px #60a5fa !important;
    }
    section[data-testid="stSidebar"] .stTextArea textarea {
        border: 1px solid var(--cv-border-bright) !important;
        box-shadow: none !important;
    }
    .stTextArea textarea::placeholder { color: #94a3b8 !important; }
    .stTextArea textarea:focus { border-color: #60a5fa !important; }

    .stTextInput input, .stNumberInput input {
        background: var(--cv-surface) !important;
        border-radius: 8px !important;
        font-size: 1.1rem !important;
        color: #ffffff !important;
        border: 1.5px solid var(--cv-border-bright) !important;
    }
    .stTextInput input::placeholder, .stNumberInput input::placeholder { color: #94a3b8 !important; }
    .stTextInput input:focus, .stNumberInput input:focus { border-color: rgba(37,99,235,0.55) !important; box-shadow: 0 0 0 3px rgba(37,99,235,0.14) !important; }

    .stNumberInput > div {
        background: transparent !important;
        border: 1.5px solid var(--cv-border-bright) !important;
        border-radius: 8px !important;
    }
    .stNumberInput > div input {
        border: none !important;
        background: transparent !important;
    }
    .stNumberInput button {
        color: var(--cv-text) !important;
        background: transparent !important;
        transition: background-color 0.1s ease, color 0.1s ease !important;
    }
    .stNumberInput button:hover { background: rgba(255,255,255,0.08) !important; }
    .stNumberInput button:focus:not(:focus-visible) {
        background-color: transparent !important;
        color: inherit !important;
        box-shadow: none !important;
        outline: none !important;
    }

    .stSelectbox > div > div {
        background: var(--cv-surface) !important;
        border: 1.5px solid var(--cv-border-bright) !important;
        border-radius: 8px !important;
        color: #ffffff !important;
    }

    /* Checkbox / Radio */
    .stCheckbox label, .stRadio div[role="radiogroup"] label { color: var(--cv-text) !important; }

    /* Widget labels */
    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] p,
    .stTextInput label, .stTextInput label p,
    .stNumberInput label, .stNumberInput label p,
    .stSelectbox label, .stSelectbox label p,
    .stCheckbox label, .stCheckbox label p,
    .stTextArea label, .stTextArea label p,
    .stRadio label, .stRadio label p,
    .stFileUploader label, .stFileUploader label p {
        font-size: 1.25rem !important;
        color: var(--cv-text-bright) !important;
    }

    /* Expander: lock color + opacity across every state and every child */
    html body [data-testid="stExpander"] details summary,
    html body [data-testid="stExpander"] details[open] summary,
    html body [data-testid="stExpander"] details summary *,
    html body [data-testid="stExpander"] details[open] summary * {
        color: #cbd5e1 !important;
        opacity: 1 !important;
        transition: none !important;
    }
    html body [data-testid="stExpander"] details summary p,
    html body [data-testid="stExpander"] details[open] summary p {
        font-size: 1.25rem !important;
        font-weight: 600 !important;
    }

    /* File uploader */
    div[data-testid="stFileUploader"] > section {
        border: 1.5px dashed var(--cv-border-bright) !important;
        border-radius: 10px !important;
        background: var(--cv-surface) !important;
        padding: 1rem !important;
        min-height: 9rem !important;
        box-shadow: 0 0 20px rgba(96, 165, 250, 0.6), 0 0 40px rgba(96, 165, 250, 0.25) !important;
    }
    div[data-testid="stFileUploaderDropzone"] {
        border: none !important;
        background: transparent !important;
        padding: 0 !important;
        min-height: 7rem !important;
        height: 7rem !important;
    }
    div[data-testid="stFileUploaderDropzone"] > div {
        min-height: 7rem !important;
        height: 7rem !important;
        justify-content: center !important;
    }
    div[data-testid="stFileUploaderDropzoneInstructions"] {
        overflow: visible !important;
        white-space: normal !important;
    }
    div[data-testid="stFileUploaderDropzoneInstructions"] span,
    div[data-testid="stFileUploaderDropzoneInstructions"] small,
    div[data-testid="stFileUploaderDropzoneInstructions"] p {
        font-size: 14px !important;
        overflow: visible !important;
        white-space: normal !important;
        text-overflow: unset !important;
        color: var(--cv-text) !important;
    }
    div[data-testid="stFileUploaderDropzone"] button {
        font-size: 1.4rem !important;
        padding: 0.6rem 1.2rem !important;
        min-height: 3rem !important;
    }

    /* Alerts */
    div[data-testid="stAlert"] {
        border-radius: 10px !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stAlert"] p {
        font-size: 1.25rem !important;
    }

    /* Expanders */
    div[data-testid="stExpander"] {
        background: var(--cv-surface) !important;
        border: 1px solid var(--cv-border) !important;
        border-radius: 10px !important;
        overflow: hidden;
    }

    /* Tabs */
    .stTabs [data-testid="stTabBar"] { background: transparent !important; border-bottom: 1px solid var(--cv-border) !important; }
    .stTabs button[role="tab"] { color: var(--cv-text) !important; background: transparent !important; border: none !important; font-size: 2rem !important; }
    .stTabs button[role="tab"][aria-selected="true"] { color: var(--cv-title) !important; font-weight: 600 !important; background: var(--cv-surface) !important; border-bottom: 2px solid var(--cv-primary) !important; }
    .stTabs div[data-testid="stAlert"] p { font-size: 1.7rem !important; }

    /* Dividers */
    hr { border-color: var(--cv-border) !important; }

    /* Step cards */
    .cv-steps-row { display: flex; gap: 0.6rem; margin: 0.6rem 0 1.2rem; }
    .cv-step-card {
        flex: 1;
        background: var(--cv-surface);
        border: 1px solid var(--cv-border-bright);
        border-radius: 12px;
        padding: 0.85rem 0.9rem;
        backdrop-filter: blur(12px);
    }
    .cv-step-card--cta { border-color: rgba(37,99,235,0.38); background: rgba(37,99,235,0.07); }
    .cv-step-num {
        font-size: 1.5rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: var(--cv-eyebrow);
        margin-bottom: -0.2rem;
    }
    .cv-step-card--cta .cv-step-num { color: #60a5fa; }
    .cv-step-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--cv-title);
        margin-bottom: 0.15rem;
    }
    .cv-step-desc {
        font-size: 1.25rem;
        color: var(--cv-text);
        line-height: 1.5;
    }

    /* Status banner */
    .cv-status-banner {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        background: rgba(37,99,235,0.09);
        border: 1px solid rgba(37,99,235,0.24);
        border-radius: 10px;
        padding: 0.6rem 1rem;
        margin-bottom: 0.8rem;
        font-size: 1.1rem;
        color: var(--cv-text-bright);
    }
    .cv-status-file { font-weight: 600; color: var(--cv-eyebrow); }
    .cv-status-sep { color: var(--cv-muted); }
    .cv-status-state { color: #ffffff; }

    /* Section label above form fields */
    .cv-field-label {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--cv-eyebrow);
        margin-bottom: 0.3rem;
    }

    .cv-kicker {
        display: inline-block;
        background: rgba(37,99,235,0.14);
        color: var(--cv-eyebrow);
        border: 1px solid rgba(37,99,235,0.25);
        border-radius: 999px;
        padding: 0.18rem 0.65rem;
        font-size: 0.74rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        margin-bottom: 0.3rem;
    }

    /* "Additional context" muted section label */
    p.cv-optional-header {
        font-size: 1.64rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.07em !important;
        color: #e2e8f0 !important;
        margin-top: 1.1rem !important;
        margin-bottom: 0.1rem !important;
    }

    /* Uploaded file chip in sidebar */
    .cv-uploaded-chip {
        display: flex;
        align-items: center;
        gap: 0.45rem;
        background: rgba(37,99,235,0.10);
        border: 1px solid rgba(37,99,235,0.24);
        border-radius: 8px;
        padding: 0.45rem 0.7rem;
        font-size: 1.05rem;
        font-weight: 600;
        color: var(--cv-eyebrow);
        margin-bottom: 0.5rem;
        word-break: break-all;
    }
</style>
""", unsafe_allow_html=True)

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# If a run is active, go straight to analysis page (home content never renders)
if st.session_state.get("run_output_dir"):
    st.switch_page("pages/analysis.py")

# ========== HOME MODE ==========
_run_validation_error = None
_run_clicked = False  # Set in sidebar below

# Sidebar: home inputs only
with st.sidebar:
    # Logo (contains CellVoyager name)
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width="stretch")
    else:
        st.markdown("## CellVoyager")
    st.divider()

    st.markdown("### 📁 Data & context")
    st.markdown('<p class="cv-field-label">Dataset</p>', unsafe_allow_html=True)
    if DEMO_MODE:
        st.info("Demo mode enabled: using fixed COVID-19 dataset.")
        st.session_state.home_h5ad_path = str(FIXED_H5AD_PATH) if FIXED_H5AD_PATH.exists() else None
        if not st.session_state.get("home_dataset_summary"):
            st.session_state["home_dataset_summary"] = (
                "The dataset consists of Seq-Well single-cell RNA-seq profiles of peripheral blood "
                "mononuclear cells from seven hospitalized COVID-19 patients (including ARDS cases) "
                "and six healthy controls, comprising 44,721 cells across eight patient samples."
            )
    else:
        _existing_path = st.session_state.get("home_h5ad_path")
        _replacing = st.session_state.get("home_replacing_h5ad", False)
        if _existing_path and not _replacing:
            _existing_name = Path(_existing_path).name
            st.markdown(
                f'<div class="cv-uploaded-chip">📂 {_existing_name}</div>',
                unsafe_allow_html=True,
            )
            _r_col, _d_col = st.columns(2)
            with _r_col:
                if st.button("Replace", key="home_h5ad_replace", use_container_width=True):
                    st.session_state["_home_pending_replace"] = True
            with _d_col:
                if st.button("Remove", key="home_h5ad_remove", use_container_width=True):
                    st.session_state["_home_pending_remove"] = True
        else:
            if _replacing and _existing_path:
                _existing_name = Path(_existing_path).name
                st.markdown(
                    f'<div class="cv-uploaded-chip">📂 {_existing_name}</div>',
                    unsafe_allow_html=True,
                )
                # Hide the file uploader UI and cancel button — both stay in the DOM
                st.markdown(
                    '<style>'
                    'section[data-testid="stSidebar"] [data-testid="stFileUploader"]'
                    '{visibility:hidden!important;height:0!important;overflow:hidden!important;'
                    'margin:0!important;padding:0!important}'
                    'section[data-testid="stSidebar"] [data-testid="stButton"]'
                    '{display:none!important}'
                    '</style>',
                    unsafe_allow_html=True,
                )
            h5ad_file = st.file_uploader(
                "Dataset (.h5ad)",
                type=["h5ad"],
                help="Single-cell dataset in AnnData format",
                key="home_h5ad_upload",
                label_visibility="collapsed",
            )
            if h5ad_file:
                save_path = UPLOADS_DIR / h5ad_file.name
                with open(save_path, "wb") as f:
                    f.write(h5ad_file.getvalue())
                st.session_state.home_h5ad_path = str(save_path)
                st.session_state.pop("home_replacing_h5ad", None)
                st.session_state["_home_pending_upload_done"] = True
            else:
                if not _replacing:
                    st.session_state.home_h5ad_path = None
                else:
                    # In replace mode: auto-open the file dialog, then auto-cancel if dismissed
                    components.html("""
<script>
(function() {
  var doc = (window.parent || window).document;
  var win = window.parent || window;
  var tries = 0;

  function findCancelBtn() {
    var sidebar = doc.querySelector('section[data-testid="stSidebar"]');
    if (!sidebar) return null;
    var btns = sidebar.querySelectorAll('button');
    for (var i = 0; i < btns.length; i++) {
      if ((btns[i].innerText || btns[i].textContent || '').trim() === 'Cancel') return btns[i];
    }
    return null;
  }

  function clickBrowse() {
    tries++;
    if (tries > 40) return;
    var sidebar = doc.querySelector('section[data-testid="stSidebar"]');
    var btn = sidebar && sidebar.querySelector('[data-testid="stFileUploaderDropzone"] button');
    if (btn) {
      btn.click();
      // Wait briefly for the dialog to open, then listen for window regaining focus
      setTimeout(function() {
        win.addEventListener('focus', function onFocus() {
          win.removeEventListener('focus', onFocus);
          // Wait to see if Streamlit reruns (file selected); if Cancel still exists, click it
          setTimeout(function() {
            var cancelBtn = findCancelBtn();
            if (cancelBtn) cancelBtn.click();
          }, 500);
        }, { once: true });
      }, 200);
    } else {
      setTimeout(clickBrowse, 100);
    }
  }
  clickBrowse();
})();
</script>
""", height=0)
                    if st.button("Cancel", key="home_h5ad_cancel"):
                        st.session_state["_home_pending_cancel"] = True

    st.markdown('<p class="cv-field-label">Context</p>', unsafe_allow_html=True)
    context_source = st.radio(
        "Context input",
        ["Structured fields", "Upload summary file"],
        horizontal=True,
        key="home_context_source",
        help="Provide context via structured fields or upload one file containing all context.",
        label_visibility="collapsed",
    )
    if context_source == "Upload summary file":
        context_file = st.file_uploader(
            "Summary file (.txt, .md)",
            type=["txt", "md"],
            help="A single file containing dataset summary, past analyses, focus directions, and biological background.",
            key="home_context_upload",
        )
        if context_file:
            txt = context_file.read().decode()
            fp = UPLOADS_DIR / f"{st.session_state.home_analysis_name}_context.txt"
            fp.write_text(txt, encoding="utf-8")
            st.session_state.home_paper_file_path = str(fp)
            st.session_state.home_paper_text = txt
        else:
            st.session_state.home_paper_file_path = None
            st.session_state.home_paper_text = st.session_state.get("home_paper_text", "")
    else:
        st.session_state.home_paper_file_path = None

    st.divider()
    st.markdown("### ⚙️ Run Configuration")

    analysis_name = st.text_input("Analysis name", key="home_analysis_name", help="Output folders and logs")
    if DEMO_MODE:
        st.session_state.home_num_analyses = 1
        st.number_input(
            "Analyses",
            min_value=1,
            max_value=1,
            value=1,
            disabled=True,
            help="Demo mode runs a single analysis to reduce memory pressure.",
        )
    else:
        st.number_input("Analyses", min_value=1, max_value=20, key="home_num_analyses", help="Number of analyses to run")
    st.number_input("Max steps per analysis", min_value=1, max_value=50, key="home_max_iterations")

    st.markdown("### 🤖 Agent Behavior")
    interactive_mode = st.checkbox("Interactive mode", key="home_interactive_mode", help="Pauses for user feedback/interaction every N steps")
    st.number_input(
        "Intervene every N steps",
        min_value=1,
        max_value=20,
        disabled=not interactive_mode,
        key="home_intervene_every",
        help="Allow user to interact with the agent every N steps in the analysis (1 = after every single step)",
    )
    if interactive_mode:
        st.checkbox("Notify when ready for feedback", key="home_ding_on_pause", help="Play a sound when the agent pauses for feedback")

    st.markdown("### 🔬 Advanced Options")
    st.checkbox("DeepResearch", key="home_use_deepresearch", help="Runs DeepResearch to get additional biological background")

    _EXEC_MODEL_OPTIONS = [
        "claude-sonnet-4-6",
        "claude-opus-4-6",
        "claude-sonnet-4-5",
        "claude-opus-4-5",
        "claude-haiku-4-5",
    ]
    _DEFAULT_EXEC_MODEL = "claude-sonnet-4-6"
    if st.session_state.get("home_execution_model") not in _EXEC_MODEL_OPTIONS:
        st.session_state["home_execution_model"] = _DEFAULT_EXEC_MODEL
    st.selectbox(
        "Execution model",
        _EXEC_MODEL_OPTIONS,
        index=_EXEC_MODEL_OPTIONS.index(st.session_state.get("home_execution_model", _DEFAULT_EXEC_MODEL)),
        key="home_execution_model",
        help="LLM to use to execute the analyses",
    )

    _MODEL_PRESETS = [
        "claude-sonnet-4-6",
        "claude-opus-4-6",
        "claude-sonnet-4-5",
        "claude-opus-4-5",
        "claude-haiku-4-5",
        "gpt-5.2",
        "gpt-5.3",
        "o3-mini",
        "o1",
        "gpt-4o",
        "gpt-4o-mini",
        "Custom...",
    ]
    _DEFAULT_MODEL = "claude-sonnet-4-6"
    if st.session_state.get("home_model_name") in (None, "", "o3-mini"):
        st.session_state["home_model_name"] = _DEFAULT_MODEL
    _current_model = st.session_state.get("home_model_name", _DEFAULT_MODEL)
    _preset_val = _current_model if _current_model in _MODEL_PRESETS else "Custom..."
    _selected_preset = st.selectbox(
        "Hypothesis generation model",
        _MODEL_PRESETS,
        index=_MODEL_PRESETS.index(_preset_val),
        key="_home_model_preset",
        help="OpenAI or Anthropic model for hypothesis/critique generation",
    )
    if _selected_preset == "Custom...":
        _custom = st.text_input("Custom model name", value=_current_model if _preset_val == "Custom..." else "", key="_home_model_custom")
        st.session_state["home_model_name"] = _custom.strip() or _DEFAULT_MODEL
    else:
        st.session_state["home_model_name"] = _selected_preset

    _model_for_validation = st.session_state.get("home_model_name", _DEFAULT_MODEL)
    def _model_provider(m):
        if m.startswith("claude-") or m.startswith("anthropic/"):
            return "anthropic"
        if m.startswith("gpt-") or m.startswith("o1") or m.startswith("o3") or m.startswith("o4"):
            return "openai"
        return "unknown"
    _provider = _model_provider(_model_for_validation)

    st.divider()
    api_keys_ok = True
    if _provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            st.error("OPENAI_API_KEY not set")
            api_keys_ok = False
        else:
            st.caption(f"Using OpenAI model `{_model_for_validation}`")
    elif _provider == "anthropic":
        if not os.getenv("ANTHROPIC_API_KEY"):
            st.error("ANTHROPIC_API_KEY not set")
            api_keys_ok = False
        else:
            st.caption(f"Using Anthropic model `{_model_for_validation}`")
    else:
        st.warning(f"Unknown provider for `{_model_for_validation}`. Ensure the correct API key is set.")
        if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
            st.error("No API keys set (OPENAI_API_KEY or ANTHROPIC_API_KEY)")
            api_keys_ok = False
    if not os.getenv("ANTHROPIC_API_KEY"):
        st.error("ANTHROPIC_API_KEY not set (required for Claude execution agent)")
        api_keys_ok = False

    # Deferred reruns — processed here so ALL sidebar widgets above have already rendered
    # and committed their values to session state before the rerun fires.
    if st.session_state.pop("_home_pending_remove", False):
        st.session_state["home_h5ad_path"] = None
        st.rerun()
    elif st.session_state.pop("_home_pending_replace", False):
        st.session_state["home_replacing_h5ad"] = True
        st.rerun()
    elif st.session_state.pop("_home_pending_cancel", False):
        st.session_state.pop("home_replacing_h5ad", None)
        st.rerun()
    elif st.session_state.pop("_home_pending_upload_done", False):
        st.rerun()

# Evaluated after the sidebar so home_h5ad_path is correctly set by the file uploader
_h5ad_uploaded = bool(st.session_state.get("home_h5ad_path")) if not DEMO_MODE else FIXED_H5AD_PATH.exists()


# Main area: status banner or getting started
_STEPS_HTML = """
<div class="cv-steps-row">
  <div class="cv-step-card">
    <div class="cv-step-num">Step 1</div>
    <div class="cv-step-title">Upload dataset</div>
    <div class="cv-step-desc">Add your <code>.h5ad</code> AnnData file using the sidebar uploader.</div>
  </div>
  <div class="cv-step-card">
    <div class="cv-step-num">Step 2</div>
    <div class="cv-step-title">Describe your data</div>
    <div class="cv-step-desc">Fill in the Dataset Summary below — cohort, cell types, preprocessing, caveats.</div>
  </div>
  <div class="cv-step-card">
    <div class="cv-step-num">Step 3</div>
    <div class="cv-step-title">Configure the run</div>
    <div class="cv-step-desc">Set the number of analyses, max steps, and execution mode in the sidebar.</div>
  </div>
  <div class="cv-step-card">
    <div class="cv-step-num">Step 4 <span style="font-weight:400;opacity:0.7;">(optional)</span></div>
    <div class="cv-step-title">Add context</div>
    <div class="cv-step-desc">Provide past analyses, focus directions, or biological background below.</div>
  </div>
  <div class="cv-step-card cv-step-card--cta">
    <div class="cv-step-num">Step 5</div>
    <div class="cv-step-title">Run</div>
    <div class="cv-step-desc">Click <strong>Run analysis</strong> in the sidebar to launch the agent.</div>
  </div>
</div>
"""

if _h5ad_uploaded:
    _fname = Path(st.session_state.get("home_h5ad_path", "")).name if not DEMO_MODE else FIXED_H5AD_PATH.name
    _summary_filled = bool((st.session_state.get("home_dataset_summary") or "").strip())
    _readiness = "✅\u2003Ready to run" if _summary_filled else "⚠️\u2003Add a Dataset Summary below, change the run configurations as desired, and then run"
    st.markdown(
        f'<div class="cv-status-banner">'
        f'<span class="cv-status-file">📂 {_fname}</span>'
        f'<span class="cv-status-sep">·</span>'
        f'<span class="cv-status-state">{_readiness}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    with st.expander("Getting Started", expanded=False):
        st.caption("CellVoyager autonomously generates and explores single-cell analysis ideas inside a live Jupyter notebook.")
        st.markdown(_STEPS_HTML, unsafe_allow_html=True)
else:
    st.markdown("# Getting Started")
    st.caption("CellVoyager autonomously generates and explores single-cell analysis ideas inside a live Jupyter notebook.")
    st.markdown(_STEPS_HTML, unsafe_allow_html=True)

# Main area: context input
if st.session_state.get("home_context_source") == "Structured fields":
    st.markdown("# Dataset Summary")
    st.caption("Required — the agent uses this to understand your dataset before generating analyses.")
    st.text_area(
        "Summary of the dataset",
        height=140,
        placeholder="Example: PBMC single-cell RNA-seq dataset from COVID-19 patients and healthy controls. ~120k cells with donor metadata including disease severity, age, and tissue source. Cells were clustered into major immune populations (T cells, B cells, NK cells, monocytes). Key columns: donor_id, disease_severity, cell_type, tissue.",
        key="home_dataset_summary",
        label_visibility="collapsed",
    )
    _run_btn_col, _ = st.columns([1, 4])
    with _run_btn_col:
        _run_clicked = st.button("▶\u2003Run analysis", type="primary", key="run_btn_main", use_container_width=True)

    st.markdown('<p class="cv-optional-header">Additional context <span style="font-weight:400;opacity:0.55;">— optional</span></p>', unsafe_allow_html=True)
    st.caption("Guide the agent and help it avoid redundant work.")

    with st.expander("Past analyses tried", expanded=False):
        st.text_area(
            "Past analyses tried",
            height=120,
            placeholder="List analyses you have tried along with their respective results.",
            key="home_past_analyses",
            label_visibility="collapsed",
        )

    with st.expander("Directions to focus on", expanded=False):
        st.text_area(
            "Directions to focus on",
            height=120,
            placeholder="Specify what kind of questions/analyses you want the agent to focus on.",
            key="home_focus_directions",
            label_visibility="collapsed",
        )

    with st.expander("Additional biological background", expanded=False):
        st.text_area(
            "Additional biological background",
            height=120,
            placeholder="Any sort of additional context about the disease, experimental setup, etc.",
            key="home_bio_background",
            label_visibility="collapsed",
        )
else:
    st.markdown("#### 📄 Uploaded context summary")
    st.info("Using uploaded summary file as full analysis context.")
    _run_btn_col2, _ = st.columns([1, 4])
    with _run_btn_col2:
        _run_clicked = st.button("▶\u2003Run analysis", type="primary", key="run_btn_main_upload", use_container_width=True)
if _run_clicked and not st.session_state.get("run_started"):
    context_source = st.session_state.get("home_context_source", "Structured fields")
    if context_source == "Upload summary file":
        uploaded_context = st.session_state.home_paper_text or ""
        _paper_file_path = st.session_state.get("home_paper_file_path")
        if not uploaded_context.strip() and _paper_file_path and Path(_paper_file_path).exists():
            try:
                uploaded_context = Path(_paper_file_path).read_text(encoding="utf-8")
            except Exception:
                pass
        _paper = f"""# USER CONTEXT PACKAGE
context_source: uploaded_summary_file

## Dataset summary
(included within uploaded context if provided by user)

## Past analyses tried
(included within uploaded context if provided by user)

## Directions to focus on
(included within uploaded context if provided by user)

## Additional biological background
(included within uploaded context if provided by user)

## Uploaded context (verbatim)
BEGIN_UPLOADED_CONTEXT
{uploaded_context}
END_UPLOADED_CONTEXT
"""
    else:
        dataset_summary = (st.session_state.get("home_dataset_summary") or "").strip()
        past_analyses = (st.session_state.get("home_past_analyses") or "").strip()
        focus_directions = (st.session_state.get("home_focus_directions") or "").strip()
        bio_background = (st.session_state.get("home_bio_background") or "").strip()
        _paper = f"""# USER CONTEXT PACKAGE
context_source: structured_fields

## Dataset summary
{dataset_summary}

## Past analyses tried
{past_analyses or "(none provided)"}

## Directions to focus on
{focus_directions or "(none provided)"}

## Additional biological background
{bio_background or "(none provided)"}
"""
        _paper_file_path = None
    _analysis_name = st.session_state.home_analysis_name
    _num_analyses = 1 if DEMO_MODE else st.session_state.home_num_analyses
    _max_iterations = st.session_state.home_max_iterations
    _interactive_mode = st.session_state.home_interactive_mode
    _intervene_every = st.session_state.home_intervene_every
    _use_deepresearch = st.session_state.home_use_deepresearch
    _model_name = st.session_state.home_model_name
    _execution_model = st.session_state.get("home_execution_model", "claude-sonnet-4-6")
    _h5ad_path = str(FIXED_H5AD_PATH) if DEMO_MODE else st.session_state.get("home_h5ad_path")
    # Claude execution agent always needs ANTHROPIC_API_KEY; hypothesis model needs its own key
    def __model_provider(m):
        if m.startswith("claude-") or m.startswith("anthropic/"):
            return "anthropic"
        return "openai"
    _hyp_provider = __model_provider(_model_name)
    _hyp_key_ok = bool(os.getenv("OPENAI_API_KEY")) if _hyp_provider == "openai" else bool(os.getenv("ANTHROPIC_API_KEY"))
    _api_ok = _hyp_key_ok and bool(os.getenv("ANTHROPIC_API_KEY"))
    _has_h5ad = _h5ad_path and Path(_h5ad_path).exists()
    if context_source == "Structured fields":
        _has_paper = bool((st.session_state.get("home_dataset_summary") or "").strip())
    else:
        _has_paper = bool(_paper.strip()) or (_paper_file_path and Path(_paper_file_path).exists())
    if not _api_ok:
        _run_validation_error = "Set API keys in your environment and restart."
    elif not _has_h5ad:
        _run_validation_error = (
            f"Fixed demo dataset not found at: {FIXED_H5AD_PATH}"
            if DEMO_MODE else "Upload a dataset (.h5ad file) to run the analysis."
        )
    elif not _has_paper:
        _run_validation_error = (
            "Provide context: dataset summary is required (other fields optional), "
            "or upload a summary file containing all context."
        )
    else:
        h5ad_path = Path(_h5ad_path)
        paper_path = UPLOADS_DIR / f"{_analysis_name}_paper.txt"
        paper_path.write_text(_paper, encoding="utf-8")
        run_output_dir = OUTPUTS_BASE / f"{_analysis_name}_gui_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        run_output_dir.mkdir(parents=True, exist_ok=True)
        # Ensure a fresh run starts with no stale pause/stop/error control files.
        for control_file in (
            g._PAUSE_REQUEST_FILE,
            g._PAUSE_RESPONSE_FILE,
            g._EXECUTE_REQUEST_FILE,
            g._STEP_COUNT_FILE,
            g._STOP_REQUEST_FILE,
            g._AGENT_SUMMARY_FILE,
            g._CHAT_REQUEST_FILE,
            g._CHAT_RESPONSE_FILE,
            g._RUN_ERROR_FILE,
        ):
            (run_output_dir / control_file).unlink(missing_ok=True)
        st.session_state.pop("run_error", None)
        st.session_state.run_output_dir = str(run_output_dir)
        st.session_state.run_num_analyses = int(_num_analyses)
        OUTPUTS_BASE.mkdir(parents=True, exist_ok=True)
        _LAST_RUN_FILE.write_text(st.session_state.run_output_dir, encoding="utf-8")
        (run_output_dir / _RUN_INTERACTIVE_FILE).write_text("1" if _interactive_mode else "0", encoding="utf-8")
        run_config = {
            "h5ad_path": str(h5ad_path),
            "paper_path": str(paper_path),
            "analysis_name": _analysis_name,
            "execution_mode": "claude",
            "max_iterations": int(_max_iterations),
            "model_name": _model_name,
            "execution_model": _execution_model,
            "use_deepresearch": _use_deepresearch,
            "intervene_every": int(_intervene_every),
            "ding_on_pause": bool(st.session_state.get("home_ding_on_pause", False)),
            "num_analyses": int(_num_analyses),
        }
        (run_output_dir / g._RUN_CONFIG_FILE).write_text(json.dumps(run_config), encoding="utf-8")
        cmd = [
            sys.executable, str(ROOT / "run_cellvoyager.py"),
            "--h5ad-path", str(h5ad_path),
            "--paper-path", str(paper_path),
            "--analysis-name", _analysis_name,
            "--num-analyses", str(_num_analyses),
            "--max-iterations", str(int(_max_iterations)),
            "--execution-mode", "claude",
            "--model-name", _model_name,
            "--output-home", str(ROOT),
            "--log-home", str(ROOT / "logs"),
            "--output-dir", st.session_state.run_output_dir,
        ]
        # Always enable interactive plumbing for GUI-driven pause/continue.
        # If step-by-step mode is off, use a very large intervene interval so pauses
        # only happen when the user clicks Stop.
        intervene = int(_intervene_every) if _interactive_mode else 999999
        cmd.extend(["--interactive", "--intervene-every", str(intervene)])
        if _execution_model:
            cmd.extend(["--execution-model", _execution_model])
        if _use_deepresearch:
            cmd.append("--deepresearch")
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["CELLVOYAGER_GUI_INTERACTIVE"] = "1"
        proc = subprocess.Popen(
            cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=env, start_new_session=True,
        )
        (run_output_dir / _RUN_PID_FILE).write_text(str(proc.pid), encoding="utf-8")
        st.session_state.run_proc = proc
        st.session_state.run_pid = proc.pid
        st.session_state.run_output = []
        st.session_state.run_cmd = cmd
        st.session_state.run_started = True
        st.session_state.run_interactive_mode = True
        st.session_state.run_thread_started = True
        st.session_state.pop("_active_run_tab", None)
        log_path = run_output_dir / _RUN_LOG_FILE
        t = threading.Thread(target=g._read_output, args=(proc, st.session_state.run_output, log_path))
        t.daemon = True
        t.start()
        # Track in session history
        if "session_runs" not in st.session_state:
            st.session_state.session_runs = []
        st.session_state.session_runs.append({
            "output_dir": str(run_output_dir),
            "analysis_name": _analysis_name,
            "num_analyses": int(_num_analyses),
            "started_at": datetime.datetime.now().strftime("%b %d, %I:%M %p"),
        })
        st.switch_page("pages/analysis.py")

if _run_validation_error:
    st.error(_run_validation_error)

# Past runs — only show runs from this session (not all runs on disk)
_session_runs = st.session_state.get("session_runs", [])
if _session_runs:
    st.markdown("---")
    st.markdown("## Past Runs")
    st.caption("Analyses from this session. Click **View** to open a run or continue it further.")

    # Most recent first
    _runs_reversed = list(reversed(_session_runs))
    _tab_labels = [
        f"{r['analysis_name']} · {r['started_at']}" for r in _runs_reversed
    ]
    _run_tabs = st.tabs(_tab_labels)
    for _tab, _run in zip(_run_tabs, _runs_reversed):
        with _tab:
            _out_dir = Path(_run["output_dir"])
            _cfg_path = _out_dir / g._RUN_CONFIG_FILE
            _err_file = _out_dir / g._RUN_ERROR_FILE
            _num = _run["num_analyses"]

            # Status badge
            if _err_file.exists():
                st.error("Crashed")
            else:
                _notebooks = g._collect_notebooks_by_analysis(str(_out_dir), _num)
                _done = sum(1 for _, nb in _notebooks if nb)
                if _done == _num:
                    st.success(f"Completed — {_done}/{_num} {'analyses' if _num > 1 else 'analysis'}")
                elif _done == 0:
                    st.info("No notebooks yet")
                else:
                    st.warning(f"Partial — {_done}/{_num} analyses")

            if st.button("View / Continue →", key=f"home_view_run_{_run['output_dir']}", type="primary"):
                st.session_state.run_output_dir = _run["output_dir"]
                st.session_state.run_num_analyses = _num
                st.session_state.pop("run_error", None)
                # Restore error state if the run crashed
                if _err_file.exists():
                    try:
                        st.session_state["run_error"] = _err_file.read_text(encoding="utf-8").strip()
                    except Exception:
                        pass
                st.switch_page("pages/analysis.py")

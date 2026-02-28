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
    ("home_paper_text", ""),
    ("home_analysis_name", "covid19"),
    ("home_num_analyses", 1),
    ("home_max_iterations", 8),
    ("home_execution_mode", "claude"),
    ("home_interactive_mode", False),
    ("home_intervene_every", 1),
    ("home_use_deepresearch", True),
    ("home_model_name", "o3-mini"),
]:
    if key not in st.session_state:
        st.session_state[key] = default

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
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _get_run_log() -> str:
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
    proc = st.session_state.run_proc
    if proc is not None and proc.poll() is None:
        return True
    pid = st.session_state.get("run_pid")
    if pid is not None and _process_alive(pid):
        return True
    return False


def _kill_analysis():
    proc = st.session_state.run_proc
    pid = st.session_state.run_pid
    out_dir = Path(st.session_state.run_output_dir) if st.session_state.run_output_dir else None
    if proc is not None:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            try:
                proc.kill()
            except OSError:
                pass
    elif pid is not None:
        try:
            os.kill(pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass
    if out_dir and out_dir.exists():
        (out_dir / _RUN_PID_FILE).unlink(missing_ok=True)
    out_dir_str = st.session_state.run_output_dir
    keys_to_drop = [k for k in st.session_state.keys() if isinstance(k, str) and out_dir_str and k.startswith("agent_chat_") and out_dir_str in k]
    for k in keys_to_drop:
        st.session_state.pop(k, None)
    st.session_state.run_proc = None
    st.session_state.run_output = []
    st.session_state.run_output_dir = None
    st.session_state.run_pid = None
    st.session_state.run_started = False
    st.session_state.run_thread_started = False
    st.session_state.run_cmd = None


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


def _render_cell_outputs(cell):
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


def _render_notebook_jupyter_style(nb_path, editable=False, pause_id=None):
    import nbformat as nbf
    try:
        with open(nb_path, encoding="utf-8") as f:
            content = f.read()
        if not content.strip():
            st.warning(f"Notebook is empty or still being written: `{Path(nb_path).name}`")
            return None
        nb = nbf.reads(content, as_version=4)
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
    add_code, add_md, continue_clicked, finish_clicked, edit_clicked = False, False, False, False, False
    if editable:
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
        try:
            from streamlit_extras.stylable_container import stylable_container
            button_container = stylable_container(
                key=f"pause_buttons_{pause_id}",
                css_styles="""
                    button {
                        padding: 1.5rem 2.5rem !important;
                        font-size: 1.6rem !important;
                        font-weight: 700 !important;
                        min-height: 72px !important;
                    }
                """,
            )
        except ImportError:
            button_container = st.container()
        with button_container:
            btn_cols = st.columns(3)
            with btn_cols[0]:
                edit_clicked = st.form_submit_button("✏️ Edit", help="Show edit options for cells", type="primary")
            with btn_cols[1]:
                continue_clicked = st.form_submit_button("▶ Continue", help="Send feedback and continue analysis", type="primary")
            with btn_cols[2]:
                finish_clicked = st.form_submit_button("🏠 Finish", help="Tell agent to finish the analysis", type="primary")
        if edit_mode:
            st.caption("Insert cells after a specific cell above, or add at end. Run code cells, then Continue or Finish.")
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
        return cell_sources, feedback, nb, run_clicked, insert_after_idx, insert_type, continue_clicked, finish_clicked, edit_clicked
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


def _render_chat_box(output_dir: str | None):
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
    prompt = st.chat_input("Ask about the results...")
    if prompt:
        st.session_state[pending_key] = prompt
        st.rerun()


def _should_show_chat():
    if st.session_state.run_started and _has_live_run():
        proc = st.session_state.run_proc
        in_pause = (
            st.session_state.run_interactive_mode
            and _pause_request_path()
            and _pause_request_path().exists()
        )
        if proc is not None:
            return in_pause or proc.poll() is not None
        return in_pause or True
    return bool(st.session_state.run_output_dir)

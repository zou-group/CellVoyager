# cellvoyager_claude_mcp.py
#
# Minimal CellVoyager executor:
# - Custom MCP notebook tools (your own server, over stdio)
# - One long-lived Jupyter kernel
# - Direct .ipynb edits + execution
# - Streaming Claude logs written to a plain text file
#
# Install:
#   pip install claude-agent-sdk mcp jupyter_client nbformat ipykernel

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import nbformat as nbf
from nbformat.v4 import (
    new_code_cell,
    new_markdown_cell,
    new_notebook,
    new_output,
)
from jupyter_client import KernelManager


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

def strip_code_fences(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"^```python\s*", "", text.strip())
    text = re.sub(r"^```\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text.strip())
    return text.strip()


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


# -----------------------------------------------------------------------------
# Notebook + kernel state (owned by the MCP server)
# -----------------------------------------------------------------------------

class NotebookSession:
    def __init__(self, notebook_path: str):
        self.path = Path(notebook_path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)

        if self.path.exists():
            self.nb = nbf.read(self.path, as_version=4)
        else:
            self.nb = new_notebook()

        self.km = KernelManager()
        self.km.start_kernel(cwd=str(self.path.parent))
        self.kc = self.km.client()
        self.kc.start_channels()
        self.kc.wait_for_ready(timeout=60)
        self.setup_executed = False

        # Make inline plots show up in notebook outputs and ensure cwd is correct.
        bootstrap = (
            "%matplotlib inline\n"
            "import warnings\n"
            "warnings.filterwarnings('ignore')\n"
            "import os\n"
            f"os.chdir(r'''{self.path.parent}''')\n"
        )
        self._execute_source(bootstrap)

        self.save()

    def shutdown(self) -> None:
        try:
            self.kc.stop_channels()
        except Exception:
            pass
        try:
            self.km.shutdown_kernel(now=True)
        except Exception:
            pass

    def restart_kernel(self) -> None:
        self.shutdown()
        self.km = KernelManager()
        self.km.start_kernel(cwd=str(self.path.parent))
        self.kc = self.km.client()
        self.kc.start_channels()
        self.kc.wait_for_ready(timeout=60)
        self.setup_executed = False
        bootstrap = (
            "%matplotlib inline\n"
            "import warnings\n"
            "warnings.filterwarnings('ignore')\n"
            "import os\n"
            f"os.chdir(r'''{self.path.parent}''')\n"
        )
        self._execute_source(bootstrap)

    def save(self) -> None:
        # Atomic write: write to a temp file then rename so the GUI never
        # sees a truncated/empty notebook.
        tmp_path = self.path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            nbf.write(self.nb, f)
        tmp_path.replace(self.path)

    def _normalize_insert_index(self, index: int | None) -> int:
        if index is None or index < 0 or index > len(self.nb.cells):
            return len(self.nb.cells)
        return index

    def _require_index(self, index: int) -> None:
        if index < 0 or index >= len(self.nb.cells):
            raise IndexError(f"Cell index {index} out of range (0..{len(self.nb.cells)-1})")

    def insert_cell(self, index: int | None, cell_type: str, source: str) -> dict[str, Any]:
        index = self._normalize_insert_index(index)

        if cell_type == "markdown":
            cell = new_markdown_cell(source)
        elif cell_type == "code":
            cell = new_code_cell(source)
        else:
            raise ValueError("cell_type must be 'markdown' or 'code'")

        self.nb.cells.insert(index, cell)
        self.save()
        return {
            "ok": True,
            "cell_index": index,
            "cell_type": cell_type,
            "num_cells": len(self.nb.cells),
        }

    def overwrite_cell_source(self, index: int, source: str) -> dict[str, Any]:
        self._require_index(index)
        self.nb.cells[index].source = source
        if self.nb.cells[index].cell_type == "code":
            self.nb.cells[index]["outputs"] = []
            self.nb.cells[index]["execution_count"] = None
        self.save()
        return {"ok": True, "cell_index": index}

    def delete_cell(self, index: int) -> dict[str, Any]:
        self._require_index(index)
        deleted_type = self.nb.cells[index].cell_type
        del self.nb.cells[index]
        self.save()
        return {
            "ok": True,
            "deleted_index": index,
            "deleted_type": deleted_type,
            "num_cells": len(self.nb.cells),
        }

    def read_notebook(self) -> dict[str, Any]:
        # Reload from disk to pick up user edits (e.g. from Jupyter UI during interactive pause)
        if self.path.exists():
            self.nb = nbf.read(self.path, as_version=4)
        cells = []
        for i, cell in enumerate(self.nb.cells):
            cells.append({
                "index": i,
                "cell_type": cell.cell_type,
                "source_preview": self._trim(cell.source, 600),
                "output_preview": self._cell_output_preview(cell, 1200),
            })
        return {
            "ok": True,
            "notebook_path": str(self.path),
            "num_cells": len(self.nb.cells),
            "cells": cells,
        }

    def read_cell(self, index: int) -> dict[str, Any]:
        self._require_index(index)
        cell = self.nb.cells[index]
        return {
            "ok": True,
            "cell_index": index,
            "cell_type": cell.cell_type,
            "source": cell.source,
            "output_preview": self._cell_output_preview(cell, 4000),
            "execution_count": cell.get("execution_count"),
        }

    def execute_cell(self, index: int) -> dict[str, Any]:
        self._require_index(index)
        cell = self.nb.cells[index]
        if cell.cell_type != "code":
            raise ValueError(f"Cell {index} is not a code cell")

        source = cell.source
        if isinstance(source, list):
            source = "\n".join(source)

        # Signal to the GUI that this cell is executing
        running_path = self.path.parent / ".cellvoyager_running_cell"
        try:
            running_path.write_text(json.dumps({
                "cell_index": index,
                "started_at": time.time(),
            }), encoding="utf-8")
        except Exception:
            pass

        result = self._execute_source(source)

        # Clear the running signal
        try:
            running_path.unlink(missing_ok=True)
        except Exception:
            pass

        cell["outputs"] = result["outputs"]
        cell["execution_count"] = result["execution_count"]
        self.save()

        out = {
            "ok": result["ok"],
            "cell_index": index,
            "execution_count": result["execution_count"],
            "output_preview": result["preview"],
            "error": result.get("error"),
        }
        if result.get("paused_by_user"):
            out["paused_by_user"] = True
        return out

    def insert_execute_code_cell(self, index: int | None, source: str) -> dict[str, Any]:
        inserted = self.insert_cell(index=index, cell_type="code", source=source)
        idx = inserted["cell_index"]
        executed = self.execute_cell(idx)
        out = {
            "ok": executed["ok"],
            "cell_index": idx,
            "execution_count": executed["execution_count"],
            "output_preview": executed["output_preview"],
            "error": executed.get("error"),
        }
        if executed.get("paused_by_user"):
            out["paused_by_user"] = True
        return out

    def _execute_source(self, source: str) -> dict[str, Any]:
        msg_id = self.kc.execute(source, allow_stdin=False, stop_on_error=False)

        outputs = []
        execution_count = None
        error_text = None
        paused_by_user = False
        killed_by_user = False

        # Kill-cell signal file lives in the notebook's parent directory
        kill_path = self.path.parent / ".cellvoyager_kill_cell"

        import queue
        while True:
            try:
                msg = self.kc.get_iopub_msg(timeout=2)
            except queue.Empty:
                # Check for kill signal during idle waits
                if kill_path.exists():
                    try:
                        kill_path.unlink(missing_ok=True)
                        self.km.interrupt_kernel()
                        killed_by_user = True
                    except Exception:
                        pass
                continue
            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue

            msg_type = msg["msg_type"]
            content = msg["content"]

            if msg_type == "status":
                if content.get("execution_state") == "idle":
                    break

            elif msg_type == "execute_input":
                execution_count = content.get("execution_count", execution_count)

            elif msg_type == "stream":
                outputs.append(new_output(
                    output_type="stream",
                    name=content["name"],
                    text=content["text"],
                ))

            elif msg_type == "display_data":
                outputs.append(new_output(
                    output_type="display_data",
                    data=content["data"],
                    metadata=content.get("metadata", {}),
                ))

            elif msg_type == "execute_result":
                execution_count = content.get("execution_count", execution_count)
                outputs.append(new_output(
                    output_type="execute_result",
                    data=content["data"],
                    metadata=content.get("metadata", {}),
                    execution_count=execution_count,
                ))

            elif msg_type == "error":
                outputs.append(new_output(
                    output_type="error",
                    ename=content["ename"],
                    evalue=content["evalue"],
                    traceback=content["traceback"],
                ))
                error_text = "\n".join(content["traceback"][-8:])

            elif msg_type == "clear_output":
                outputs = []

        preview = self._outputs_preview(outputs, 4000)
        ok = error_text is None and not paused_by_user and not killed_by_user

        if killed_by_user:
            error_text = "Cell execution was interrupted by user."

        return {
            "ok": ok,
            "outputs": outputs,
            "execution_count": execution_count,
            "preview": preview,
            "error": error_text,
            "paused_by_user": paused_by_user,
            "killed_by_user": killed_by_user,
        }

    @staticmethod
    def _trim(text: str, limit: int) -> str:
        text = text or ""
        return text if len(text) <= limit else text[:limit] + "...[truncated]"

    def _cell_output_preview(self, cell: Any, limit: int) -> str:
        outputs = cell.get("outputs", []) if cell.cell_type == "code" else []
        return self._outputs_preview(outputs, limit)

    def _outputs_preview(self, outputs: list[Any], limit: int) -> str:
        parts = []

        for out in outputs:
            ot = out.get("output_type")

            if ot == "stream":
                parts.append(out.get("text", ""))

            elif ot in ("display_data", "execute_result"):
                data = out.get("data", {})
                if "text/plain" in data:
                    parts.append(str(data["text/plain"]))
                elif "image/png" in data:
                    parts.append("[image/png output]")
                elif "text/html" in data:
                    parts.append("[text/html output]")
                else:
                    parts.append("[rich output]")

            elif ot == "error":
                parts.append("\n".join(out.get("traceback", [])))

        joined = "\n".join(parts).strip()
        return self._trim(joined, limit)


class SessionRegistry:
    def __init__(self):
        self.current: NotebookSession | None = None

    def use_notebook(self, notebook_path: str) -> NotebookSession:
        notebook_path = str(Path(notebook_path).resolve())

        if self.current is not None:
            if str(self.current.path) == notebook_path:
                return self.current
            self.current.shutdown()

        self.current = NotebookSession(notebook_path)
        return self.current

    def require_current(self) -> NotebookSession:
        if self.current is None:
            raise RuntimeError("No active notebook. Call use_notebook first.")
        return self.current


REGISTRY = SessionRegistry()


# -----------------------------------------------------------------------------
# MCP server
# -----------------------------------------------------------------------------

def run_mcp_server() -> None:
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("jupyter")

    def _force_gui_pause_if_requested(session: NotebookSession | None) -> tuple[bool, str]:
        """Server-side pause gate: honor GUI Stop even if the agent misses check_user_stop."""
        if os.environ.get("CELLVOYAGER_GUI_INTERACTIVE") != "1":
            return False, ""
        output_dir = Path(os.environ.get("CELLVOYAGER_INTERACTIVE_OUTPUT_DIR", "."))
        request_path = output_dir / _PAUSE_REQUEST_FILE
        response_path = output_dir / _PAUSE_RESPONSE_FILE
        stop_request_path = output_dir / _STOP_REQUEST_FILE
        execute_request_path = output_dir / _EXECUTE_REQUEST_FILE
        should_pause = stop_request_path.exists() or request_path.exists()
        if not should_pause:
            return False, ""

        nb_path = str(session.path) if session else ""
        if session and not request_path.exists():
            request_path.write_text(nb_path, encoding="utf-8")

        # Wait for GUI Continue/Finish; allow GUI-triggered execute while paused.
        while True:
            if response_path.exists():
                feedback = response_path.read_text(encoding="utf-8").strip()
                response_path.unlink(missing_ok=True)
                request_path.unlink(missing_ok=True)
                stop_request_path.unlink(missing_ok=True)
                if session and session.path.exists():
                    session.nb = nbf.read(session.path, as_version=4)
                return True, feedback

            if session and execute_request_path.exists():
                try:
                    req = json.loads(execute_request_path.read_text(encoding="utf-8"))
                    execute_request_path.unlink(missing_ok=True)
                    idx = int(req.get("cell_index", -1))
                    if session.path.exists():
                        session.nb = nbf.read(session.path, as_version=4)
                    if 0 <= idx < len(session.nb.cells) and session.nb.cells[idx].cell_type == "code":
                        session.execute_cell(idx)
                except Exception as e:
                    sys.stderr.write(f"[CellVoyager] Forced-pause execute failed: {e}\n")

            time.sleep(0.05)

    def _paused_ack(user_feedback: str) -> dict[str, Any]:
        return {
            "ok": True,
            "paused_by_user": True,
            "user_feedback": user_feedback,
            "output_preview": "",
        }

    @mcp.tool()
    def use_notebook(notebook_path: str) -> dict[str, Any]:
        session = REGISTRY.use_notebook(notebook_path)
        paused_by_user, user_feedback = _force_gui_pause_if_requested(session)
        if paused_by_user:
            out = _paused_ack(user_feedback)
            out["notebook_path"] = str(session.path)
            out["num_cells"] = len(session.nb.cells)
            return out
        # Auto-execute setup cell exactly once per kernel session so adata is loaded once.
        if (
            not session.setup_executed
            and len(session.nb.cells) > 1
            and session.nb.cells[1].cell_type == "code"
        ):
            session.execute_cell(1)
            session.setup_executed = True
        # Insert initial analysis plan only after setup has finished executing.
        if session.setup_executed and not bool(session.nb.metadata.get("cellvoyager_plan_inserted", False)):
            plan = session.nb.metadata.get("cellvoyager_initial_plan")
            if isinstance(plan, list) and plan:
                plan_md = "# Analysis Plan\n\n" + "\n".join(
                    f"{i+1}. {step}" for i, step in enumerate(plan)
                )
                session.insert_cell(index=2, cell_type="markdown", source=plan_md)
            session.nb.metadata["cellvoyager_plan_inserted"] = True
            session.save()
        out = {
            "ok": True,
            "notebook_path": str(session.path),
            "num_cells": len(session.nb.cells),
        }
        return out

    @mcp.tool()
    def read_notebook() -> dict[str, Any]:
        _force_gui_pause_if_requested(REGISTRY.current)
        return REGISTRY.require_current().read_notebook()

    @mcp.tool()
    def read_cell(index: int) -> dict[str, Any]:
        _force_gui_pause_if_requested(REGISTRY.current)
        return REGISTRY.require_current().read_cell(index)

    @mcp.tool()
    def insert_cell(index: int | None, cell_type: str, source: str) -> dict[str, Any]:
        # In GUI interactive mode, always append to preserve user-inserted cell positions
        if os.environ.get("CELLVOYAGER_GUI_INTERACTIVE") == "1":
            index = None
        session = REGISTRY.require_current()
        paused_by_user, user_feedback = _force_gui_pause_if_requested(session)
        if paused_by_user:
            return _paused_ack(user_feedback)
        out = session.insert_cell(index=index, cell_type=cell_type, source=source)
        return out

    @mcp.tool()
    def overwrite_cell_source(index: int, source: str) -> dict[str, Any]:
        session = REGISTRY.require_current()
        paused_by_user, user_feedback = _force_gui_pause_if_requested(session)
        if paused_by_user:
            return _paused_ack(user_feedback)
        out = session.overwrite_cell_source(index=index, source=source)
        return out

    @mcp.tool()
    def delete_cell(index: int) -> dict[str, Any]:
        session = REGISTRY.require_current()
        paused_by_user, user_feedback = _force_gui_pause_if_requested(session)
        if paused_by_user:
            return _paused_ack(user_feedback)
        out = session.delete_cell(index=index)
        return out

    @mcp.tool()
    def execute_cell(index: int) -> dict[str, Any]:
        session = REGISTRY.require_current()
        paused_by_user, user_feedback = _force_gui_pause_if_requested(session)
        if paused_by_user:
            return _paused_ack(user_feedback)
        out = session.execute_cell(index=index)
        return out

    @mcp.tool()
    def insert_execute_code_cell(index: int | None, source: str) -> dict[str, Any]:
        if os.environ.get("CELLVOYAGER_GUI_INTERACTIVE") == "1":
            index = None
        session = REGISTRY.require_current()
        paused_by_user, user_feedback = _force_gui_pause_if_requested(session)
        if paused_by_user:
            return _paused_ack(user_feedback)
        inserted = session.insert_cell(index=index, cell_type="code", source=source)
        # Re-check pause after insertion so Stop can block before execution starts.
        paused_by_user_2, user_feedback_2 = _force_gui_pause_if_requested(session)
        if paused_by_user_2:
            out = _paused_ack(user_feedback_2)
            out["cell_index"] = inserted["cell_index"]
            return out
        executed = session.execute_cell(inserted["cell_index"])
        out = {
            "ok": executed["ok"],
            "cell_index": inserted["cell_index"],
            "execution_count": executed["execution_count"],
            "output_preview": executed["output_preview"],
            "error": executed.get("error"),
        }
        if executed.get("paused_by_user"):
            out["paused_by_user"] = True
        return out

    @mcp.tool()
    def restart_kernel() -> dict[str, Any]:
        session = REGISTRY.require_current()
        paused_by_user, user_feedback = _force_gui_pause_if_requested(session)
        if paused_by_user:
            out = _paused_ack(user_feedback)
            out["notebook_path"] = str(session.path)
            return out
        session.restart_kernel()
        return {"ok": True, "notebook_path": str(session.path)}

    if os.environ.get("CELLVOYAGER_INTERACTIVE_MODE") == "1":
        output_dir = Path(os.environ.get("CELLVOYAGER_INTERACTIVE_OUTPUT_DIR", "."))
        request_path = output_dir / _PAUSE_REQUEST_FILE
        response_path = output_dir / _PAUSE_RESPONSE_FILE
        execute_request_path = output_dir / _EXECUTE_REQUEST_FILE
        step_count_path = output_dir / _STEP_COUNT_FILE
        agent_summary_path = output_dir / _AGENT_SUMMARY_FILE
        stop_request_path = output_dir / _STOP_REQUEST_FILE

        @mcp.tool()
        def check_user_stop() -> dict[str, Any]:
            """Check if the user has requested to stop or pause. Call before each new step and before tool calls.
            If stop_requested is True, do not add any more steps; exit immediately.
            If pause_requested is True (GUI Stop button), call pause_for_user_review immediately."""
            stop_exists = stop_request_path.exists()
            gui_mode = os.environ.get("CELLVOYAGER_GUI_INTERACTIVE") == "1"
            # In GUI mode, STOP file means pause (not exit); agent should call pause_for_user_review
            if gui_mode and stop_exists:
                return {"stop_requested": False, "pause_requested": True}
            return {"stop_requested": stop_exists, "pause_requested": False}

        _FEEDBACK_CELL_MARKER = "## 📝 Your feedback"
        _FEEDBACK_INSTRUCTION = "*Type your message below. You can also edit any cells above. Save, then press Enter in the terminal.*"
        _GUI_MODE = os.environ.get("CELLVOYAGER_GUI_INTERACTIVE") == "1"
        _INTERVENE_EVERY = max(1, int(os.environ.get("CELLVOYAGER_INTERVENE_EVERY", "1")))

        _SUMMARY_MAX_BULLETS = 5

        def _extract_agent_summary(nb: Any) -> str:
            """Use an LLM to produce a 2-bullet summary of what the agent has done so far."""
            md_parts = []
            for cell in nb.cells:
                if cell.cell_type != "markdown":
                    continue
                src = cell.source
                text = "\n".join(src) if isinstance(src, list) else (src or "")
                text = text.strip()
                if text and not text.startswith(_FEEDBACK_CELL_MARKER):
                    md_parts.append(text)
            if not md_parts:
                return "No steps completed yet."
            context = "\n\n".join(md_parts[-6:])[:4000]
            prompt = (
                "Below are markdown cells from a single-cell transcriptomics analysis notebook. "
                "Summarize what the analysis has done so far in exactly 2 short bullet points. "
                "Just return the 2 bullet points, nothing else.\n\n" + context
            )
            anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
            if anthropic_key:
                try:
                    import anthropic
                    client = anthropic.Anthropic(api_key=anthropic_key)
                    resp = client.messages.create(
                        model="claude-haiku-4-5-20251001", max_tokens=150,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    if resp.content:
                        return resp.content[0].text.strip()
                except Exception:
                    pass
            openai_key = os.environ.get("OPENAI_API_KEY")
            if openai_key:
                try:
                    from openai import OpenAI
                    client = OpenAI(api_key=openai_key)
                    resp = client.chat.completions.create(
                        model="gpt-4o-mini", max_tokens=150,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    return (resp.choices[0].message.content or "").strip()
                except Exception:
                    pass
            return "No steps completed yet."

        @mcp.tool()
        def pause_for_user_review() -> dict[str, Any]:
            """Pause so the user can edit the notebook and/or add feedback.
            In terminal mode: adds a feedback cell. In GUI mode: uses the GUI feedback box only."""
            session = REGISTRY.current
            if not session:
                return {"ready": True, "user_feedback": ""}
            nb_path = str(session.path)
            force_gui_pause = _GUI_MODE and stop_request_path.exists()
            has_gui_request = _GUI_MODE and request_path.exists()

            # In GUI mode, a pause can be requested either by explicit pause file or by STOP.
            # Treat STOP as a forced pause even if the request file was cleared by the UI.
            if _GUI_MODE and (has_gui_request or force_gui_pause):
                if not has_gui_request:
                    request_path.write_text(nb_path, encoding="utf-8")
                agent_summary_path.unlink(missing_ok=True)
                def _write_summary_bg():
                    try:
                        agent_summary_path.write_text(_extract_agent_summary(session.nb), encoding="utf-8")
                    except Exception:
                        pass
                threading.Thread(target=_write_summary_bg, daemon=True).start()
                if response_path.exists():
                    feedback = response_path.read_text(encoding="utf-8").strip()
                    response_path.unlink(missing_ok=True)
                    request_path.unlink(missing_ok=True)
                    stop_request_path.unlink(missing_ok=True)  # Clear so agent doesn't pause again
                    if session.path.exists():
                        session.nb = nbf.read(session.path, as_version=4)
                    return {"ready": True, "user_feedback": feedback}
                response_path.unlink(missing_ok=True)
                # Fall through to poll loop below
            else:
                # Step counting: skip pause if not at an "intervene" step
                step_count = 0
                try:
                    if step_count_path.exists():
                        step_count = int(step_count_path.read_text(encoding="utf-8").strip() or "0")
                except (ValueError, OSError):
                    pass
                step_count += 1
                step_count_path.write_text(str(step_count), encoding="utf-8")

                if step_count % _INTERVENE_EVERY != 0:
                    return {"ready": True, "user_feedback": ""}

            response_path.unlink(missing_ok=True)
            request_path.write_text(nb_path, encoding="utf-8")
            # Clear stale summary and generate new one in background
            agent_summary_path.unlink(missing_ok=True)
            def _write_summary():
                try:
                    agent_summary_path.write_text(_extract_agent_summary(session.nb), encoding="utf-8")
                except Exception:
                    pass
            threading.Thread(target=_write_summary, daemon=True).start()
            _poll_interval = 0.05  # 50ms for responsive execute handling
            _iter_limit = None  # Wait indefinitely for user feedback in both modes
            _iter = 0
            while _iter_limit is None or _iter < _iter_limit:
                _iter += 1
                # User requested stop (cooperative): return immediately so agent exits cleanly
                if stop_request_path.exists() and not _GUI_MODE:
                    if session.path.exists():
                        session.nb = nbf.read(session.path, as_version=4)
                    return {"ready": True, "user_feedback": "__STOP__"}
                # GUI mode: process execute requests (run a cell, save, continue waiting)
                if _GUI_MODE and execute_request_path.exists():
                    try:
                        req = json.loads(execute_request_path.read_text(encoding="utf-8"))
                        idx = int(req.get("cell_index", -1))
                        session.nb = nbf.read(session.path, as_version=4)  # Reload user edits from disk
                        if 0 <= idx < len(session.nb.cells) and session.nb.cells[idx].cell_type == "code":
                            session.execute_cell(idx)
                        execute_request_path.unlink(missing_ok=True)
                    except Exception as e:
                        execute_request_path.unlink(missing_ok=True)
                        sys.stderr.write(f"[CellVoyager] Execute request failed: {e}\n")
                if response_path.exists():
                    response_feedback = response_path.read_text(encoding="utf-8").strip()
                    response_path.unlink(missing_ok=True)
                    if _GUI_MODE:
                        stop_request_path.unlink(missing_ok=True)  # Clear so agent doesn't pause again
                        # GUI mode: reload notebook from disk so agent gets user edits
                        if session.path.exists():
                            session.nb = nbf.read(session.path, as_version=4)
                        return {"ready": True, "user_feedback": response_feedback}
                    # Terminal mode: feedback comes directly from terminal
                    return {"ready": True, "user_feedback": response_feedback}
                time.sleep(_poll_interval)
            return {"ready": True, "user_feedback": "(timeout)" if not _GUI_MODE else ""}

    mcp.run(transport="stdio")


# -----------------------------------------------------------------------------
# Interactive mode: file-based handoff (main process has terminal, MCP runs in subprocess)
# -----------------------------------------------------------------------------

import threading

_PAUSE_REQUEST_FILE = ".cellvoyager_pause_request"
_PAUSE_RESPONSE_FILE = ".cellvoyager_pause_response"
_EXECUTE_REQUEST_FILE = ".cellvoyager_execute_request"
_STEP_COUNT_FILE = ".cellvoyager_step_count"
_AGENT_SUMMARY_FILE = ".cellvoyager_agent_summary"
_STOP_REQUEST_FILE = ".cellvoyager_stop_request"


class _InteractiveWatcher:
    """Background thread that watches for pause requests and prompts user at terminal."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.request_path = self.output_dir / _PAUSE_REQUEST_FILE
        self.response_path = self.output_dir / _PAUSE_RESPONSE_FILE
        self._stop = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop = True
        if self._thread:
            self._thread.join(timeout=2)

    def _watch_loop(self) -> None:
        while not self._stop:
            if self.request_path.exists():
                try:
                    nb_path = self.request_path.read_text(encoding="utf-8").strip()
                except Exception:
                    nb_path = "(unknown)"
                self.request_path.unlink(missing_ok=True)
                try:
                    print("\n=== PAUSE: Agent is waiting for your feedback ===", flush=True)
                    print(f"Notebook: {nb_path}", flush=True)
                    print("You can edit the notebook directly before continuing.", flush=True)
                    print("Enter feedback below (or press Enter to continue without feedback):", flush=True)
                    if Path("/dev/tty").exists():
                        tty = open("/dev/tty", "r")
                        print("> ", end="", flush=True)
                        feedback = tty.readline().rstrip()
                        tty.close()
                    else:
                        feedback = input("> ").strip()
                except Exception as e:
                    feedback = f"(error reading input: {e})"
                self.response_path.write_text(feedback, encoding="utf-8")
            time.sleep(0.2)


def _start_interactive_watcher(output_dir: Path) -> "_InteractiveWatcher":
    watcher = _InteractiveWatcher(output_dir)
    watcher.start()
    return watcher


# -----------------------------------------------------------------------------
# Simple file logger
# -----------------------------------------------------------------------------

class FileLogger:
    def __init__(self, log_path: str):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, tag: str, text: str) -> None:
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"[{now_str()}] [{tag}] {text}\n")

    def log_json(self, tag: str, payload: Any) -> None:
        self.log(tag, json.dumps(payload, ensure_ascii=False))


# -----------------------------------------------------------------------------
# Claude runner
# -----------------------------------------------------------------------------

class CellVoyagerClaudeRunner:
    """
    Minimal executor.

    Expected analysis dict:
        {
            "hypothesis": "...",
            "analysis_plan": ["step 1", "step 2", ...],
            "first_step_code": "..."
        }
    """

    def __init__(
        self,
        output_dir: str,
        h5ad_path: str,
        log_file: str,
        anthropic_api_key: str | None = None,
        adata_summary: str = "",
        paper_summary: str = "",
        coding_guidelines: str = "",
        max_turns: int = 40,
        max_iterations: int = 8,
        analysis_name: str = "cellvoyager",
        interactive_mode: bool = False,
        intervene_every: int = 1,
        execution_model: str | None = None,
    ):
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.h5ad_path = str(Path(h5ad_path).resolve())
        self.logger = FileLogger(log_file)
        self.adata_summary = adata_summary
        self.paper_summary = paper_summary
        self.coding_guidelines = coding_guidelines
        self.max_turns = max_turns
        self.max_iterations = max_iterations
        self.analysis_name = analysis_name
        self.interactive_mode = interactive_mode
        self.intervene_every = max(1, int(intervene_every))

        self.anthropic_api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required")
        self.execution_model = execution_model or None

    def _server_command(self) -> list[str]:
        return [sys.executable, str(Path(__file__).resolve()), "mcp-server"]

    def _write_initial_notebook(self, analysis: dict[str, Any], analysis_idx: int) -> Path:
        nb = new_notebook()

        hypothesis = analysis.get("hypothesis", "No hypothesis provided")
        plan = analysis.get("analysis_plan", [])

        nb.cells.append(new_markdown_cell(f"# Analysis\n\n**Hypothesis**: {hypothesis}"))

        _scvi_import = "" if os.getenv("CELLVOYAGER_DEMO_MODE", "0") == "1" else "import scvi\n"
        setup_code = f"""import scanpy as sc
{_scvi_import}import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("Loading data...")
adata = sc.read_h5ad(r'''{self.h5ad_path}''')
print(f"Loaded: {{adata.n_obs}} cells x {{adata.n_vars}} genes")
"""
        nb.cells.append(new_code_cell(setup_code))
        # Defer rendering the plan cell until setup finishes so the UI order is clear.
        nb.metadata["cellvoyager_initial_plan"] = plan
        nb.metadata["cellvoyager_plan_inserted"] = False

        notebook_path = self.output_dir / f"{self.analysis_name}_analysis_{analysis_idx + 1}.ipynb"
        with open(notebook_path, "w", encoding="utf-8") as f:
            nbf.write(nb, f)

        return notebook_path

    def _build_prompt(self, analysis: dict[str, Any], notebook_path: Path) -> str:
        hypothesis = analysis.get("hypothesis", "No hypothesis provided")
        plan = analysis.get("analysis_plan", [])
        first_step_code = strip_code_fences(analysis.get("first_step_code", ""))

        plan_text = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))

        interactive_block = ""
        if self.interactive_mode:
            gui_mode = os.environ.get("CELLVOYAGER_GUI_INTERACTIVE") == "1"
            if gui_mode:
                interactive_block = """
INTERACTIVE MODE (GUI): The user gives feedback via the GUI. The user can also edit the notebook in the GUI.
- Before each new step AND before every tool call, call check_user_stop. If stop_requested: true, do NOT add any more steps; stop immediately. If pause_requested: true, call pause_for_user_review immediately (do nothing else first).
- If execute_cell or insert_execute_code_cell returns paused_by_user: true (user clicked Stop), call pause_for_user_review immediately.
- After EVERY interpretation cell (including after step 1), you MUST call pause_for_user_review.
- If pause_for_user_review returns user_feedback exactly "__STOP__", the user stopped the analysis. Do NOT add any more steps; stop immediately.
- If pause_for_user_review returns user_feedback exactly "__FINISH__", the user has requested to finish early. Do NOT add any more code or analysis cells. Instead, add exactly one final markdown cell that concisely summarizes the key findings, visualizations, and conclusions from all analyses completed in the notebook so far, then stop immediately. Do not continue to the next analysis.
- The tool blocks. The user edits the notebook and/or types feedback in the GUI, then clicks Continue.
- When it returns, the tool provides user_feedback. You also get the updated notebook state (read_notebook to see changes).
- CRITICAL: Preserve all existing cells. The insert_cell tool in GUI mode ONLY appends — never pass a numeric index. Your new cells will always go at the end. This preserves user-inserted cells in their positions.
- Do NOT use delete_cell. Do NOT use overwrite_cell_source except to fix a code cell that YOU added and that failed to run — never overwrite cells the user may have added.
- Incorporate user_feedback and any user edits into your next steps.
- Do NOT add interpretation cells that merely summarize or repeat user-added code. User-added cells stay as-is; proceed with your next analysis step.
- Proceed with the next step only after pause_for_user_review returns.

"""
            else:
                interactive_block = """
INTERACTIVE MODE (TERMINAL): The user provides feedback directly in the terminal. The user can also edit the notebook between steps.
- Before each new step AND before every tool call, call check_user_stop. If stop_requested: true, do NOT add any more steps; stop immediately. If pause_requested: true, call pause_for_user_review immediately.
- If execute_cell or insert_execute_code_cell returns paused_by_user: true (user clicked Stop), call pause_for_user_review immediately.
- After EVERY interpretation cell (including after step 1), you MUST call pause_for_user_review.
- If pause_for_user_review returns user_feedback exactly "__STOP__", the user stopped the analysis. Do NOT add any more steps; stop immediately.
- If pause_for_user_review returns user_feedback exactly "__FINISH__", the user has requested to finish early. Do NOT add any more code or analysis cells. Instead, add exactly one final markdown cell that concisely summarizes the key findings, visualizations, and conclusions from all analyses completed in the notebook so far, then stop immediately. Do not continue to the next analysis.
- The tool blocks until the user enters feedback in the terminal. The user can also edit the notebook directly while paused. They press Enter to continue (with or without typed feedback).
- When it returns, the tool provides user_feedback from the terminal. After resuming, call read_notebook to pick up any edits the user made to the notebook.
- CRITICAL: Preserve all existing cells. Use insert_cell with index=None (append) so your new cells go at the end. Do NOT use delete_cell. Do NOT use overwrite_cell_source except to fix a code cell that YOU added and that failed to run.
- Incorporate user_feedback and any user edits into your next steps. Proceed with the next step only after pause_for_user_review returns.

"""

        return f"""
You are executing a single-cell transcriptomics analysis in a LIVE notebook.

You have custom notebook tools. Use them directly.
{interactive_block}

Required workflow:
1. Call use_notebook with notebook_path="{notebook_path}" — this automatically runs the setup cell (loads AnnData ONCE per kernel session). Do NOT add or run step 1 until use_notebook returns successfully.
2. Add the step 1 markdown summary cell and step 1 code cell (append to end), execute that new code cell, inspect with read_cell, then add a markdown interpretation cell (output summary + whether changing next steps + why).
   - IMPORTANT: AnnData is already loaded in memory as `adata` by setup. Reuse that in step 1 and all later steps. Do NOT call sc.read_h5ad again.
3. For every remaining step in the analysis plan:
   - add a markdown summary cell in this format:
     ## Step N summary - Short summary in header
     
     A more detailed 1-2 sentences explaining the motivation behind this step.
     (Use the word "summary" in the header, e.g. "## Step 2 summary - Load and QC data")
   - add a code cell implementing that step
   - execute it
   - inspect outputs with read_cell
   - if it fails, fix that same code cell with overwrite_cell_source and re-run
   - you may try at most 3 fixes for the same step
   - if still failing after 3 fixes, abandon that step and move to a different useful step
   - after every successful code execution, add a markdown interpretation cell (header like "## Step N — Interpretation: ...") that:
     (a) interprets the output (figures and printed text): what do the results show?
     (b) states whether you are changing the next steps or keeping the original plan
     (c) explains why: if changing, why the results justify a different approach; if keeping, why the current plan still holds
4. If the results suggest a better next step, update the plan in notebook markdown and continue.
5. End with a final markdown summary of findings.

CRITICAL — Step limit: You MUST complete the analysis in at most {self.max_iterations} interpretation steps (each step = one code cell + one interpretation markdown). Do NOT exceed this limit. Once you have reached step {self.max_iterations}, write your final summary and stop. Prioritize the most important steps if the plan is long.

Critical behavior:
- Actually execute code. Do not just describe what you would do.
- Use read_cell after running code so you can interpret outputs.
- After each code cell execution, add an interpretation markdown cell covering: what the output shows, whether you are adjusting your next steps, and why.
- Keep the notebook clean and readable.
- Do not use hidden scratchpads; put summaries/interpretations in markdown cells.
- Never re-load the dataset after setup; always reuse the existing `adata` object.

Notebook already contains:
- cell 0: hypothesis markdown
- cell 1: setup code
- initial analysis plan is inserted automatically only after setup finishes
- Step 1 cells do not exist yet; create them only after setup has finished.

Hypothesis:
{hypothesis}

Analysis plan:
{plan_text}

First step code template (insert this as your step 1 code, adapted to reuse existing `adata`):
```python
{first_step_code}
```

Context:
adata summary: {self.adata_summary[:3000]}

user context (dataset summary / past analyses / focus directions / biological background): {self.paper_summary[:3000]}

coding guidelines: {self.coding_guidelines[:3000]}
""".strip()

    def _log_stream_item(self, item: Any) -> None:
        """ Logs:
        - partial streamed text
        - tool starts
        - final assistant text
        - final result/error
        """
        event = getattr(item, "event", None)
        # Streaming event path
        if isinstance(event, dict):
            ev_type = event.get("type")

            if ev_type == "content_block_start":
                block = event.get("content_block", {})
                if block.get("type") == "tool_use":
                    self.logger.log_json("tool_start", {
                        "name": block.get("name"),
                        "input": block.get("input"),
                    })

            elif ev_type == "content_block_delta":
                delta = event.get("delta", {})
                if delta.get("type") == "text_delta":
                    text = delta.get("text", "")
                    if text:
                        self.logger.log("text_delta", text)
                        print(text, end="", flush=True)
                elif delta.get("type") == "input_json_delta":
                    pass  # Partial tool input; full block logged via assistant_tool_block

            elif ev_type == "message_delta":
                self.logger.log_json("message_delta", event)

            elif ev_type == "message_stop":
                self.logger.log("message_stop", "done")

            return

        # Non-streaming / final message path
        content = getattr(item, "content", None)
        if content:
            for block in content:
                text = getattr(block, "text", None)
                if text:
                    self.logger.log("assistant_text", text)

                name = getattr(block, "name", None)
                tool_input = getattr(block, "input", None)
                if name:
                    self.logger.log_json("assistant_tool_block", {
                        "name": name,
                        "input": tool_input,
                    })

        result = getattr(item, "result", None)
        if result:
            self.logger.log("result", str(result))

        is_error = getattr(item, "is_error", None)
        if is_error:
            self.logger.log("error", "Agent returned an error flag")

    def _build_resume_prompt(self, notebook_path: Path, user_feedback: str | None = None, extend: bool = False) -> str:
        """Prompt for resume mode: execute all code cells to restore kernel state, then pause (or extend)."""
        if extend:
            feedback_line = f"\n\nUser feedback:\n{user_feedback}" if user_feedback else ""
            return f"""
You are EXTENDING a completed single-cell analysis with additional steps. The notebook already exists.

Your tasks:
1. Call use_notebook with notebook_path="{notebook_path}" — this runs the setup cell (loads AnnData)
2. Before EVERY tool call, call check_user_stop. If pause_requested: true, call pause_for_user_review immediately.
3. Execute EVERY existing code cell in order to restore kernel state (skip markdown cells)
4. After restoring kernel state, ACTIVELY ADD NEW analysis steps — the user has asked you to extend this analysis further.
5. Add meaningful new analyses, visualizations, or investigations that build on the existing work.
6. After EVERY interpretation cell you add, call pause_for_user_review to let the user review and give feedback.
7. If pause_for_user_review returns user_feedback exactly "__STOP__", stop immediately.
8. If pause_for_user_review returns user_feedback exactly "__FINISH__", add one final summary markdown cell then stop.

CRITICAL: You must add new cells and new analyses. Only append (insert_cell with index=None). Do NOT delete or overwrite existing cells. Do NOT call sc.read_h5ad again (adata is already loaded).{feedback_line}
""".strip()
        feedback_section = (
            f"\n\nThe user has provided the following feedback to guide your continuation:\n{user_feedback}"
            if user_feedback else ""
        )
        return f"""
You are RESUMING a completed single-cell analysis. The notebook already exists with all cells.

Phase 1 — Restore kernel state:
1. Call use_notebook with notebook_path="{notebook_path}" — this automatically runs the setup cell (loads AnnData)
2. Before EVERY tool call, call check_user_stop. If pause_requested: true, call pause_for_user_review immediately.
3. Execute EVERY remaining code cell in the notebook (skip markdown cells) in order to restore kernel state.
4. After all code cells are executed, call pause_for_user_review to let the user review the notebook.

Phase 2 — Interactive extension (after the user clicks Continue):
INTERACTIVE MODE (GUI): The user gives feedback via the GUI. The user can also edit the notebook in the GUI.
- Before each new step AND before every tool call, call check_user_stop. If stop_requested: true, do NOT add any more steps; stop immediately. If pause_requested: true, call pause_for_user_review immediately (do nothing else first).
- If execute_cell or insert_execute_code_cell returns paused_by_user: true, call pause_for_user_review immediately.
- After EVERY interpretation cell you add, you MUST call pause_for_user_review.
- If pause_for_user_review returns user_feedback exactly "__STOP__", stop immediately.
- If pause_for_user_review returns user_feedback exactly "__FINISH__", add one final summary markdown cell then stop.
- The tool blocks. The user edits the notebook and/or types feedback in the GUI, then clicks Continue.
- When it returns, the tool provides user_feedback. You also get the updated notebook state (read_notebook to see changes).
- Incorporate user_feedback and any user edits into your next steps.
- Proceed with the next step only after pause_for_user_review returns.

Required workflow for each new step:
- Add a markdown summary cell in this format:
  ## Step N summary - Short summary in header

  A more detailed 1-2 sentences explaining the motivation behind this step.
  (Use the word "summary" in the header, e.g. "## Step 5 summary - Differential expression")
- Add a code cell implementing that step (insert_cell with index=None to append)
- Execute it, then inspect outputs with read_cell
- If it fails, fix with overwrite_cell_source and re-run (up to 3 attempts for the same step)
- If still failing after 3 fixes, abandon that step and move to a different useful step
- After every successful code execution, add a markdown interpretation cell (header like "## Step N — Interpretation: ...") that:
  (a) interprets the output: what do the results show?
  (b) states whether you are changing the next steps or keeping the plan
  (c) explains why

CRITICAL — Step limit: Complete at most {self.max_iterations} NEW interpretation steps. Once you reach {self.max_iterations} new steps, write a final summary markdown and stop.

CRITICAL: Only append new cells (insert_cell with index=None). Do NOT delete or overwrite existing cells. Do NOT call sc.read_h5ad again (adata is already loaded). Do NOT use delete_cell. Only use overwrite_cell_source to fix a code cell YOU just added that failed to run — never overwrite cells the user may have added.

Context:
adata summary: {self.adata_summary[:3000]}

user context (dataset summary / past analyses / focus directions / biological background): {self.paper_summary[:3000]}

coding guidelines: {self.coding_guidelines[:3000]}{feedback_section}
""".strip()

    def execute_idea(self, analysis: dict[str, Any], analysis_idx: int = 0) -> str:
        """
        Returns the notebook path.
        """
        from claude_agent_sdk import query, ClaudeAgentOptions
        notebook_path = self._write_initial_notebook(analysis, analysis_idx)
        prompt = self._build_prompt(analysis, notebook_path)

        self.logger.log("analysis_start", f"analysis_idx={analysis_idx} notebook={notebook_path}")
        self.logger.log("prompt", prompt)

        os.environ["ANTHROPIC_API_KEY"] = self.anthropic_api_key

        mcp_env = {}
        if self.interactive_mode:
            mcp_env["CELLVOYAGER_INTERACTIVE_MODE"] = "1"
            mcp_env["CELLVOYAGER_INTERACTIVE_OUTPUT_DIR"] = str(self.output_dir)
            mcp_env["CELLVOYAGER_INTERVENE_EVERY"] = str(self.intervene_every)
            if os.environ.get("CELLVOYAGER_GUI_INTERACTIVE") == "1":
                mcp_env["CELLVOYAGER_GUI_INTERACTIVE"] = "1"
        mcp_config = {
            "command": self._server_command()[0],
            "args": self._server_command()[1:],
            "env": mcp_env,
        }

        allowed_tools = [
            "mcp__jupyter__use_notebook",
            "mcp__jupyter__read_notebook",
            "mcp__jupyter__read_cell",
            "mcp__jupyter__insert_cell",
            "mcp__jupyter__overwrite_cell_source",
            "mcp__jupyter__delete_cell",
            "mcp__jupyter__execute_cell",
            "mcp__jupyter__insert_execute_code_cell",
            "mcp__jupyter__restart_kernel",
            "mcp__jupyter__check_user_stop",
        ]
        if self.interactive_mode:
            allowed_tools.append("mcp__jupyter__pause_for_user_review")

        options = ClaudeAgentOptions(
            mcp_servers={"jupyter": mcp_config},
            cwd=str(self.output_dir),
            permission_mode="bypassPermissions",
            allowed_tools=allowed_tools,
            include_partial_messages=True,
            max_turns=self.max_turns,
            **({"model": self.execution_model} if self.execution_model else {}),
        )

        interactive_watcher = None
        if self.interactive_mode and os.environ.get("CELLVOYAGER_GUI_INTERACTIVE") != "1":
            # Terminal-based watcher only when not running from GUI
            interactive_watcher = _start_interactive_watcher(self.output_dir)

        async def _run() -> None:
            async def prompt_gen():
                yield {
                    "type": "user",
                    "message": {
                        "role": "user",
                        "content": prompt,
                    },
                }

            async for item in query(prompt=prompt_gen(), options=options):
                self._log_stream_item(item)

        try:
            asyncio.run(_run())
        finally:
            if interactive_watcher is not None:
                interactive_watcher.stop()

        self.logger.log("analysis_complete", str(notebook_path))
        return str(notebook_path)

    def inter_analysis_pause(self, notebook_path: str, analysis_idx: int) -> str:
        """Block in GUI interactive mode until the user clicks Continue between analyses.

        Writes the standard pause-request file so the existing GUI pause UI appears,
        writes a clear agent-summary message, then polls for the response.
        Returns the user's feedback string, or "__STOP__" / "__FINISH__" as appropriate.
        Only active when interactive_mode=True and CELLVOYAGER_GUI_INTERACTIVE=1.
        """
        if not self.interactive_mode:
            return ""
        if os.environ.get("CELLVOYAGER_GUI_INTERACTIVE") != "1":
            return ""

        request_path = self.output_dir / _PAUSE_REQUEST_FILE
        response_path = self.output_dir / _PAUSE_RESPONSE_FILE
        stop_path = self.output_dir / _STOP_REQUEST_FILE
        summary_path = self.output_dir / _AGENT_SUMMARY_FILE

        response_path.unlink(missing_ok=True)
        stop_path.unlink(missing_ok=True)

        summary_path.write_text(
            f"✅ Analysis {analysis_idx + 1} complete.\n"
            f"Review the notebook above, optionally add feedback below, "
            f"then click **Continue** to start Analysis {analysis_idx + 2}.",
            encoding="utf-8",
        )
        request_path.write_text(str(notebook_path), encoding="utf-8")

        while True:
            if stop_path.exists():
                stop_path.unlink(missing_ok=True)
                request_path.unlink(missing_ok=True)
                return "__STOP__"
            if response_path.exists():
                feedback = response_path.read_text(encoding="utf-8").strip()
                response_path.unlink(missing_ok=True)
                request_path.unlink(missing_ok=True)
                stop_path.unlink(missing_ok=True)
                return feedback
            time.sleep(0.05)


class ClaudeJupyterExecutor(CellVoyagerClaudeRunner):
    """
    Adapter for agent_v2: accepts IdeaExecutor-style kwargs and adapts
    execute_idea to return past_analyses string instead of notebook path.
    """

    def __init__(self, *, logger, output_dir, h5ad_path, adata_summary, paper_summary,
                 coding_guidelines, analysis_name, anthropic_api_key,
                 max_iterations=8, max_turns=60, interactive_mode=False, intervene_every=1,
                 execution_model=None, **kwargs):
        log_file = getattr(logger, "log_file", str(Path(output_dir) / "claude_execution.log"))
        super().__init__(
            output_dir=output_dir,
            h5ad_path=h5ad_path,
            log_file=log_file,
            anthropic_api_key=anthropic_api_key,
            adata_summary=adata_summary or "",
            paper_summary=paper_summary or "",
            coding_guidelines=coding_guidelines or "",
            max_turns=max_turns,
            max_iterations=max_iterations,
            analysis_name=analysis_name,
            interactive_mode=interactive_mode,
            intervene_every=intervene_every,
            execution_model=execution_model,
        )

    def execute_idea(self, analysis: dict[str, Any], past_analyses: str = "",
                    analysis_idx: int = 0, seeded: bool = False) -> str:
        """Returns updated past_analyses string for agent_v2 compatibility."""
        notebook_path = super().execute_idea(analysis, analysis_idx)
        # Build a rich summary so the next analysis can be distinct
        hypothesis = analysis.get("hypothesis", "")
        plan = analysis.get("analysis_plan", [])
        plan_str = "\n".join(f"  - {step}" for step in plan) if plan else ""
        # Extract key findings from notebook markdown cells
        findings = ""
        try:
            nb = nbf.read(notebook_path, as_version=4)
            for cell in reversed(nb.cells):
                if cell.cell_type == "markdown":
                    src = cell.source if isinstance(cell.source, str) else "\n".join(cell.source)
                    first_line = src.strip().split("\n")[0].lower()
                    if "summary" in first_line or "finding" in first_line or "conclusion" in first_line:
                        findings = src.strip()[:500]
                        break
        except Exception:
            pass
        summary = f"Analysis {analysis_idx + 1}:\n  Hypothesis: {hypothesis}\n  Plan:\n{plan_str}\n"
        if findings:
            summary += f"  Key findings:\n  {findings}\n"
        return f"{past_analyses}{summary}\n"

    def resume_from_notebook(self, notebook_path: str, analysis_idx: int = 0, user_feedback: str | None = None, extend: bool = False) -> None:
        """Resume a completed analysis: restore kernel state by executing cells, then pause or extend."""
        from claude_agent_sdk import query, ClaudeAgentOptions
        nb_path = Path(notebook_path).resolve()
        prompt = self._build_resume_prompt(nb_path, user_feedback=user_feedback, extend=extend)

        print("Agent running (streaming output below)...", flush=True)
        self.logger.log("resume_start", str(nb_path))
        self.logger.log("prompt", prompt)

        os.environ["ANTHROPIC_API_KEY"] = self.anthropic_api_key

        mcp_env = {
            "CELLVOYAGER_INTERACTIVE_MODE": "1",
            "CELLVOYAGER_INTERACTIVE_OUTPUT_DIR": str(self.output_dir),
            "CELLVOYAGER_GUI_INTERACTIVE": "1",
            "CELLVOYAGER_INTERVENE_EVERY": str(self.intervene_every),
        }
        mcp_config = {
            "command": self._server_command()[0],
            "args": self._server_command()[1:],
            "env": mcp_env,
        }

        allowed_tools = [
            "mcp__jupyter__use_notebook",
            "mcp__jupyter__read_notebook",
            "mcp__jupyter__read_cell",
            "mcp__jupyter__insert_cell",
            "mcp__jupyter__overwrite_cell_source",
            "mcp__jupyter__delete_cell",
            "mcp__jupyter__execute_cell",
            "mcp__jupyter__insert_execute_code_cell",
            "mcp__jupyter__restart_kernel",
            "mcp__jupyter__check_user_stop",
            "mcp__jupyter__pause_for_user_review",
        ]

        options = ClaudeAgentOptions(
            mcp_servers={"jupyter": mcp_config},
            cwd=str(self.output_dir),
            permission_mode="bypassPermissions",
            allowed_tools=allowed_tools,
            include_partial_messages=True,
            max_turns=self.max_turns,
            **({"model": self.execution_model} if self.execution_model else {}),
        )

        async def _run() -> None:
            async def prompt_gen():
                yield {"type": "user", "message": {"role": "user", "content": prompt}}

            async for item in query(prompt=prompt_gen(), options=options):
                self._log_stream_item(item)

        asyncio.run(_run())
        self.logger.log("resume_complete", str(nb_path))

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "mcp-server":
        run_mcp_server()
    else:
        print(
            "This file is meant to be imported and used as a library.\n"
            "It also runs the MCP server when invoked as:\n"
            f"  python {Path(__file__).name} mcp-server"
        )
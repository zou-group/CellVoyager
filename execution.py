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

        # Make inline plots show up in notebook outputs and ensure cwd is correct.
        bootstrap = (
            "%matplotlib inline\n"
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
        bootstrap = (
            "%matplotlib inline\n"
            "import os\n"
            f"os.chdir(r'''{self.path.parent}''')\n"
        )
        self._execute_source(bootstrap)

    def save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            nbf.write(self.nb, f)

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
        result = self._execute_source(source)

        cell["outputs"] = result["outputs"]
        cell["execution_count"] = result["execution_count"]
        self.save()

        return {
            "ok": result["ok"],
            "cell_index": index,
            "execution_count": result["execution_count"],
            "output_preview": result["preview"],
            "error": result["error"],
        }

    def insert_execute_code_cell(self, index: int | None, source: str) -> dict[str, Any]:
        inserted = self.insert_cell(index=index, cell_type="code", source=source)
        idx = inserted["cell_index"]
        executed = self.execute_cell(idx)
        return {
            "ok": executed["ok"],
            "cell_index": idx,
            "execution_count": executed["execution_count"],
            "output_preview": executed["output_preview"],
            "error": executed["error"],
        }

    def _execute_source(self, source: str) -> dict[str, Any]:
        msg_id = self.kc.execute(source, allow_stdin=False, stop_on_error=False)

        outputs = []
        execution_count = None
        error_text = None

        while True:
            msg = self.kc.get_iopub_msg(timeout=300)
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
        ok = error_text is None

        return {
            "ok": ok,
            "outputs": outputs,
            "execution_count": execution_count,
            "preview": preview,
            "error": error_text,
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

    @mcp.tool()
    def use_notebook(notebook_path: str) -> dict[str, Any]:
        session = REGISTRY.use_notebook(notebook_path)
        return {
            "ok": True,
            "notebook_path": str(session.path),
            "num_cells": len(session.nb.cells),
        }

    @mcp.tool()
    def read_notebook() -> dict[str, Any]:
        return REGISTRY.require_current().read_notebook()

    @mcp.tool()
    def read_cell(index: int) -> dict[str, Any]:
        return REGISTRY.require_current().read_cell(index)

    @mcp.tool()
    def insert_cell(index: int | None, cell_type: str, source: str) -> dict[str, Any]:
        # In GUI interactive mode, always append to preserve user-inserted cell positions
        if os.environ.get("CELLVOYAGER_GUI_INTERACTIVE") == "1":
            index = None
        return REGISTRY.require_current().insert_cell(index=index, cell_type=cell_type, source=source)

    @mcp.tool()
    def overwrite_cell_source(index: int, source: str) -> dict[str, Any]:
        return REGISTRY.require_current().overwrite_cell_source(index=index, source=source)

    @mcp.tool()
    def delete_cell(index: int) -> dict[str, Any]:
        return REGISTRY.require_current().delete_cell(index=index)

    @mcp.tool()
    def execute_cell(index: int) -> dict[str, Any]:
        return REGISTRY.require_current().execute_cell(index=index)

    @mcp.tool()
    def insert_execute_code_cell(index: int | None, source: str) -> dict[str, Any]:
        if os.environ.get("CELLVOYAGER_GUI_INTERACTIVE") == "1":
            index = None
        return REGISTRY.require_current().insert_execute_code_cell(index=index, source=source)

    @mcp.tool()
    def restart_kernel() -> dict[str, Any]:
        session = REGISTRY.require_current()
        session.restart_kernel()
        return {"ok": True, "notebook_path": str(session.path)}

    if os.environ.get("CELLVOYAGER_INTERACTIVE_MODE") == "1":
        output_dir = Path(os.environ.get("CELLVOYAGER_INTERACTIVE_OUTPUT_DIR", "."))
        request_path = output_dir / _PAUSE_REQUEST_FILE
        response_path = output_dir / _PAUSE_RESPONSE_FILE
        execute_request_path = output_dir / _EXECUTE_REQUEST_FILE
        step_count_path = output_dir / _STEP_COUNT_FILE
        agent_summary_path = output_dir / _AGENT_SUMMARY_FILE

        _FEEDBACK_CELL_MARKER = "## 📝 Your feedback"
        _FEEDBACK_INSTRUCTION = "*Type your message below. You can also edit any cells above. Save, then press Enter in the terminal.*"
        _GUI_MODE = os.environ.get("CELLVOYAGER_GUI_INTERACTIVE") == "1"
        _INTERVENE_EVERY = max(1, int(os.environ.get("CELLVOYAGER_INTERVENE_EVERY", "1")))

        _SUMMARY_MAX_BULLETS = 5

        def _extract_agent_summary(nb: Any) -> str:
            """Build a concise bullet-point summary of what the agent has done so far."""

            def _first_bullets(block: str, n: int = _SUMMARY_MAX_BULLETS) -> list[str]:
                """Extract first n bullet items, stripped of markdown."""
                bullets = []
                for line in block.split("\n"):
                    m = re.match(r"^\s*[-*]\s+(.+)|^\s*\d+[.)]\s+(.+)", line)
                    if m:
                        item = (m.group(1) or m.group(2) or "").strip().rstrip(".:")
                        item = re.sub(r"\*\*([^*]+)\*\*", r"\1", item)
                        if len(item) > 8:
                            bullets.append(item)
                            if len(bullets) >= n:
                                break
                return bullets

            def _to_bullets(items: list[str]) -> str:
                """Format as '- item\\n' for display."""
                return "\n".join(f"- {it}" for it in items)

            last_interpretation = None
            last_step_summary = None
            for cell in nb.cells:
                if cell.cell_type != "markdown":
                    continue
                src = cell.source
                text = "\n".join(src) if isinstance(src, list) else (src or "")
                text = text.strip()
                if not text or text.startswith(_FEEDBACK_CELL_MARKER):
                    continue
                first_line = text.split("\n")[0].lower()
                if "— Interpretation" in text or "Interpretation:" in text:
                    last_interpretation = text
                elif "step" in first_line or "steps" in first_line:
                    if "interpretation" not in first_line:
                        last_step_summary = text
            candidate = last_interpretation or last_step_summary
            if not candidate:
                return "No steps completed yet."
            for pattern in (r"\*\*Plan going forward[^*]*\*\*[:\s]*\n?(.+?)(?=\n\n|\n\*\*|$)", r"\*\*What the output shows[^*]*\*\*[:\s]*\n?(.+?)(?=\n\n|\n\*\*|$)", r"\*\*Output summary[^*]*\*\*[:\s]*\n?(.+?)(?=\n\n|\n\*\*|$)"):
                m = re.search(pattern, candidate, re.DOTALL)
                if m:
                    block = m.group(1).strip()
                    bullets = _first_bullets(block)
                    if bullets:
                        return _to_bullets(bullets)
            no_header = re.sub(r"^#+\s+[^\n]+\n*", "", candidate)
            no_header = re.sub(r"\*\*[^*]+\*\*[:\s]*", "", no_header)
            bullets = _first_bullets(no_header)
            if bullets:
                return _to_bullets(bullets)
            first_line = no_header.split("\n")[0].strip()
            if len(first_line) > 10 and not first_line.endswith(":"):
                return f"- {first_line[:300]}"
            return "No steps completed yet."

        @mcp.tool()
        def pause_for_user_review() -> dict[str, Any]:
            """Pause so the user can edit the notebook and/or add feedback.
            In terminal mode: adds a feedback cell. In GUI mode: uses the GUI feedback box only."""
            session = REGISTRY.current
            if not session:
                return {"ready": True, "user_feedback": ""}
            nb_path = str(session.path)

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

            if not _GUI_MODE:
                # Add feedback cell so the notebook serves as the UI in terminal mode
                feedback_cell_source = f"""{_FEEDBACK_CELL_MARKER}

{_FEEDBACK_INSTRUCTION}


"""
                session.insert_cell(index=None, cell_type="markdown", source=feedback_cell_source)
            response_path.unlink(missing_ok=True)
            request_path.write_text(nb_path, encoding="utf-8")
            agent_summary_path.write_text(_extract_agent_summary(session.nb), encoding="utf-8")
            _poll_interval = 0.05  # 50ms for responsive execute handling
            _iter_limit = None if _GUI_MODE else 1200  # GUI: wait indefinitely; terminal: 60 sec
            _iter = 0
            while _iter_limit is None or _iter < _iter_limit:
                _iter += 1
                # GUI mode: process execute requests (run a cell, save, continue waiting)
                if _GUI_MODE and execute_request_path.exists():
                    try:
                        req = json.loads(execute_request_path.read_text(encoding="utf-8"))
                        execute_request_path.unlink(missing_ok=True)
                        idx = int(req.get("cell_index", -1))
                        session.nb = nbf.read(session.path, as_version=4)  # Reload user edits from disk
                        if 0 <= idx < len(session.nb.cells) and session.nb.cells[idx].cell_type == "code":
                            session.execute_cell(idx)
                    except Exception as e:
                        sys.stderr.write(f"[CellVoyager] Execute request failed: {e}\n")
                if response_path.exists():
                    response_feedback = response_path.read_text(encoding="utf-8").strip()
                    response_path.unlink(missing_ok=True)
                    if _GUI_MODE:
                        # GUI mode: reload notebook from disk so agent gets user edits
                        if session.path.exists():
                            session.nb = nbf.read(session.path, as_version=4)
                        return {"ready": True, "user_feedback": response_feedback}
                    # Terminal mode: reload notebook, prefer feedback from the cell
                    if session.path.exists():
                        nb = nbf.read(session.path, as_version=4)
                        session.nb = nb
                    for cell in session.nb.cells:
                        if cell.cell_type == "markdown" and cell.source.strip().startswith(_FEEDBACK_CELL_MARKER):
                            raw = cell.source.replace(_FEEDBACK_CELL_MARKER, "").replace(_FEEDBACK_INSTRUCTION, "").strip()
                            content = "\n".join(l for l in raw.split("\n") if l.strip()).strip()
                            if content or response_feedback:
                                return {"ready": True, "user_feedback": content or response_feedback}
                            break
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
                    print("\n=== PAUSE: The notebook is your UI ===", flush=True)
                    print(f"Notebook: {nb_path}", flush=True)
                    print("Edit cells and/or type feedback in the '📝 Your feedback' cell. Save, then press Enter here to continue.", flush=True)
                    if Path("/dev/tty").exists():
                        tty = open("/dev/tty", "r")
                        tty.readline()
                        print("Type feedback for the agent (optional, Enter to skip): ", end="", flush=True)
                        feedback = tty.readline().rstrip()
                        tty.close()
                    else:
                        input()
                        feedback = input("Type feedback for the agent (optional, Enter to skip): ").strip()
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
        analysis_name: str = "cellvoyager",
        interactive_mode: bool = False,
        intervene_every: int = 1,
    ):
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.h5ad_path = str(Path(h5ad_path).resolve())
        self.logger = FileLogger(log_file)
        self.adata_summary = adata_summary
        self.paper_summary = paper_summary
        self.coding_guidelines = coding_guidelines
        self.max_turns = max_turns
        self.analysis_name = analysis_name
        self.interactive_mode = interactive_mode
        self.intervene_every = max(1, int(intervene_every))

        self.anthropic_api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required")

    def _server_command(self) -> list[str]:
        return [sys.executable, str(Path(__file__).resolve()), "mcp-server"]

    def _write_initial_notebook(self, analysis: dict[str, Any], analysis_idx: int) -> Path:
        nb = new_notebook()

        hypothesis = analysis["hypothesis"]
        plan = analysis["analysis_plan"]
        first_step_code = strip_code_fences(analysis["first_step_code"])

        nb.cells.append(new_markdown_cell(f"# Analysis\n\n**Hypothesis**: {hypothesis}"))

        setup_code = f"""import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("Loading data...")
adata = sc.read_h5ad(r'''{self.h5ad_path}''')
print(f"Loaded: {{adata.n_obs}} cells x {{adata.n_vars}} genes")
"""
        nb.cells.append(new_code_cell(setup_code))

        plan_md = "# Analysis Plan\n\n" + "\n".join(
            f"{i+1}. {step}" for i, step in enumerate(plan)
        )
        nb.cells.append(new_markdown_cell(plan_md))

        step1_summary = plan[0] if plan else "Execute the first analysis step."
        nb.cells.append(new_markdown_cell(f"## Step 1 summary\n\n{step1_summary}"))
        nb.cells.append(new_code_cell(first_step_code))

        notebook_path = self.output_dir / f"{self.analysis_name}_analysis_{analysis_idx + 1}.ipynb"
        with open(notebook_path, "w", encoding="utf-8") as f:
            nbf.write(nb, f)

        return notebook_path

    def _build_prompt(self, analysis: dict[str, Any], notebook_path: Path) -> str:
        hypothesis = analysis["hypothesis"]
        plan = analysis["analysis_plan"]
        first_step_code = strip_code_fences(analysis["first_step_code"])

        plan_text = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))

        interactive_block = ""
        if self.interactive_mode:
            gui_mode = os.environ.get("CELLVOYAGER_GUI_INTERACTIVE") == "1"
            if gui_mode:
                interactive_block = """
INTERACTIVE MODE (GUI): The user gives feedback via the GUI. The user can also edit the notebook in the GUI.
- After EVERY interpretation cell (including after step 1), you MUST call pause_for_user_review.
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
INTERACTIVE MODE (TERMINAL): The notebook is the user's UI. The user edits the notebook and types feedback in the feedback cell.
- After EVERY interpretation cell (including after step 1), you MUST call pause_for_user_review.
- The tool adds a "## 📝 Your feedback" cell and blocks. The user edits the notebook, types in that cell, saves, then presses Enter in the terminal to continue.
- When it returns, the tool provides user_feedback (extracted from the feedback cell). You also get the updated notebook state.
- CRITICAL: Preserve all existing cells. Use insert_cell with index=None (append) so your new cells go at the end. Do NOT use delete_cell. Do NOT use overwrite_cell_source except to fix a code cell that YOU added and that failed to run — never overwrite cells the user may have added. Incorporate user_feedback and any user edits (new cells, modified code) into your next steps. Do NOT add interpretation cells that merely summarize or repeat user-added code.
- Proceed with the next step only after pause_for_user_review returns.

"""

        return f"""
You are executing a single-cell transcriptomics analysis in a LIVE notebook.

You have custom notebook tools. Use them directly.
{interactive_block}

Required workflow:
1. Call use_notebook with notebook_path="{notebook_path}"
2. Execute the setup cell at index 1
3. Execute the first analysis code cell at index 4, inspect with read_cell, then add a markdown interpretation cell (output summary + whether changing next steps + why)
4. For every remaining step in the analysis plan:
   - add a markdown summary cell with header "## Step N summary" (use the word "summary"—e.g. "## Step 2 summary", "## Steps 3 & 4 summary")
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
5. If the results suggest a better next step, update the plan in notebook markdown and continue.
6. End with a final markdown summary of findings.

Critical behavior:
- Actually execute code. Do not just describe what you would do.
- Use read_cell after running code so you can interpret outputs.
- After each code cell execution, add an interpretation markdown cell covering: what the output shows, whether you are adjusting your next steps, and why.
- Keep the notebook clean and readable.
- Do not use hidden scratchpads; put summaries/interpretations in markdown cells.

Notebook already contains:
- cell 0: hypothesis markdown
- cell 1: setup code
- cell 2: initial analysis plan
- cell 3: step 1 summary markdown
- cell 4: first step code

Hypothesis:
{hypothesis}

Analysis plan:
{plan_text}

First step code (already in cell 4):
```python
{first_step_code}
```

Context:
adata summary: {self.adata_summary[:3000]}

paper summary: {self.paper_summary[:3000]}

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


class ClaudeJupyterExecutor(CellVoyagerClaudeRunner):
    """
    Adapter for agent_v2: accepts IdeaExecutor-style kwargs and adapts
    execute_idea to return past_analyses string instead of notebook path.
    """

    def __init__(self, *, logger, output_dir, h5ad_path, adata_summary, paper_summary,
                 coding_guidelines, analysis_name, anthropic_api_key,
                 max_iterations=40, max_turns=60, interactive_mode=False, intervene_every=1, **kwargs):
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
            analysis_name=analysis_name,
            interactive_mode=interactive_mode,
            intervene_every=intervene_every,
        )

    def execute_idea(self, analysis: dict[str, Any], past_analyses: str = "",
                    analysis_idx: int = 0, seeded: bool = False) -> str:
        """Returns updated past_analyses string for agent_v2 compatibility."""
        notebook_path = super().execute_idea(analysis, analysis_idx)
        return f"{past_analyses}Analysis {analysis_idx + 1}: Completed. Notebook: {notebook_path}\n\n"

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "mcp-server":
        run_mcp_server()
    else:
        print(
            "This file is meant to be imported and used as a library.\n"
            "It also runs the MCP server when invoked as:\n"
            f"  python {Path(__file__).name} mcp-server"
        )
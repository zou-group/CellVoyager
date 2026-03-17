"""
Custom Jupyter MCP tools for Claude Agent SDK.

Provides in-process tools to read, edit, and execute notebook cells using
nbformat and jupyter_client—no external jupyter-mcp-server or Jupyter Lab required.
Works entirely within the CellVoyager process.
"""
import os
import asyncio
from pathlib import Path
from typing import Any

import nbformat as nbf
from jupyter_client import KernelManager

# Lazy SDK import
_sdk = None


def _get_sdk():
    global _sdk
    if _sdk is None:
        try:
            from claude_agent_sdk import tool, create_sdk_mcp_server
            _sdk = {"tool": tool, "create_sdk_mcp_server": create_sdk_mcp_server}
        except ImportError as e:
            raise ImportError(
                "claude-agent-sdk required for jupyter_tools. pip install claude-agent-sdk"
            ) from e
    return _sdk


def _outputs_to_text(outputs: list) -> str:
    """Convert cell outputs to a readable text summary."""
    parts = []
    for out in outputs:
        if isinstance(out, dict):
            ot = out.get("output_type", "")
            if ot == "stream":
                parts.append(out.get("text", ""))
            elif ot == "execute_result":
                data = out.get("data", {})
                if "text/plain" in data:
                    parts.append(str(data["text/plain"]))
                elif "image/png" in data:
                    parts.append("[Image output]")
            elif ot == "display_data":
                data = out.get("data", {})
                if "text/plain" in data:
                    parts.append(str(data["text/plain"]))
                elif "image/png" in data:
                    parts.append("[Image output]")
            elif ot == "error":
                parts.append(f"Error: {out.get('ename', '')}: {out.get('evalue', '')}")
        else:
            if hasattr(out, "output_type"):
                if out.output_type == "stream":
                    parts.append(getattr(out, "text", ""))
                elif out.output_type == "execute_result":
                    data = getattr(out, "data", {})
                    if "text/plain" in data:
                        parts.append(str(data["text/plain"]))
                    else:
                        parts.append("[Execute result]")
                elif out.output_type == "error":
                    parts.append(f"Error: {getattr(out, 'ename', '')}: {getattr(out, 'evalue', '')}")
    return "\n".join(parts) if parts else "(no output)"


def _run_cell_sync(kernel_client, code: str, timeout: int = 300,
                    kernel_manager=None, kill_file_path=None):
    """Execute code in kernel (sync). Returns (success, outputs_list, error_msg).

    If kernel_manager and kill_file_path are provided, checks for a kill signal
    file and interrupts the kernel when found.
    """
    msg_id = kernel_client.execute(code)
    outputs = []

    while True:
        try:
            msg = kernel_client.get_iopub_msg(timeout=2)
        except Exception:
            # Check for kill signal during idle waits
            if kernel_manager and kill_file_path:
                kp = Path(kill_file_path)
                if kp.exists():
                    try:
                        kp.unlink(missing_ok=True)
                        kernel_manager.interrupt_kernel()
                    except Exception:
                        pass
            continue

        msg_type = msg["msg_type"]
        content = msg["content"]

        if msg_type == "status" and content.get("execution_state") == "idle":
            break

        if msg.get("parent_header", {}).get("msg_id") != msg_id:
            continue

        if msg_type == "stream":
            outputs.append({"output_type": "stream", "name": content.get("name", "stdout"), "text": content.get("text", "")})
        elif msg_type == "execute_result":
            outputs.append({
                "output_type": "execute_result",
                "data": content.get("data", {}),
                "execution_count": content.get("execution_count"),
            })
        elif msg_type == "display_data":
            outputs.append({
                "output_type": "display_data",
                "data": content.get("data", {}),
                "metadata": content.get("metadata", {}),
            })
        elif msg_type == "error":
            outputs.append({
                "output_type": "error",
                "ename": content.get("ename", ""),
                "evalue": content.get("evalue", ""),
                "traceback": content.get("traceback", []),
            })

    for o in outputs:
        if isinstance(o, dict) and o.get("output_type") == "error":
            return False, outputs, f"{o.get('ename', '')}: {o.get('evalue', '')}"
    return True, outputs, None


def create_jupyter_mcp_server(
    notebook_path: str,
    output_dir: str,
    kernel_manager: KernelManager,
    kernel_client,
):
    """
    Create an in-process MCP server with Jupyter notebook tools.

    Args:
        notebook_path: Path to the .ipynb file (can be relative to output_dir)
        output_dir: Working directory for relative paths
        kernel_manager: Started KernelManager
        kernel_client: Connected kernel client for execution

    Returns:
        MCP server instance to pass to ClaudeAgentOptions (type "sdk")
    """
    sdk = _get_sdk()
    tool = sdk["tool"]
    create_sdk_mcp_server = sdk["create_sdk_mcp_server"]

    def _resolve_path(path: str) -> str:
        if os.path.isabs(path):
            return path
        return os.path.join(output_dir, path)

    @tool(
        "notebook_read",
        "Read the full notebook. Returns each cell's type and source. Use this to inspect current state before editing.",
        {"notebook_path": str},
    )
    async def notebook_read(args: dict[str, Any]) -> dict[str, Any]:
        path = _resolve_path(args["notebook_path"])
        if not os.path.exists(path):
            return {"content": [{"type": "text", "text": f"Error: Notebook not found at {path}"}]}
        nb = nbf.read(path, as_version=4)
        lines = []
        for i, cell in enumerate(nb.cells):
            ct = cell.cell_type
            src = cell.source if isinstance(cell.source, str) else "".join(cell.source)
            lines.append(f"--- Cell {i} ({ct}) ---\n{src}")
        return {"content": [{"type": "text", "text": "\n\n".join(lines)}]}

    @tool(
        "notebook_add_cell",
        "Add a new cell to the notebook. cell_type must be 'code' or 'markdown'. position is 0-based (use -1 for end).",
        {"notebook_path": str, "cell_type": str, "source": str},
    )
    async def notebook_add_cell(args: dict[str, Any]) -> dict[str, Any]:
        path = _resolve_path(args["notebook_path"])
        ct = args.get("cell_type", "code").lower()
        src = args.get("source", "")
        pos = args.get("position", -1)  # -1 = append at end
        if ct not in ("code", "markdown"):
            return {"content": [{"type": "text", "text": "Error: cell_type must be 'code' or 'markdown'"}]}
        if not os.path.exists(path):
            return {"content": [{"type": "text", "text": f"Error: Notebook not found at {path}"}]}
        nb = nbf.read(path, as_version=4)
        if ct == "code":
            new_cell = nbf.v4.new_code_cell(src)
        else:
            new_cell = nbf.v4.new_markdown_cell(src)
        if pos < 0 or pos >= len(nb.cells):
            nb.cells.append(new_cell)
        else:
            nb.cells.insert(pos, new_cell)
        with open(path, "w", encoding="utf-8") as f:
            nbf.write(nb, f)
        idx = len(nb.cells) - 1 if pos < 0 else pos
        return {"content": [{"type": "text", "text": f"Added {ct} cell at index {idx}"}]}

    @tool(
        "notebook_overwrite_cell",
        "Overwrite the source of an existing cell by index (0-based). Use notebook_read first to see cell indices.",
        {"notebook_path": str, "cell_index": int, "source": str},
    )
    async def notebook_overwrite_cell(args: dict[str, Any]) -> dict[str, Any]:
        path = _resolve_path(args["notebook_path"])
        idx = args["cell_index"]
        src = args.get("source", "")
        if not os.path.exists(path):
            return {"content": [{"type": "text", "text": f"Error: Notebook not found at {path}"}]}
        nb = nbf.read(path, as_version=4)
        if idx < 0 or idx >= len(nb.cells):
            return {"content": [{"type": "text", "text": f"Error: Invalid cell_index {idx}"}]}
        nb.cells[idx].source = src
        with open(path, "w", encoding="utf-8") as f:
            nbf.write(nb, f)
        return {"content": [{"type": "text", "text": f"Overwrote cell {idx}"}]}

    @tool(
        "notebook_execute_cell",
        "Execute a code cell by index (0-based). Returns stdout, results, or error. Run setup cells first (e.g. 0, 1) before analysis cells.",
        {"notebook_path": str, "cell_index": int},
    )
    async def notebook_execute_cell(args: dict[str, Any]) -> dict[str, Any]:
        path = _resolve_path(args["notebook_path"])
        idx = args["cell_index"]
        if not os.path.exists(path):
            return {"content": [{"type": "text", "text": f"Error: Notebook not found at {path}"}]}
        nb = nbf.read(path, as_version=4)
        if idx < 0 or idx >= len(nb.cells):
            return {"content": [{"type": "text", "text": f"Error: Invalid cell_index {idx}"}]}
        cell = nb.cells[idx]
        if cell.cell_type != "code":
            return {"content": [{"type": "text", "text": f"Error: Cell {idx} is not a code cell"}]}
        code = cell.source if isinstance(cell.source, str) else "".join(cell.source)
        _kill_path = os.path.join(output_dir, ".cellvoyager_kill_cell")
        _running_path = os.path.join(output_dir, ".cellvoyager_running_cell")
        # Signal to the GUI that this cell is executing
        import json as _json, time as _time
        try:
            Path(_running_path).write_text(_json.dumps({
                "cell_index": idx,
                "started_at": _time.time(),
            }), encoding="utf-8")
        except Exception:
            pass
        loop = asyncio.get_event_loop()
        success, outputs, err = await loop.run_in_executor(
            None, lambda: _run_cell_sync(kernel_client, code,
                                          kernel_manager=kernel_manager,
                                          kill_file_path=_kill_path)
        )
        # Clear the running signal
        try:
            Path(_running_path).unlink(missing_ok=True)
        except Exception:
            pass
        # Attach outputs to cell and save
        clean = []
        for o in outputs:
            if isinstance(o, dict):
                if o.get("output_type") == "stream":
                    clean.append(nbf.v4.new_output("stream", name=o.get("name", "stdout"), text=o.get("text", "")))
                elif o.get("output_type") == "execute_result":
                    clean.append(nbf.v4.new_output("execute_result", data=o.get("data", {}), execution_count=o.get("execution_count")))
                elif o.get("output_type") == "display_data":
                    clean.append(nbf.v4.new_output("display_data", data=o.get("data", {}), metadata=o.get("metadata", {})))
                elif o.get("output_type") == "error":
                    clean.append(nbf.v4.new_output("error", ename=o.get("ename", ""), evalue=o.get("evalue", ""), traceback=o.get("traceback", [])))
        cell.outputs = clean
        with open(path, "w", encoding="utf-8") as f:
            nbf.write(nb, f)
        summary = _outputs_to_text(outputs)
        if not success:
            return {"content": [{"type": "text", "text": f"Execution failed:\n{summary}"}]}
        return {"content": [{"type": "text", "text": f"Execution succeeded:\n{summary}"}]}

    return create_sdk_mcp_server(
        name="jupyter",
        version="1.0.0",
        tools=[notebook_read, notebook_add_cell, notebook_overwrite_cell, notebook_execute_cell],
    )

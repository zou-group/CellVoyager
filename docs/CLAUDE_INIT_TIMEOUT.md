# Claude Agent SDK: "Control request timeout: initialize"

## Root Cause (SDK MCP)

The timeout occurred when using **SDK MCP** (in-process custom tools). The CLI would start, reach "Policy limits: Fetched successfully", but never send the initialize response back to the Python SDK. The SDK MCP handshake appears buggy or incompatible between the Python SDK and bundled CLI.

Tests showed:
- **Without SDK MCP**: Init completes in ~2 seconds
- **With SDK MCP**: Init times out after 60+ seconds

An earlier EPERM error (`~/.claude.json.lock`) was mitigated by setting `CLAUDE_CONFIG_HOME` to a writable dir.

## Solution: Use External MCP

CellVoyager now uses the **external jupyter-mcp-server** (stdio) instead of SDK MCP. The CLI spawns it as a subprocess and communicates via stdin/stdout—this path works reliably.

Requirements:
- `pip install jupyter-mcp-server` (or `uvx jupyter-mcp-server@latest`)
- Jupyter Lab running (CellVoyager auto-starts it)
- `ANTHROPIC_API_KEY` set

## Fallback

Use `--execution-mode legacy` to avoid the Claude SDK entirely; uses OpenAI + programmatic kernel.

## References

- Issue #387: CLI exit errors not propagated (fixed in later SDK)
- SDK: claude-agent-sdk 0.1.44, bundled CLI 2.1.59

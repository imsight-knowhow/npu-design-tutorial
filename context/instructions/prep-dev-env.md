# Dev Environment Notes (Windows host)

This repository uses `pixi` for Python execution and package management.

## Python (use `pixi`, not system Python)

- Do **not** use `python`, `pip`, or `venv` from the system installation for this repo.
- Prefer:
  - `pixi shell` to enter the environment
  - `pixi run <cmd>` to run a command in the environment

Common examples:

- Run Python: `pixi run python -V`
- Install editable: `pixi run python -m pip install -e .`
- Run tests: `pixi run pytest -q`

## Temporary scripts and outputs

- Put one-off scripts and any generated outputs under `tmp/<subdir>/` (create a new subdir per task).

## Document processing libraries available (in the `pixi` env)

These are available in the workspace `pixi` environment:

- `python-docx` (import as `docx`) for `.docx`
- `PyMuPDF` (import as `fitz`) for PDF parsing/rendering
- `mdutils` for programmatically generating Markdown

Quick import check:

- `pixi run python -c "import docx, mdutils, fitz; print('ok')"`

## Host tools available

The host system provides:

- `pandoc` (e.g. `pandoc --version`)
- LaTeX toolchain (`pdflatex --version` / MiKTeX)

## `uv` tools available (global)

`uv` is installed (e.g. `uv --version`) and the following `uv tool` entries are available on this machine:

- `aider-chat` (`aider.exe`)
- `awscli` (`aws`, `aws.cmd`, `aws_bash_completer`, `aws_completer`, `aws_zsh_completer.sh`)
- `blender-remote` (`blender-remote-cli.exe`, `blender-remote.exe`)
- `claude-monitor` (`ccm.exe`, `ccmonitor.exe`, `claude-code-monitor.exe`, `claude-monitor.exe`, `cmonitor.exe`)
- `codex-as-mcp` (`codex-as-mcp.exe`)
- `gdown` (`gdown.exe`)
- `httpie` (`http.exe`, `httpie.exe`, `https.exe`)
- `litellm` (`litellm-proxy.exe`, `litellm.exe`)
- `llm` (`llm.exe`)
- `llm-anygate` (`llm-anygate-cli.exe`, `llm-anygate.exe`)
- `markitdown` (`markitdown.exe`)
- `mitmproxy` (`mitmdump.exe`, `mitmproxy.exe`, `mitmweb.exe`)
- `pgcli` (`pgcli.exe`)
- `podman-compose` (`podman-compose.exe`)
- `pysshpass` (`pysshpass.exe`)
- `serena-agent` (`index-project.exe`, `serena-mcp-server.exe`, `serena.exe`)
- `shot-scraper` (`shot-scraper.exe`)
- `specify-cli` (`specify.exe`)
- `vscode-offline` (`vscode-offline.exe`)

Notes:

- Use `pixi` for this repo’s Python environment and dependencies.
- Use `uv tool ...` for running global CLI tools; avoid using `uv` to manage this repo’s Python dependencies.


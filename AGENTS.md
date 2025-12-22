# Repository Guidelines

This repository is an **English-first** tutorial project focused on NPU (Neural Processing Unit) design and implementation. Chinese materials are welcome when helpful; those files should usually include `-cn` or `.cn` in the filename (e.g., `setup-cn.md`, `README.cn.md`).

## Project Structure & Module Organization

- `src/npu_design_tutorial/`: main Python package (core code lives here).
- `tests/`: pytest suite (`test_*.py`).
- `context/`: working context for docs, summaries, plans, and instructions (project-specific knowledge base).
- `paper-source/`: paper artifacts, typically `paper-source/<paper-slug>/tex/...` and `paper-source/<paper-slug>/<paper-slug>.pdf`.
- `dockers/`: optional container/dev tooling.

## Build, Test, and Development Commands

Use `pixi` for Python execution and dependency management; **do not use system** `python`/`pip`/`venv`.

- Enter env: `pixi shell`
- Run a command: `pixi run <cmd>` (example: `pixi run python -V`)
- Editable install: `pixi run python -m pip install -e .`
- Tests: `pixi run pytest -q`
- Build: `pixi run python -m pip install build && pixi run python -m build`

For one-off scripts and generated outputs, use `tmp/<subdir>/`.

## Coding Style & Naming Conventions

- Python 3.11+, UTF-8, 4-space indentation.
- Naming: `snake_case` (modules/functions/vars), `PascalCase` (classes), `UPPER_SNAKE_CASE` (constants).
- Keep changes minimal and consistent with nearby code; use `black`/`ruff` if already configured in the project.

## Testing Guidelines

- Framework: `pytest`.
- Conventions: `tests/test_*.py`, functions `test_*`.
- Prefer small unit tests for new logic; add fixtures only when they simplify reuse.

## Commit & Pull Request Guidelines

- Use Conventional Commits as seen in history: `docs: ...`, `docs(dockers): ...`, `chore(dockers): ...`, `feat: ...`.
- Keep the subject line short; use the body for rationale and verification steps; include `BREAKING CHANGE:` when applicable.
- PRs should explain what/why, link related issues, and include tests or doc updates when behavior changes.

## Agent-Specific Instructions

- Prefer root-cause fixes, avoid unrelated refactors, and don’t introduce new dependencies unless explicitly required.

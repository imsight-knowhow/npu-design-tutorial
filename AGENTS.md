# Repository Guidelines

本仓库为中文优先（Chinese‑first）的教学项目，聚焦 NPU 设计与实现。以下指南面向贡献者与自动化代理（agents）。

## Project Structure & Module Organization
- `src/npu_design_tutorial/`：主 Python 包（当前为骨架）。
- `tests/`：测试请放置于此（使用 `pytest`，文件命名 `test_*.py`）。
- `docs/`（可选）：中文文档；英文译文建议置于 `docs/en/`。
- 根目录脚本：`setup-envs.sh`（代理/开发环境与代理配置）。

## Build, Test, and Development Commands
- 进入开发环境：`pixi shell`（推荐；需已安装 pixi）。
- 安装为可编辑包：`pip install -e .`。
- 运行测试：`pytest -q`（首次请安装：`pip install pytest`）。
- 构建分发包：`python -m pip install build && python -m build`（使用 hatchling 作为构建后端）。

## Coding Style & Naming Conventions
- 语言与版本：Python 3.11+；缩进 4 空格，UTF‑8。
- 命名：模块/函数/变量用 `snake_case`；类用 `PascalCase`；常量用 `UPPER_SNAKE_CASE`。
- 格式与静态检查（建议）：`black`、`ruff`；导入分组 `standardlib | third‑party | local`。
- 文档与注释：中文为主；必要术语可保留英文并附中文解释。

## Testing Guidelines
- 框架：`pytest`；测试文件 `tests/test_*.py`；用例函数 `test_*`。
- 关注可测性：偏向纯函数与小单元；必要时使用 fixtures。
- 覆盖率目标（建议）：核心模块 ≥80%。

## Commit & Pull Request Guidelines
- 提交信息：中文优先，遵循 Conventional Commits（如 `feat: 添加卷积算子 IR`）。
- 示例：`fix(parser): 处理空张量形状的越界问题`。
- PR 要求：
  - 说明动机与变更摘要，关联 Issue（如有）。
  - 包含测试或更新文档；附运行与验证步骤。
  - 变更聚焦、原子化；避免无关格式化改动。

## Agent‑Specific Notes
- 遵循本文件；最小化变更范围；优先修根因并更新文档/测试。
- 触及多个文件时，保持与现有风格一致；不要引入无必要依赖。

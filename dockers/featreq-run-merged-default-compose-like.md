# Feature Request: 让 run-merged.sh 的默认行为与 `docker compose up stage-2` 一致

## 背景与问题
- 目前 `run-merged.sh` 默认前台启动（非 `-d`）可正常阻塞；但若以 `-d` 后台运行，未分配 TTY/STDIN 时，容器会因入口脚本启动交互式 shell 而“秒退”。
- `docker compose up stage-2`（由生成的 compose 文件决定）默认设置了 `tty: true` 与 `stdin_open: true`，即使 `-d` 后台，也能保持容器持续运行。
- 用户预期：
  1) `run-merged.sh` 的默认行为应与 `docker compose up stage-2` 一致（compose-like）：前台阻塞、分配 TTY/STDIN；后台 `-d` 也保留 TTY/STDIN，从而不会“秒退”。
  2) 只有在明确通过 CLI 参数时，才“直接进入 Shell（等价 docker run -it）”。

## 目标与建议的行为
- 默认（无额外参数）：compose-like 行为
  - 前台：阻塞，分配 `-it`，等价于 compose 的 `tty: true` + `stdin_open: true`。
  - 后台（`-d`）：仍旧分配 `-it`（或分别加 `-t` 与 `-i`），保证容器不因交互式 shell 退出而终止。
- 显式 Shell 模式：通过 CLI 参数进入“直接进入交互式 shell”语义（覆盖/强化 `-it`），与 `docker run -it` 等价，且可选传递自定义命令。

## 拟议的 CLI 设计
- 维持现有参数：`-d/--detach`、`-n/--name`、`-p/--publish`、`-v/--volume`、`--gpus` 等。
- 新增/调整：
  - `--shell`：强制进入交互式 shell（默认 `/bin/bash -l`），忽略默认 CMD；可与 `--` 后的自定义命令共存，优先级：显式命令 > `--shell` 默认 shell。
  - `--no-tty`：显式关闭 TTY/STDIN（非常用；默认开启）。
  - 若需要更细粒度：
    - `--stdin-open/--no-stdin-open`（对应 compose 的 `stdin_open`）
    - `--tty/--no-tty`（对应 compose 的 `tty`）

## 环境变量与默认值（`merged.env`）
- 保持并扩展现有键：
  - `RUN_DETACH='0'`、`RUN_TTY='1'`、`RUN_DEVICE_TYPE='gpu'`、`RUN_PORTS=...` 等。
  - 新增建议：
    - `RUN_STDIN_OPEN='1'`：是否在 docker run 时附加 `-i`（默认 1，匹配 compose）。
    - `RUN_KEEP_TTY_IN_DETACH='1'`：在 `-d` 模式也保留 `-t/-i`（默认 1）。
- 兼容性：保留当前 `RUN_EXTRA_ARGS`，但不再要求通过该变量手动加 `-it` 才能让后台保持运行。

## 实现建议（run-merged.sh）
1) 解析新增环境变量并规范化：
   ```bash
   # 新增
   STDIN_OPEN=$(normalize_bool "${RUN_STDIN_OPEN:-1}")
   KEEP_TTY_IN_DETACH=$(normalize_bool "${RUN_KEEP_TTY_IN_DETACH:-1}")
   ```
2) 构建 docker 命令时的 TTY/STDIN 逻辑：
   ```bash
   cmd=( docker run )
   if [[ "$DETACH" == "1" ]]; then
     cmd+=( -d )
     # 与 compose 对齐：后台也保持 -t/-i（可通过 --no-tty 或 KEEP_TTY_IN_DETACH=0 关闭）
     if [[ "$KEEP_TTY_IN_DETACH" == "1" ]]; then
       [[ "$TTY" == "1" ]] && cmd+=( -t )
       [[ "$STDIN_OPEN" == "1" ]] && cmd+=( -i )
     fi
   else
     # 前台：默认 -it（compose-like）
     [[ "$TTY" == "1" ]] && cmd+=( -t )
     [[ "$STDIN_OPEN" == "1" ]] && cmd+=( -i )
   fi
   ```
3) 新增 `--shell` 参数：
   ```bash
   SHELL_MODE=0
   # 解析参数时：
   --shell) SHELL_MODE=1; shift ;;
   # 组装 CMD：
   if [[ $SHELL_MODE -eq 1 && ${#POSITIONAL[@]} -eq 0 ]]; then
     POSITIONAL=( "/bin/bash" "-l" )
   fi
   ```
4) 增加 `--no-tty`、`--tty`、`--stdin-open/--no-stdin-open` 的处理（可选）。
5) 更新 `usage()`：明确“默认 compose-like；使用 `--shell` 进入交互式 shell”。

## 兼容性与迁移
- 兼容旧参数与 `merged.env`，默认行为更贴近 compose，降低“秒退”踩坑概率。
- 对依赖“非交互后台”的场景：
  - 可通过 `--no-tty` 或设置 `RUN_TTY='0' RUN_STDIN_OPEN='0'` 来禁用 `-t/-i`。

## 测试要点
- 前台默认（无参）：阻塞、输出入口日志、可 Ctrl+C 结束。
- 后台 `-d`：容器保持运行（不“秒退”）；`docker logs -f` 可看到同样日志。
- `--shell`：无命令时进入 `/bin/bash -l`；显式 `-- <CMD>` 时以 CMD 为准。
- `--no-tty`：验证后台不分配 `-t/-i`，容器是否仍能按需存活（取决于 entrypoint/CMD）。

## 文档更新
- 更新 `dockers/README.md`：
  - 明确默认是 compose-like（前台阻塞；后台也保持 TTY/STDIN）。
  - 新增 `--shell` 用法示例，减少新手对 `RUN_EXTRA_ARGS='-it'` 的依赖。

## 开发步骤（建议顺序）
- [ ] 在 `run-merged.sh` 实现上述参数与默认逻辑。
- [ ] 在 `merged.env` 增加 `RUN_STDIN_OPEN` 与 `RUN_KEEP_TTY_IN_DETACH`，设置默认值为 1。
- [ ] 更新 usage 文案与示例。
- [ ] 本地验证：前台、后台、`--shell`、`--no-tty`、GPU/非 GPU。
- [ ] 更新 `dockers/README.md` 并提交。

---
通过上述调整，`run-merged.sh` 的体验将与 `docker compose up stage-2` 更一致，并为“显式进入 shell”提供清晰直观的 CLI 参数，既兼容老用户，又显著降低新手的使用门槛。

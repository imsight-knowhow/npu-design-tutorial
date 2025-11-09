# PeiDocker 容器项目使用指南（dockers/main）

本目录用于管理本项目的开发容器。PeiDocker CLI（`pei-docker-cli`）可以把一个简单的 YAML 配置转换为可复现的 Docker 开发环境，支持两阶段构建：
- stage‑1：系统与基础工具层（APT 源、SSH、常用工具等）
- stage‑2：应用与开发者层（用户、工作目录、挂载卷、常用开发工具等）

本文一步一步带你从零开始安装工具、创建工程、生成配置、构建镜像，并启动和进入容器。

---

## 0. 环境准备（新手向）

1) 安装 Docker 与 Compose v2（必需）
- 验证安装：
  ```bash
  docker --version
  docker compose version
  ```

2) 准备 GPU（可选）
- 若需要在容器中使用 GPU：
  - 安装 NVIDIA 显卡驱动与 `nvidia-container-toolkit`
  - 验证宿主可用：
    ```bash
    nvidia-smi
    docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi
    ```

3) 安装 uv（用来安装/运行 pei-docker）
- 一条命令安装（Linux/macOS）：
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- 确保 `~/.local/bin` 在 PATH 中（必要时重开终端），验证：
  ```bash
  uv --version
  ```
- 不了解 uv？可临时运行：
  ```bash
  uvx --from pei-docker pei-docker-cli --help
  ```

4) 安装 pei-docker（两种方式，任选其一）
- 方式 A：安装为 uv tool（推荐，方便直接调用）
  ```bash
  uv tool install pei-docker
  pei-docker-cli --help
  ```
- 方式 B：临时执行（不安装到 PATH）
  ```bash
  uvx --from pei-docker pei-docker-cli --help
  ```

---

## 1. 创建容器工程骨架（cn-dev 模板）

在仓库根目录执行：
```bash
pei-docker-cli create -p dockers/main --quick cn-dev
```
含义：
- `-p dockers/main`：目标工程目录
- `--quick cn-dev`：选择“大陆开发环境”模板（APT 镜像源友好，默认启用 SSH）

然后用仓库内提供的配置覆盖默认配置：
```bash
cp dockers/user_config.main.yml dockers/main/user_config.yml
```

执行完成后，`dockers/main` 会包含：
- `user_config.yml`：你的工程主配置（可编辑）
- `compose-template.yml`：Compose 模板
- `stage-1.Dockerfile`、`stage-2.Dockerfile`：两阶段 Dockerfile 模板
- `installation/`：分阶段的安装脚本与钩子
- `examples/`：示例配置
- `reference_config.yml`：参考完整配置

---

## 2. 生成 Compose 与合并构建文件

在仓库根目录执行：
```bash
pei-docker-cli configure -p dockers/main --with-merged
```
作用：
- 处理 `user_config.yml`（支持 `${VAR:-default}` 的环境变量替换）
- 生成用于直接部署的 `dockers/main/docker-compose.yml`
- 生成“合并构建”产物：
  - `dockers/main/merged.Dockerfile`（把两阶段合并为一个 Dockerfile）
  - `dockers/main/merged.env`（构建与运行参数）
  - `dockers/main/build-merged.sh`、`dockers/main/run-merged.sh`（便捷脚本）

提示：修改 `user_config.yml` 后，建议再次执行 `configure` 重新生成上述文件。

---

## 3. 构建镜像（两种方式）

方式 A（推荐开发）：合并构建，一步得到 stage‑2 镜像
```bash
cd dockers/main
./build-merged.sh
# 传递原生命令给 docker build：
./build-merged.sh -- --no-cache --progress=plain
# 指定输出镜像名：
./build-merged.sh -o myrepo/npu-dev:stage-2
```
构建完成后，镜像通常为 `npu-dev:stage-2`（见 `merged.env`）。

方式 B：Docker Compose 分阶段构建
```bash
docker compose -f dockers/main/docker-compose.yml build stage-1 stage-2
```

验证镜像是否存在：
```bash
docker images | grep npu-dev
```

---

## 4. 启动容器（run-merged.sh 或 Docker Compose）

方式 A：使用 `run-merged.sh`（读取 `merged.env`，启动单个 stage‑2 容器）
```bash
cd dockers/main
# 前台交互运行（默认）：
./run-merged.sh
# 后台运行（detached）：
./run-merged.sh -d
```
注意：Entrypoint 默认启动交互式 `bash`。若后台运行后容器立即退出，请：
- 永久方式：编辑 `dockers/main/merged.env`，设置 `RUN_EXTRA_ARGS='-it'`
- 临时方式：`RUN_EXTRA_ARGS='-it' ./run-merged.sh -d`

方式 B：使用 Docker Compose（只运行 stage‑2）
```bash
# 后台启动并保持运行
docker compose -f dockers/main/docker-compose.yml up -d stage-2
# 跟随日志（可观察 SSH 服务是否启动）
docker compose -f dockers/main/docker-compose.yml logs -f stage-2
```

默认端口与卷（可在 `user_config.yml`/`docker-compose.yml` 调整）：
- 端口：SSH `40012:22`
- 卷：`app:/hard/volume/app`，`data:/hard/volume/data`，`/workspace:/hard/volume/workspace`，`home_me:/home/me`
- GPU：模板已声明 NVIDIA 设备（需 `nvidia-container-toolkit` 与合适驱动）

---

## 5. 通过 SSH 登录并检查

登录（首次会提示接收指纹，输入 `yes`）：
```bash
ssh -p 40012 me@127.0.0.1
# 默认密码：123456（强烈建议改为密钥登录）
```
进到容器后，验证：
```bash
whoami      # 期望输出：me
id          # 期望包含 sudo 等组
getent passwd me
```

---

## 6. 常用运维命令

停止/删除：
```bash
docker stop pei-stage-2                                      # 停止 run-merged.sh 启动的容器
docker compose -f dockers/main/docker-compose.yml stop stage-2
docker compose -f dockers/main/docker-compose.yml down        # 移除网络/容器
```
进入容器（无需 SSH）：
```bash
docker exec -it <container> /bin/bash
```
清理镜像/容器（PeiDocker 提供）：
```bash
pei-docker-cli remove -p dockers/main -y
```

---

## 7. 故障排查（FAQ）

- 容器“秒退”或后台启动后立刻退出
  - 原因：Entrypoint 启动的是交互式 shell；非交互后台会很快结束。
  - 解决：设置 `RUN_EXTRA_ARGS='-it'` 后再后台运行，或用 Compose 的 `-d` 方式。

- 端口冲突（`40012` 已被占用）
  - 修改 `dockers/main/user_config.yml` 中 SSH `host_port` 或直接编辑生成的 `docker-compose.yml`。

- GPU 不可用
  - 确认宿主已安装 `nvidia-container-toolkit` 且 `docker run --gpus all ... nvidia-smi` 可用。
  - 若无 GPU，可在 `merged.env` 设置 `RUN_DEVICE_TYPE='cpu'`，或在 `user_config.yml` 里将设备改为 CPU。

- 代理/网络慢
  - `merged.env` 中 `PEI_HTTP_PROXY_*`/`PEI_HTTPS_PROXY_*` 与 `ENABLE_GLOBAL_PROXY` 控制构建/运行时代理。
  - `user_config.yml` 中 `repo_source` 可设置 APT 镜像（如 `tuna`/`aliyun`）。

- 文件换行符/权限
  - Windows/CRLF 文件会在构建时自动转换为 LF；若脚本权限问题，可检查 `installation/` 下脚本是否有执行权限。

---

## 8. 快速上手清单（复制即用）

```bash
# 0) 安装 uv（Linux/macOS）
curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version

# 1) 安装 pei-docker
uv tool install pei-docker
pei-docker-cli --help

# 2) 生成工程并覆盖配置
pei-docker-cli create -p dockers/main --quick cn-dev
cp dockers/user_config.main.yml dockers/main/user_config.yml

# 3) 生成配置与合并构建文件
pei-docker-cli configure -p dockers/main --with-merged

# 4A) 合并构建并运行（推荐）
cd dockers/main
./build-merged.sh
RUN_EXTRA_ARGS='-it' ./run-merged.sh -d

# SSH 验证
ssh -p 40012 me@127.0.0.1   # 默认密码 123456

# 4B) 或使用 Compose
docker compose -f dockers/main/docker-compose.yml build stage-1 stage-2
docker compose -f dockers/main/docker-compose.yml up -d stage-2
```

---

若需定制更多服务、卷或环境变量，直接编辑 `dockers/main/user_config.yml`，然后再次执行 `configure` 生成新配置。

# 项目设计与实施建议（Project Design）

## HEADER
- Purpose: 将《project-idea》转化为可执行的实施路线与技术选型
- Status: Draft
- Date: 2025-11-06
- Dependencies: Python 3.11+, TVM(Relay/BYOC), ONNX, NumPy
- Target: 开发者与 AI 助手（中文优先）

## 概述
面向教学与实践的 NPU 软件仿真项目：
- 以 TVM 生态为核心实现“ONNX → Relay → 自定义 IR/ISA → 模拟器”流程。
- 支持 Transformer 与 CNN 常用子图；必选算子含 MatMul、Conv2D、Add、ReLU、Softmax；可选包含 GELU、FlashAttention、Transpose 等。
- 数据类型：INT8、FP16（必选），FP32（可选，用于校准/对比）。
- 产出：可执行 .onnx 模型的软件仿真器、基于 TVM 的编译工具链、性能分析工具与示例。

## 技术栈与工具
- Python 为主：TVM(前端导入/Relay Pass/BYOC 外挂 Codegen)、NumPy/Numba(仿真内核)、onnx(模型解析)、typer+rich(CLI)、pydantic/yaml(配置)、pytest/hypothesis(测试)。
- 可选加速：
  - C++(pybind11) 或 Rust(PyO3/maturin) 实现热点内核/内存模型；以 FFI 方式暴露给 Python。
  - 仅在性能瓶颈明确后逐步引入，保持教学可读性。

## 推荐目录结构（参考社区最佳实践与 TVM BYOC 指南）
```
context/                        # 项目协作知识库（已建立）
docs/
  zh/                          # 中文文档
  en/                          # 英文译文（可选）
examples/
  models/                      # 示例 ONNX 模型
  notebooks/                   # 教学 Notebook
scripts/                       # 开发/发布脚本
benchmarks/                    # 基准与性能分析脚本
tests/
  unit/
  integration/
  e2e/
src/npu_design_tutorial/
  ir/                          # 自定义 IR（图/指令级）
  isa/                         # 指令集与编码
  compiler/
    frontends/                 # Relay/ONNX 导入与兼容层
    passes/                    # 图优化/融合/量化
    codegen/                   # BYOC/自定义 Codegen → IR/ISA
  runtime/
    simulator/                 # 指令解释器/调度器/内存模型
    kernels/                   # 算子内核（numpy/numba/可选C++/Rust）
  tools/                       # CLI 与可视化/分析
  utils/                       # 公共组件（日志、配置、校验）
```

说明：
- 采用 `src/` 布局、`tests/` 并列；划分 `ir/isa/compiler/runtime` 以贴合“编译器-执行器”分层。
- BYOC 路径：在 `compiler/codegen/` 内实现 Relay 分区后的外部 Codegen 与 Runtime 适配（可先做最小可行子集）。

## 分阶段实施（Step-by-step）
1) 基础设施与骨架
   - 搭建环境（pixi/pyproject），完善格式化(lint/black/ruff) 与测试(pytest)。
   - 初始化 `src/…` 目录与 CLI（`npu`：compile/run/profile）。

2) IR/ISA 最小子集
   - 定义张量形状/数据类型、内存布局与指令格式；选取核心算子：MatMul、Add、ReLU、Softmax、Conv2D（逐步）。
   - 约定量化与精度策略：INT8/PTQ 或 FakeQuant；FP16 作为主精度，FP32 用于校准对比。

3) 软件仿真运行时（runtime/simulator）
   - 实现指令解释器、调度与内存模型（线性内存 + 简单 DMA 抽象）。
   - 以 NumPy/Numba 完成内核原型并建立单元测试与数值对齐基线。

4) 编译前端与图处理
   - 导入 ONNX → TVM Relay；标注可 offload 的子图（Compiler Tag）。
   - Pass：常量折叠、算子融合、layout/precision 处理与校验。

5) Codegen 与可执行产物
   - Relay 子图 → 自定义 IR → ISA 序列；产出可供模拟器执行的程序与元数据。
   - 打通最小 e2e：若干小模型（MLP/CNN 小网络）可跑通且数值对齐（容差设定）。

6) 扩展与优化
   - Transformer 支持（逐步引入：Attention → FlashAttention 可选）。
   - 量化/分块/向量化等优化；必要时以 C++/Rust 重写热点内核。

7) 基准与验证
   - `benchmarks/`：微基准 + 模型级评测；指标含吞吐/延迟/带宽利用率。
   - 端到端对齐：与 PyTorch/ONNXRuntime/TVM Host 作数值与性能对比。

## 里程碑（示例）
- M1：IR/ISA 草案 + 核心算子原型 + 单测通过
- M2：ONNX→Relay→IR 打通最小子图 + e2e 演示
- M3：CLI 工具链 + 示例模型 + 文档
- M4：Transformer/CNN 完整子集 + 性能基准
- M5：可选 C++/Rust 热点优化 + 教学材料完善

## 参考
- TVM: How to Bring Your Own Codegen — https://tvm.apache.org/2020/07/15/how-to-bring-your-own-codegen-to-tvm
- （讨论）BYOC JSON 运行时与分区实践 — https://discuss.tvm.apache.org/t/byoc-json-codegen-for-byoc/9808
- Python 项目结构与最佳实践（src 布局）— https://dagster.io/blog/python-project-best-practices

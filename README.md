# workAI · 超声非线性导波研究工作台

研究对象是铝板中超声导波的材料非线性：基频激励下二次谐波（2f）的产生与沿程累积。仓库包含五部分：
- **数值求解器**：三维谱元、三维四面体、二维平面应变。
- **物理 AI**：冻结图网络及其继续训练、v2 输入层、粗网格算子。
- **COMSOL 参考解流程**。
- **可离线运行的软件**。
- **网页工作台**。

所有结论都附有证据文件；没有通过的验收如实保留。整合前的完整说明见 [项目历史](docs/history.md)。

## 现状一览

| 部分 | 状态 | 关键结果 | 详细 |
|---|---|---|---|
| 离线软件 0.3.0 | 研究预发布版，Windows / Linux 便携包；界面已重建为仿真工作台（模型树、三维模型视图、属性面板、激励信号预览、2f 分辨率与验收范围检查），可在界面中编辑多源、任意信号、对称面和吸收层；材料可用自定义应变能密度 W(E) 定义（各向异性，如 CFRP），见 [材料](docs/materials.md) | 默认权重的 5 ms 长记录：20 组 0 组达到 5%，全场相对 L2 24.83%–141.97%，能量收支全部通过；原 0.2.0 模型 14 μs 误差 0.438% | [5 ms 验证](research/long-validation/README.md)、[离线说明](offline/README-OFFLINE.txt) |
| 冻结图网络继续训练 v6–v10 | 均未采纳为发布模型 | 最好的 v10（六阶、h 1.6 mm）5 ms 每组最差接收点差异 15.01%–105.71%，0/20 | [第 10 轮](research/comsol-validation/CONTINUATION-V10.md)、[路线总表](research/comsol-validation/README.md) |
| 模态降阶 v11–v12 | 研究候选，未采纳 | v12 原生 COMSOL 20 μs 核对 1.96% / 1.62%；5 ms 线性对照波形向量 19/20，全部分量与频带 11/20，最差 35%–40%；尚无收敛的 5 ms 非线性参考 | [第 11 轮](research/comsol-validation/CONTINUATION-V11.md)、[第 12 轮](research/comsol-validation/CONTINUATION-V12.md) |
| 数值求解器 | 可用，含收敛复核 | 谱元线性解与 COMSOL 三个 P1 线性结果相差 0.01%–0.3% | [激励 v2](docs/solver-excitation-v2.md)、[扫描与收敛复核](docs/sweep-and-datasets.md) |
| 图网络 v2 输入层 | 可用，离线软件可导入 v2 场景 | 任意信号、多源、对称面、吸收层；与冻结网络逐位一致，与谱元求解相差 4e-14 | [输入层](docs/graph-network-input-design.md) |
| 粗网格算子（方向 B） | 研究模型，逐例核验后使用；离线软件的快速引擎，可选两种模型 | 快速模型（4 阶 h 3 mm）比冻结网络细网格推理快约 16 倍，单源激励 4 个留出工况全部门槛通过，双贴片 1/8。精确模型（4 阶 h 2 mm 重训）12 个单源和双贴片留出工况全部通过，计算量与冻结网络 h 2 mm 相同（后者 10/12）。梳状激励两者都是 0/4，因为齿宽在粗网格上表示不准 | [粗网格算子](docs/coarse-graph-operator.md) |
| COMSOL P1 参考 | 24 次中完成 9 次：线性与 Murnaghan 基准齐全，StVK 和收敛变体待算 | 谱元标签与 COMSOL 的非线性增量 2f 一致到 0.06%–2.7%（三个工况），完整波形一致到 0.035% 以内 | [数据要求与进度](docs/comsol/comsol-data-request.md)、[P1 目录](research/comsol-p1-20261024/README.md) |
| 早期代理模型 | 已停止 | 坐标神经场、谱算子、物理约束场网络、模态能量模型都没有通过全部 5% 门槛 | [项目历史](docs/history.md) |

## 跨路线的结论：精度受离散分辨率限制

- **冻结图网络本质上是谱元格式。** 它学到的求导滤波器、能量系数和状态更新，都已收敛到精确的 GLL 谱元算子，所以它的精度和计算量就是谱元离散的精度和计算量（[核对](docs/graph-network-input-design.md)）。
- **main 路线的数据显示，训练收益远小于离散改进。**
  - v9：同一网格上训练使每组最大差异下降 0.04 到 0.22 个百分点；单独把网格从 2.5 mm 加密到 1.6 mm，下降 3.8 到 22.7 个百分点。
  - v10：训练下降 0.11 到 0.63 个百分点；从 5 阶升到 6 阶，下降 8.3 到 45.1 个百分点。
- **二次谐波需要足够的节点。** 方向 B 的实验中，2f 处 A0 模态每个波长约 3 个节点时，学习无法把 2f 误差压到 5%；约 4.5 个节点时，学习后原始 2f 全部达标。
- **参考解必须先收敛。** 5 ms 的 COMSOL 线性参考本身没有通过模态截断和网格检查，所以 5 ms 的 5% 门槛目前无法认证；30 μs 级的短记录有收敛的参考（谱元标签与 COMSOL 一致到万分之一），可以认证。

据此，下一步的优先顺序是：
1. 先补齐收敛的参考，包括 P1 的非线性运行和更细的 5 ms 参考。
2. 按“2f 波长内的节点数”选网格和阶次。
3. 把训练用在能带来收益的地方，即适度粗化的粗网格算子，以及不受载波波长限制的模态包络表示。

## 模块地图

| 目录 | 内容 |
|---|---|
| `physics/` | 场景 v2：信号、空间分布、多源、对称面与吸收层的统一定义；材料（Murnaghan 或自定义应变能 W(E)）；接收点与运行输出格式 |
| `spectral3d/`、`solid/`、`spectral2d/` | 三维 GLL 谱元（显式与隐式 Newmark）、三维四面体有限元、二维平面应变谱元 |
| `dispersion/`、`waveguide/` | Rayleigh–Lamb 频散、模态匹配、谐波分析；有限宽截面 SAFE 核对 |
| `sweep/` | 参数扫描、训练数据集、自动网格收敛复核，支持全部求解器与图网络 |
| `graphwave/` | 冻结图网络（离线软件的推理核心，运行时文件按哈希锁定）及 v6–v12 继续训练、模态降阶 |
| `graphnet/` | 冻结网络的 v2 输入层：多源载荷、按分量约束、吸收层，短记录与分块长记录 |
| `coarsegraph/` | 粗网格算子学习、训练好的模型、COMSOL 数据检查导入与比较 |
| `offline/` | 离线便携软件（本机浏览器界面）；`tools/build-offline.py` 打包 |
| `server/`、`public/` | 网页工作台，连接远程 COMSOL 服务 |
| `tools/` | COMSOL 作业提交与收集、验证矩阵、证据打包、离线构建、`run-unit-tests.py` |
| `research/` | 训练与验证证据：COMSOL 对照、5 ms 验证、P1 数据、各轮报告 |
| `training/`、`modal3d/`、`operator3d/`、`spectralop/` | 训练产物与早期代理模型（研究记录，未用于推理） |
| `docs/` | 方法与结果文档，见下方索引 |
| `environment/` | 2026-10-23 的环境与源码备份说明 |

## 文档索引

- **总览：** [项目历史](docs/history.md)、[网页工作台操作](docs/workbench.md)、[原研究模型恢复](docs/recovered-research.md)
- **物理与材料：** [自定义应变能材料（CFRP 等）](docs/materials.md)、[铝材料参数与三阶本构](docs/aluminum-material.md)、[非线性导波核对依据](docs/nonlinear-guided-waves.md)、[COMSOL 非线性实现](docs/nonlinear-comsol.md)
- **数值求解：** [三维超弹性](docs/solid-hyperelastic.md)、[五组对照](docs/solid-harmonics.md)、[直达波与沿程比较](docs/solid-propagation.md)、[高阶参考解](docs/spectral-reference.md)、[激励 v2](docs/solver-excitation-v2.md)、[平面应变与频散](docs/plane-strain-dispersion.md)、[有限宽截面核对](docs/rectangular-waveguide.md)
- **扫描与数据：** [参数扫描、数据集与收敛复核](docs/sweep-and-datasets.md)，示例规格在 `docs/examples/`
- **软件核验：** [与文献和解析解对照（材料、波速、频散、二次谐波、吸收层）](research/literature-verification/README.md)、[1 ms 长记录精度](research/long-record-accuracy/README.md)、[缺口与通孔散射](research/scattering/README.md)
- **物理 AI：** [图网络输入层与 v2 设计](docs/graph-network-input-design.md)、[粗网格算子](docs/coarse-graph-operator.md)、[COMSOL 对照与继续训练总表](research/comsol-validation/README.md)、[5 ms 验证](research/long-validation/README.md)、[继续训练八例复核](research/continued-training/README.md)
- **早期代理模型：** [物理约束训练](docs/physics-training.md)、[三维跨工况训练](docs/operator-training-3d.md)、[传播表示改进](docs/operator-training-improvement.md)、[谱算子](docs/spectral-operator.md)、[模态能量耦合](docs/modal-energy-training.md)
- **COMSOL 参考数据：** [仿真要求、数据格式与进度](docs/comsol/comsol-data-request.md)、[P1 数据任务](research/comsol-p1-20261024/README.md)

## 快速开始

**离线软件。** 从 [v0.3.0 发布页](https://github.com/lx016922-dot/workAI/releases/tag/v0.3.0) 下载便携包；Windows 解压后运行 `Start-WorkAI.bat`，Linux 运行 `sh start-workai.sh`。从源码构建：

```sh
python tools/download-offline.py --cache build-cache --platform linux
python tools/build-offline.py --cache build-cache --output portable --platform linux --example --archive
```

**网页工作台。** 需要 Node.js 22 或以上，`npm start`；部署、环境变量和远程 COMSOL 连接见 [workbench.md](docs/workbench.md)。

**数值求解与扫描。** 三维依赖由 `bash solid/prepare-solid.sh` 安装（gmsh、numba）。

```sh
python -m spectral3d.run --scene scene.json --output out --degree 3
python -m spectral2d.run --scene docs/examples/plate-2d-s0-line.json --output out2d
python -m sweep run --spec docs/examples/sweep-3d-convergence.json --output runs/3d
python -m sweep converge --campaign runs/3d
```

**图网络与粗网格算子。**

```sh
python -m graphnet.run --scene scene.json --output out-net --degree 3
python -m coarsegraph.run --model coarsegraph/models/p4-h3-harmonic/model.json --scene scene.json --output out-coarse
```

**COMSOL 数据。** 按 [数据要求](docs/comsol/comsol-data-request.md) 导出后：

```sh
python -m coarsegraph.comsol check <数据目录> [--partial]
python -m coarsegraph.comsol compare-linear <数据目录> --labels <谱元标签目录…>
```

## 测试

```sh
python tools/run-unit-tests.py --group core       # 场景、频散、COMSOL 数据与采集、模态工具
python tools/run-unit-tests.py --group solvers    # 求解器、截面核对、扫描与收敛复核（需要 gmsh）
python tools/run-unit-tests.py --group networks   # v2 输入层、粗网格算子、离线软件
python tools/run-unit-tests.py --group torch      # 训练相关测试（需要 PyTorch）
npm test                                          # 网页工作台与离线界面
```

测试入口会检查是否有测试模块没有归入任何一组。`.github/workflows/tests.yml` 在推送到 main 或 `claude/` 分支、以及提交 PR 时运行全部测试。其余工作流中，离线构建在离线软件或冻结网络的文件改动时自动运行；验证矩阵、证据审计和发布按需手动触发。

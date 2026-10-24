# 参数扫描与训练数据批处理（`sweep`）

`sweep` 包把一个基准场景展开为一组不可变用例，在隔离进程中并行、可恢复地运行，再汇总非线性导波指标、自动做网格收敛复核，并打包为训练数据集。它只调度仓库内的数值求解器（二维平面应变谱元、三维高阶谱元、三维四面体），不训练模型，也不把任何结果路由给代理。

| `engine` | 基准场景 | 求解器入口 | 阶次 | 说明 |
|---|---|---|---|---|
| `2d_plane_strain` | v2，`dimension: 2d_plane_strain` | `python -m spectral2d.run` | 2–6，默认 4 | 未写 `engine` 时按场景自动选择 |
| `3d_spectral` | v1 或 v2 三维板、缺口、管、通孔 | `python -m spectral3d.run` | 2–6，默认 3 | 通孔走隐式 Newmark，`substeps` 为每个输出采样的内部步数（默认 4） |
| `3d_tetra` | v1 或 v2 三维参数化结构（不含 CAD） | `python -m solid.run` | 无 | 一阶四面体，h 即单元尺寸 |

## 1. 扫描规格 `workai.sweep.v1`

```json
{
  "schema": "workai.sweep.v1",
  "name": "2 mm 铝板 · S0 载频扫描 · A/2A 幅值标度",
  "base_scene": "plate-2d-s0-line.json",
  "degree": 4,
  "grid": { "source.frequency_mhz": [0.2, 0.3, 0.4, 0.5], "source.amplitude_mpa": [1.0, 2.0] },
  "cases": [ { "geometry.notch_depth_mm": 0.3 } ],
  "sampling": { "method": "latin_hypercube", "count": 24, "seed": 20261024, "ranges": { "geometry.thickness_mm": [1.5, 3.0] } },
  "links": { "source.cycles": { "path": "source.frequency_mhz", "scale": 16.0, "offset": 0.0 } },
  "split": { "train": 0.75, "validation": 0.125, "test": 0.125, "group_by": ["geometry.thickness_mm"], "seed": 7 },
  "analysis": { "line": "top", "window": { "mode": "moving" }, "mode_for_window": "S0" },
  "outputs": { "fields": false, "csv": false },
  "budget": { "seconds_per_case": 1800, "workers": 2 }
}
```

- `base_scene`：内联对象或相对规格文件的路径。`engine` 与场景维度必须一致：二维场景只能用 `2d_plane_strain`，三维场景用 `3d_spectral`（默认）或 `3d_tetra`。
- `degree`、`substeps`：谱元阶次与通孔隐式求解的子步数；对四面体无作用。
- `grid`：路径 → 取值列表，全因子展开；`cases`：显式覆盖字典列表，与网格叉乘；`sampling`：拉丁超立方或均匀随机，在 `ranges` 内按 `seed` 可复现地采样。路径用点号，支持列表下标，例如 `receiver_lines.0.from_mm.2`。
- `links`：线性联动，`target = scale × value(path) + offset`，用于让激励中心、接收点和接收线随板厚移动，或让周期数随载频变化以保持脉冲时长。
- 每个用例在展开后按求解器的规则完整校验（分辨率筛选按 h/阶次），校验后的场景与阶次的 SHA-256 决定用例编号，重复用例自动合并。
- `split`：按 `group_by` 的取值分组，同一组落在同一子集，避免同一几何同时出现在训练与测试集；比例按组数四舍五入。
- `analysis`：汇总时的载频（默认取激励载频）、接收线名、时间窗（`moving` 按 x/c_g 移动，群速度默认取该板厚理论模态 `mode_for_window` 的值；`fixed` 用 `start_us`/`end_us`）。三维场景不套用 Lamb 理论值（有限宽板条不是 Lamb 问题），带接收线时必须用 `fixed` 窗或给出 `window.group_velocity_m_s`，否则在规格校验时就报错。
- `convergence`（可选）：网格收敛复核的默认设置，见第 4 节。

## 2. 命令

```sh
python -m sweep plan      --spec docs/examples/sweep-s0-frequency.json
python -m sweep run       --spec docs/examples/sweep-s0-frequency.json --output runs/s0-sweep --workers 2
python -m sweep run       --spec ... --output runs/s0-sweep --resume
python -m sweep summarize --campaign runs/s0-sweep
python -m sweep dataset   --campaign runs/s0-sweep --output runs/s0-sweep/dataset.npz
python -m sweep converge  --campaign runs/s0-sweep --cases '{"constitutive": "murnaghan"}'
```

运行目录：`spec.json`、`manifest.json`（规格哈希、源码哈希、求解器标识与入口模块、每个用例的参数、划分、阶段、结果哈希、节点数、步数、能量收支、耗时或错误）、`cases/<id>/{scene.json,settings.json,run.log,out/}`，`out/` 即对应求解器入口的输出。每个用例在独立进程中运行，`workers` 控制并发数，`seconds_per_case` 为单例超时。`--resume` 只重跑结果缺失或哈希不符的用例；规格改变会被拒绝。失败用例记录错误并继续，命令以退出码 2 结束。

### 三维运行入口的输出

三个求解器入口写出同一套目录结构，`sweep` 的汇总、收敛复核和数据集因此对二维、三维一视同仁：

- `result.json`：场景、诊断、`dimension`（`2d_plane_strain` 或 `3d`）、点接收器的各位移分量（二维 `u_x_m`/`u_z_m`，三维再加 `u_y_m`）与方向投影 `projected_m`、接收线清单、各文件 SHA-256。
- `lines.npz`：每条接收线的坐标、沿程距离、全部位移分量与方向投影；可选 `fields.npz` 与逐线 CSV。

三维接收线可以放在实体内任意位置（例如板中面），越出实体的点在求解前报错。三维网页任务暂不接受接收线，请用命令行或扫描。`spectral3d.run` 对 v1 场景的点接收波形与网页 worker 的求解调用逐位相同。

```sh
python -m spectral3d.run --scene plate.json --output out/spectral --degree 3 [--substeps 4] [--fields] [--csv]
python -m solid.run      --scene plate.json --output out/tetra [--fields] [--csv]
```

四面体网格生成改为单线程：同一场景两次划分的网格逐位相同，用例编号与结果因此可复现，收敛复核里的差异也不会混入网格随机性。

## 3. 汇总（`summary.json` / `summary.csv`）

每个已完成用例：接收点在给定窗内的 f/2f 幅值与相位；接收线逐点 A₁、A₂，A₂ 与 A₂/A₁² 对距离的线性拟合与 R²，相位斜率相速度与包络群速度，并列出该板厚、该载频的理论模态相速度与群速度；能量收支、最大应变、节点数、步数、耗时。参数只在 `source.amplitude_mpa` 上相差一倍的用例自动配成 A/2A 对，输出 A₁≈2 倍、A₂≈4 倍的逐点检查与总判定。CSV 每行一个用例，便于直接绘图。

**线性对照与非线性增量。** 参数只在 `constitutive` 上不同的用例自动配成对照对（`linear` 与 `murnaghan`/`stvk`）：在同一网格与时间步上把非线性解减去线性解，再提取增量的 f/2f。短脉冲自身在 2f 处的线性频谱成分随幅值线性增长（倍率 2），会淹没真正的二次谐波；增量把它完全扣除。A/2A 配对因此给出两种判定：原始 2f（常因上述原因不通过）与非线性增量（应为 4 倍）。要用这一功能，在 `grid` 中加入 `"constitutive": ["linear", "murnaghan"]`。

谐波幅值依赖时间窗，A₂/A₁² 的斜率只在同一窗口设置下用于相对比较，不是材料 β。数据集的 `parameters` 只含数值路径，`categorical_parameters` 以 JSON 字符串保存本构等非数值参数。

## 4. 自动网格收敛复核（`sweep converge`）

```sh
python -m sweep converge --campaign runs/3d [--cases N | id1,id2 | '{"constitutive": "murnaghan"}'] [--levels space,degree,time] [--space-factor 0.7071] [--time-factor 0.5] [--workers 2]
```

对选中的已完成用例，用同一场景独立重算三个细化层级，再与原始结果逐接收点比较：

| 层级 | 做法 | 适用 |
|---|---|---|
| `space` | h × `space_factor`（默认 1/√2） | 所有引擎；显式求解时 CFL 规则会同时缩小 Δt |
| `degree` | 阶次 + 1 | 谱元引擎；四面体与 6 阶时跳过并写明原因 |
| `time` | CFL × `time_factor`（默认 0.5）；通孔隐式求解改为 Newmark 子步加倍 | 所有引擎；CFL 已在 0.05 下限或子步将超过 32 时跳过 |

判定沿用 `spectral3d/compare.py`：投影波形相对 L2 误差（以细化结果为参照，低于 1e-13 m 报告下限时按绝对差判定），以及分析窗内精确频率 f、2f 幅值的相对差。默认容差都是 5%，可在规格的 `convergence.tolerance` 或命令行覆盖。每个层级还核对细化运行确实更细（节点更多或内部步长更小），否则记为错误。所有接收点与接收线点都通过、所有未跳过的层级都通过，该用例才算通过；任一不通过时命令以退出码 3 结束。

**非线性增量的收敛。** 短脉冲自身在 2f 处的线性频谱成分可能比真正的二次谐波大两个数量级，原始 A₂ 收敛并不说明非线性部分已被解析。因此选中一个非线性用例时会自动带上它的线性对照用例（参数只差 `constitutive`），在每个层级上分别求「非线性 − 线性」增量，再比较基准与细化两次的增量 A₂，只用 A₂ 判定。

**用例选择**：缺省为全部已完成用例；`N` 取前 N 个；也可给逗号分隔的编号，或 JSON 参数筛选 `{路径: 取值或取值列表}`。规格里的 `convergence.cases` 是同样的格式，作为缺省值。

**输出**：`cases/<id>/convergence/<层级>/{scene.json,settings.json,run.log,out/}`；`report.json` 记录逐层逐接收点的波形与谐波差异、基准与细化的诊断、汇总判定；非线性用例另有 `increment.json`。`manifest.json` 中该用例记下判定、最差误差与报告哈希；`summary.csv` 增加 `convergence_pass`、`convergence_worst_waveform_l2`、`convergence_worst_a2_relative`、`convergence_increment_pass`、`convergence_increment_worst_a2_relative` 五列。

通过只说明这一个用例在这些细化层级下差异小于容差，不是收敛证明，也不能推广到别的频率、厚度或传播距离。

### 实测：40 mm 铝板，0.3 MHz

示例规格 `docs/examples/sweep-3d-convergence.json`：40 × 2 × 2 mm 铝板，左端面整面加载 0.3 MHz 三周期 Hann 脉冲，记录 16 μs；中面 10–30 mm 五点接收线，35 mm 处一个点接收器。线性与 Murnaghan × 1、2 MPa 共 4 例，对 Murnaghan 2 MPa 用例及其线性对照做三层复核。同一规格改用 `3d_tetra`、h = 0.6 mm（初始筛选允许的最粗档）再跑一次。

| 引擎与网格 | 层级 | 细化后节点数 | 波形最大相对 L2 | 原始 A₂ 最大相对差 | 增量 A₂ 最大相对差 | 判定 |
|---|---|---|---|---|---|---|
| 谱元 2 阶，h = 0.9 mm，4459 节点 | space | 10287 | 0.04% | 0.98% | 0.17% | 通过 |
| | degree | 13600 | 0.07% | 0.73% | 0.20% | 通过 |
| | time | 4459 | 0.03% | 0.11% | 0.06% | 通过 |
| 四面体，h = 0.6 mm，1486 节点 | space | 3371 | 3.5% | 10.8% | 7.2% | **不通过** |
| | time | 1486 | 0.01% | 0.04% | 0.02% | 通过 |

四面体网格的波形差异只有 3.5%，单看波形会判为通过，但二次谐波差了 11%，增量差 7%：满足「每个 2f 剪切波长 8 个单元」的初始筛选并不足以解析二次谐波。谱元网格在节点数约为四面体 3 倍的情况下三层都在 1% 以内。两种引擎上非线性增量的 A/2A 标度检查都通过（增量 A₂ 为 4 倍）。单例耗时：谱元基准约 11 s，三层复核合计约 100 s；四面体基准约 21 s。

## 5. 数据集（`dataset.npz` + `dataset.json`）

把所有已完成用例的接收线投影位移堆叠为一个 NPZ：`line_projected_m [cases, frames, points]`、`receiver_u_m [cases, receivers, components, frames]`（二维 2 个分量 x、z，三维 3 个分量 x、y、z，分量名见 `receiver_components`，同一数据集不能混合二维与三维）、`parameters [cases, P]` 与 `parameter_names`、`split`、`case_ids`、`scene_sha256`、`times_us`、`line_distance_mm`，并内嵌 manifest（来源扫描哈希、求解器源码哈希、阶次、单位、形状、划分计数）。要求各用例的时间网格、接收线采样与接收点集合一致，否则报错而不是静默对齐。

```python
from sweep.dataset import load_dataset, select
data = load_dataset('runs/s0-sweep/dataset.npz'); train = select(data, 'train')
train['parameters'], train['line_projected_m']      # 输入参数与目标波形
```

## 6. 测试

`sweep/test_sweep.py`：路径读写、网格与显式用例展开、联动、确定性编号；拉丁超立方每层恰一个样本且可复现；规格拒绝（含引擎与场景维度不符、未知引擎、非法收敛设置）；一次真实的四用例并行运行、篡改后的恢复只重跑一例、规格不一致被拒、汇总含两对 A/2A 标度检查全部通过、数据集形状与划分、2A 用例波形约为 A 用例两倍。

`sweep/test_convergence.py`：各引擎的细化变体（谱元三层、通孔子步加倍、6 阶与 CFL 下限跳过、四面体跳过阶次）；设置合并与拒绝；用例选择（数量、编号、参数筛选、自动带上线性对照）；比较指标（相同结果为零、整体放大 10% 时波形与 A₂ 相对差恰为 0.1/1.1、低于报告下限时按绝对差）；三维谱元（线性 + Murnaghan，含增量报告）与三维四面体两次真实扫描加收敛复核，核对报告、manifest 条目与汇总列。

## 7. 示例

`docs/examples/sweep-s0-frequency.json`：4 个载频 × A/2A 共 8 例，周期数随载频联动以固定脉冲时长 16 μs。`docs/examples/dataset-random-plates.json`：厚度、载频、幅值三维拉丁超立方 24 例，激励与接收随厚度联动，按厚度分组划分 18/3/3。两者都以 `plate-2d-s0-line.json` 为基准，单例约 1–2 分钟。`docs/examples/sweep-3d-convergence.json`：第 4 节的三维谱元扫描与收敛复核。

## 8. 尚未包含

网页端的扫描提交与进度、三维网页任务的接收线、收敛复核结果的缓存复用（每次 `converge` 都会重算所选层级）、Richardson 外推与收敛阶估计，以及在数据集上的任何训练或代理模型。

# 参数扫描与训练数据批处理（`sweep`）

`sweep` 包把一个二维平面应变基准场景展开为一组不可变用例，在隔离进程中并行、可恢复地运行，再汇总非线性导波指标并打包为训练数据集。它只调度 `spectral2d` 数值求解器，不训练模型，也不把任何结果路由给代理。

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

- `base_scene`：内联对象或相对规格文件的路径，必须是 `dimension: 2d_plane_strain` 的 v2 场景。
- `grid`：路径 → 取值列表，全因子展开；`cases`：显式覆盖字典列表，与网格叉乘；`sampling`：拉丁超立方或均匀随机，在 `ranges` 内按 `seed` 可复现地采样。路径用点号，支持列表下标，例如 `receiver_lines.0.from_mm.2`。
- `links`：线性联动，`target = scale × value(path) + offset`，用于让激励中心、接收点和接收线随板厚移动，或让周期数随载频变化以保持脉冲时长。
- 每个用例在展开后按求解器的规则完整校验（分辨率筛选按 h/阶次），校验后的场景与阶次的 SHA-256 决定用例编号，重复用例自动合并。
- `split`：按 `group_by` 的取值分组，同一组落在同一子集，避免同一几何同时出现在训练与测试集；比例按组数四舍五入。
- `analysis`：汇总时的载频（默认取激励载频）、接收线名、时间窗（`moving` 按 x/c_g 移动，群速度默认取该板厚理论模态 `mode_for_window` 的值；`fixed` 用 `start_us`/`end_us`）。

## 2. 命令

```sh
python -m sweep plan      --spec docs/examples/sweep-s0-frequency.json
python -m sweep run       --spec docs/examples/sweep-s0-frequency.json --output runs/s0-sweep --workers 2
python -m sweep run       --spec ... --output runs/s0-sweep --resume
python -m sweep summarize --campaign runs/s0-sweep
python -m sweep dataset   --campaign runs/s0-sweep --output runs/s0-sweep/dataset.npz
```

运行目录：`spec.json`、`manifest.json`（规格哈希、源码哈希、每个用例的参数、划分、阶段、结果哈希、节点数、步数、能量收支、耗时或错误）、`cases/<id>/{scene.json,settings.json,run.log,out/}`，`out/` 即 `spectral2d.run` 的输出。每个用例在独立进程中运行，`workers` 控制并发数，`seconds_per_case` 为单例超时。`--resume` 只重跑结果缺失或哈希不符的用例；规格改变会被拒绝。失败用例记录错误并继续，命令以退出码 2 结束。

## 3. 汇总（`summary.json` / `summary.csv`）

每个已完成用例：接收点在给定窗内的 f/2f 幅值与相位；接收线逐点 A₁、A₂，A₂ 与 A₂/A₁² 对距离的线性拟合与 R²，相位斜率相速度与包络群速度，并列出该板厚、该载频的理论模态相速度与群速度；能量收支、最大应变、节点数、步数、耗时。参数只在 `source.amplitude_mpa` 上相差一倍的用例自动配成 A/2A 对，输出 A₁≈2 倍、A₂≈4 倍的逐点检查与总判定。CSV 每行一个用例，便于直接绘图。

**线性对照与非线性增量。** 参数只在 `constitutive` 上不同的用例自动配成对照对（`linear` 与 `murnaghan`/`stvk`）：在同一网格与时间步上把非线性解减去线性解，再提取增量的 f/2f。短脉冲自身在 2f 处的线性频谱成分随幅值线性增长（倍率 2），会淹没真正的二次谐波；增量把它完全扣除。A/2A 配对因此给出两种判定：原始 2f（常因上述原因不通过）与非线性增量（应为 4 倍）。要用这一功能，在 `grid` 中加入 `"constitutive": ["linear", "murnaghan"]`。

谐波幅值依赖时间窗，A₂/A₁² 的斜率只在同一窗口设置下用于相对比较，不是材料 β。数据集的 `parameters` 只含数值路径，`categorical_parameters` 以 JSON 字符串保存本构等非数值参数。

## 4. 数据集（`dataset.npz` + `dataset.json`）

把所有已完成用例的接收线投影位移堆叠为一个 NPZ：`line_projected_m [cases, frames, points]`、`receiver_u_m [cases, receivers, 2, frames]`、`parameters [cases, P]` 与 `parameter_names`、`split`、`case_ids`、`scene_sha256`、`times_us`、`line_distance_mm`，并内嵌 manifest（来源扫描哈希、求解器源码哈希、阶次、单位、形状、划分计数）。要求各用例的时间网格、接收线采样与接收点集合一致，否则报错而不是静默对齐。

```python
from sweep.dataset import load_dataset, select
data = load_dataset('runs/s0-sweep/dataset.npz'); train = select(data, 'train')
train['parameters'], train['line_projected_m']      # 输入参数与目标波形
```

## 5. 测试

`sweep/test_sweep.py`：路径读写、网格与显式用例展开、联动、确定性编号；拉丁超立方每层恰一个样本且可复现；规格拒绝；一次真实的四用例并行运行、篡改后的恢复只重跑一例、规格不一致被拒、汇总含两对 A/2A 标度检查全部通过、数据集形状与划分、2A 用例波形约为 A 用例两倍。

## 6. 示例

`docs/examples/sweep-s0-frequency.json`：4 个载频 × A/2A 共 8 例，周期数随载频联动以固定脉冲时长 16 μs。`docs/examples/dataset-random-plates.json`：厚度、载频、幅值三维拉丁超立方 24 例，激励与接收随厚度联动，按厚度分组划分 18/3/3。两者都以 `plate-2d-s0-line.json` 为基准，单例约 1–2 分钟。

## 7. 尚未包含

三维求解器的扫描（worker 契约相同，待接入）、网页端的扫描提交与进度、自动网格/阶次收敛复核、以及在数据集上的任何训练或代理模型。

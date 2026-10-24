# 二维平面应变非线性谱元求解器与导波频散工具链

本文记录两个新模块：`spectral2d`（x–z 截面的平面应变超弹性谱元求解器）和 `dispersion`（Rayleigh–Lamb 频散、同步模态对、激励设计、谐波/沿程/速度/f–k 分析、厚度向模态分解）。两者都是纯数值与解析工具，不含神经网络，不改变离线便携包和被哈希固定的源文件。

## 1. 为什么需要二维

非线性 Lamb 波研究的绝大多数数值工作在平面应变下进行：无限宽板、沿 y 不变、u_y = 0。相比三维，同样的板长与频率下成本低两到三个量级，使百毫米量级传播距离、数十周期脉冲和数百个接收点成为日常计算，也使几千次参数扫描和代理模型的训练数据成为可能。

## 2. 场景：`dimension: "2d_plane_strain"`

二维场景就是场景 v2 加 `"dimension": "2d_plane_strain"`。几何沿用 `plate` / `notch`（缺口贯穿宽度，在截面上是顶面的矩形切口），`width_mm` 保留但无物理作用，所有力、质量和能量按**每米宽度**计。限制：激励面、固定面、边界面只能是 `xmin/xmax/zmin/zmax`；激励与接收方向的 y 分量必须为零；梳状方向不能是 y。信号、空间分布、多源、对称/反对称面、吸收层与三维完全相同（见 `docs/solver-excitation-v2.md`）。

新增 `receiver_lines`（目前仅二维）：

```json
"receiver_lines": [
  { "name": "top", "from_mm": [60, 0, 2], "to_mm": [220, 0, 2], "count": 161, "direction": [1, 0, 0] }
]
```

每条线最多 1024 个点，等距采样，结果里按线成组保存，直接用于沿程谐波、相速度/群速度和 f–k 分析。

## 3. 求解器

- `spectral2d/mesh.py`：x–z 截面的共形 GLL 四边形网格（板与贯穿缺口），GLL 节点质量，张量 Lagrange 插值。
- `spectral2d/solver.py`：把二维位移梯度嵌入 3×3 矩阵（y 行列为零），直接调用 `spectral3d.solver.constitutive`，因此线性、StVK、Murnaghan 三种本构与三维逐位同源；显式速度 Verlet；多源、任意信号、对称约束、速度隐式吸收阻尼与三维一致。
- `spectral2d/loads.py`：边载荷按 GLL 一维求积与区域/分布权重装配；约束与阻尼复用 `physics.boundaries`。
- `spectral2d/run.py`：

```sh
python -m spectral2d.run --scene docs/examples/plate-2d-s0-line.json --output out/s0 --degree 4 [--fields] [--csv]
```

输出 `result.json`（场景、诊断、点接收波形）、`lines.npz`（各接收线的坐标、沿程距离、二维位移与方向投影）、可选 `fields.npz`（全场）与 CSV。Python 接口 `spectral2d.run.simulate(scene, degree, keep_fields)` 返回网格、接收点与计算记录。

### 验证（`spectral2d/test_plane_strain.py`）

1. **核与三维逐位同源**：任意沿 y 不变的位移场，三维内力沿 y 求和等于二维内力乘宽度，能量亦然，三种本构均到 1e-15 相对；刚体平面转动零力零能。
2. **瞬态与三维一致**：三维板两侧 y 面施加对称边界（u_y = 0）、均匀端面加载，与二维解在全部节点上一致到 1e-10 相对，三维 u_y 为零到 1e-12。这同时是上一轮对称边界的独立复核。
3. **半厚度对称**：z 中面对称的半模型与整板一致到 1e-10，说明用 `symmetry`/`antisymmetry` 可以分别只保留 S 族或 A 族。
4. **S0 相速度对 Rayleigh–Lamb**：120 mm 板、0.5 MHz 五周期脉冲、远端 30 mm 吸收层、顶面 41 点接收线，由沿程相位斜率得到的相速度与解析 S0 相差小于 2%，群速度相差小于 5%，相位拟合 R² > 0.999；记录末段残余回波小于直达波的 5%。
5. 梳状/圆盘边载荷长度精确，接收线展开与越界判定。

## 4. 频散工具链（`dispersion/`）

### 4.1 `lamb.py`
Rayleigh–Lamb 特征函数以复数 p、q 统一书写，D_S/q 与 D_A/p 对实波数为实值且在 q = 0 或 p = 0 处无伪根；双曲区间按 exp(−|Im|) 缩放避免溢出；在慢度网格上找变号后用 Brent 法精化；随频率跟踪并连接。零群速度点处两支同时出现，按同一阶编号，反向支加后缀 `b`（铝板 S1 的 ZGV 点约 2.87 MHz·mm）。输出每个模态的 fd、频率、相速度、波数、群速度、截止与 ZGV 位置，以及厚度向振型（`mode_shape`，S 族 u_x 偶、u_z 奇；两分量正交相位）。

```sh
python -m dispersion.lamb --thickness-mm 2 --fd-max 8 --output lamb.json --csv lamb.csv
```

测试核对：S0 低频极限等于板速度 2c_T√(1−c_T²/c_L²)；A0 低频与薄板弯曲理论一致；高频 S0、A0 趋于 Rayleigh 速度；A1 截止 fd = c_T/2，S2 截止 fd = c_L/2；S 族在 fd* = 1/√(1/c_T² − 1/c_L²)（本材料 3.559 MHz·mm）处相速度恰为 c_L，S2 在 2fd* 处亦然；所有根满足特征方程。

### 4.2 `matching.py`
给定频散结果，列出 (f, 2f) 同步模态对：相速度曲线精确交叉，以及在容差内的 fd 区间。材料非线性的二次谐波为对称族，默认次级模态取 S 族。铝板得到的两类典型结果：低 fd 的 S0–S0 近似匹配区间，以及 S1(fd*)–S2(2fd*) 的 c_L 同步点。工具同时报告两模态的群速度，但不据此筛除。

### 4.3 `design.py`
为选定模态与频率给出波长、梳状间距与齿宽、楔块角度（给定楔块声速，低于相速度时才存在）、半厚度模型应采用的对称类型、脉冲时长与 −6 dB 带宽、到给定距离的群时延、建议直达波时间窗（含带宽内群速度差造成的展宽）、传播波长数。

### 4.4 `analysis.py`
- `harmonic_coefficients`：与浏览器端完全相同的 Hann 加权、去均值、准确频率复系数；
- `line_harmonics`：沿接收线逐点提取，支持固定窗或按 x/c_g 移动的窗；
- `accumulation_fit`：A₂ 与 A₂/A₁² 对距离的线性拟合与 R²。时间窗与包络的加权因子在 A₂/A₁² 中不抵消（A₁ 被平方），所以斜率与 β 成正比但不等于 β；用于同一窗口设置下的相对比较，绝对值需要标定，工具只报告不解释；
- `amplitude_scaling`：A 与 2A 两次计算的 A₁ 倍率≈2、A₂ 倍率≈4 检查；
- `line_velocities`：由展开相位斜率得相速度，由包络峰到达时间得群速度；
- `fk_spectrum`、`mode_overlay`：接收线的 f–k 幅值谱与理论曲线叠加数据（向 +x 传播的波出现在正 k）；
- `thickness_decomposition`：按给定振型最小二乘分解厚度向复位移，报告残差。

注意：时间窗边缘处基频拖尾的截断会向 2f 泄漏约 1e-7 到 1e-8 的相对量；在 A₂/A₁ ≥ 1e-4 的真实非线性水平下可以忽略，但在极弱信号或人工构造的测试中会显现。合成数据测试覆盖幅值与相位恢复、相速度与 f–k 峰、累积斜率与幅值标度、模态分解。

## 5. 推荐工作流

1. `dispersion.lamb` 算出板的频散；`dispersion.matching` 选同步模态对，或在低 fd 用 S0。
2. `dispersion.design` 得到波长、梳状间距/楔块角、半模型对称类型和直达波窗。
3. 写二维场景：半厚度模型加 `symmetry`/`antisymmetry`，远端吸收层，顶面接收线，必要时梳状分布。
4. `spectral2d.run` 求解，再用 `line_harmonics` → `accumulation_fit` 看 A₂/A₁² 随距离的斜率，用两次幅值做 `amplitude_scaling`，用 `fk_spectrum` 核对模态纯度，用 `thickness_decomposition` 核对厚度向成分。
5. 换网格或阶次重算一次，作为该工况的收敛记录。

## 6. 示例运行记录（开发环境，2026-10-24）

`docs/examples/plate-2d-s0-line.json`：300 mm × 2 mm 铝板，0.5 MHz 八周期 Tukey 脉冲端面加载，远端 60 mm 吸收层，顶面 60–220 mm 共 161 点接收线，x = 150 mm 处 9 点厚度向接收线。4 阶、h = 1 mm：10 809 节点、600 单元、9 800 内部步，单核约 106 s，能量收支误差 6.3e-5。

| 核对项 | 结果 |
| --- | --- |
| 沿程相位斜率得到的 S0 相速度 | 5344.9 m/s，Rayleigh–Lamb 为 5336.5 m/s，相差 0.16%，相位拟合 R² = 1.0000 |
| f–k 谱在 0.5 MHz 的峰值波数 | 0.5854 rad/mm，理论 0.5887 rad/mm |
| 厚度向模态分解 | S0 占比 99.96%，A0 分量 1e-16 量级，残差 3.8e-4 |
| A₂ 随距离 | 3–7 pm 之间振荡而非线性增长。fd = 1 MHz·mm 处 S0(2f) 与 S0(f) 相速度失配 11%，拍长约 42 mm，与同步模态工具给出的“S0–S0 仅在 fd ≤ 0.42 MHz·mm 内匹配”一致 |

要观察累积增长，应把载频降到 fd ≤ 0.4 MHz·mm，或改用 S1(3.56 MHz·mm)–S2 同步点并用梳状/楔块激励选模。

## 7. 尚未包含

二维结果的浏览器查看器与网页提交入口、层合板与各向异性、流体加载、周期边界、二维四面体/三角形网格与任意截面几何、以及把二维求解器用作训练数据生成器的批处理框架。

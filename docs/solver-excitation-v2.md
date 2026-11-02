# 求解器扩展：任意信号、多源激励、对称面与吸收层（场景 v2）

本文记录两套本地数值求解器（`solid/fem.py` 四面体显式有限元，`spectral3d/solver.py` 与 `spectral3d/implicit.py` 谱元）新增的激励与边界能力，以及承载它们的场景文件版本 2。离线便携包所用的冻结图网络不受影响：它只读取版本 1 场景，`offline/model/model.json` 固定校验的源文件（`solid/schema.py`、`spectral3d/mesh.py`、`spectral3d/curved_mesh.py`、`graphwave/*`）一个字节都没有改动。

## 1. 兼容性

- 版本 1 场景（`workai.hyperelastic.solid.v1`）走原来的代码路径。两套求解器上板、缺口、管（显式）与通孔（隐式 Newmark）的位移场、接收点波形、能量记录和诊断字典均与修改前**逐位一致**（见本文末尾的核对记录）。
- 版本 2 场景只是版本 1 场景把 `schema` 改为 `workai.hyperelastic.solid.v2`，再按需增加下面的字段。没有新增字段的 v2 场景与 v1 结果一致到舍入增长量级（约 1e-10 相对），差别仅来自信号的向量化采样。
- 校验入口统一为 `physics.scene.validate_scene`：v1 原样交给 `solid.schema.validate_scene`；v2 先剥离扩展字段，用 v1 规则校验核心与每一个激励，再校验扩展字段。`solid/cli.py validate`、两套 worker、五组对照（`solid/study.py`）都已切换到这个入口。

## 2. 场景 v2 字段

### 2.1 `source.signal`（可选）与 `sources[].signal`

| `type` | 额外字段 | 说明 |
| --- | --- | --- |
| `tone_burst` | `window` ∈ hann / tukey / gaussian / rectangular；tukey 需 `tukey_alpha`(0.05–1)；gaussian 需 `gaussian_width`(0.05–0.5) | 使用激励自身的 `frequency_mhz`、`cycles`、`delay_us`、`phase_deg`。hann 等于版本 1 的公式，不重新归一化 |
| `gaussian_pulse` | `bandwidth`(0.1–2)，为 −6 dB 相对带宽 | 中心频率取 `frequency_mhz`；包络在 ±3.5σ 截断 |
| `ricker` | 无 | 峰值频率取 `frequency_mhz`，支撑 3/f |
| `chirp` | `end_frequency_mhz`、`duration_us`、`window` ∈ hann / rectangular | 起始频率取 `frequency_mhz`，线性调频 |
| `multi_tone` | `components:[{frequency_mhz, weight, phase_deg}]`(1–8)、`duration_us`、`window` | `frequency_mhz` 必须等于某一分量，作为 f/2f 分析载频 |
| `samples` | `sample_step_us`、`values`(8–20000)、`normalize` | 用户时间序列，线性插值到内部步长；默认归一化到单位峰值，不归一化时峰值不能超过 1 |

约定：`delay_us` 永远是信号支撑的起点。`gaussian_pulse`、`chirp`、`multi_tone` 归一化为单位峰值，所以 `amplitude_mpa` 始终是峰值牵引；`tone_burst` 保持版本 1 的含义（窗峰值 ≤ 1）。非 `tone_burst` 信号不使用 `cycles`，但该字段仍需在 1–20 内。

校验：信号支撑必须落在记录时长内；网格筛选改为对信号的 −30 dB 带宽与 2f 取大值，要求对应剪切波长至少 8 个单元。谱元 worker 与以前一样用 h/阶次 作为筛选尺寸。

### 2.2 `source.profile`（可选）与 `sources[].profile`

| `type` | 字段 | 说明 |
| --- | --- | --- |
| `uniform` | 无 | 版本 1 行为 |
| `gaussian` | `sigma_mm` | 以 `center_mm` 为中心、面内距离的高斯权重 |
| `comb` | `pitch_mm`、`teeth`(1–64)、`tooth_width_mm`、`along` ∈ x/y/z、`alternating_sign` | 梳状换能器：沿 `along` 排列的齿，`alternating_sign` 为真时相邻齿反号 |

区域（整面、`radius_mm` 圆盘、`surface_patch`）仍按版本 1 规则选取，分布在区域内加权。谱元在 GLL 面节点上取值，四面体在表面三角形质心上取值。恰好落在区域或齿边缘的节点按一半权重计入，因此齿边落在节点上时积分面积精确。诊断中同时给出 `loaded_area_mm2`（指示函数积分）与 `profile_weighted_area_mm2`（加权积分），请据此检查名义面积与离散面积的差别。

### 2.3 `sources`（可选，最多 15 个额外激励）

每个额外激励与主激励结构相同，并多一个唯一的 `name`。各自的信号、时延、幅值、方向与区域独立，节点力按时间步叠加。阵列与相控延时直接用多个激励的 `delay_us` 表达。

### 2.4 `boundaries`（可选，最多 8 项）

| `type` | 字段 | 实现 |
| --- | --- | --- |
| `symmetry` | `face` | 该面节点的法向位移分量置零，切向自由 |
| `antisymmetry` | `face` | 该面节点的切向位移分量置零，法向自由 |
| `absorbing` | `face`、`length_mm`、`ramp` ∈ linear / quadratic / cubic，再加 `attenuation_db`(10–120) 或 `alpha_max_per_s` | 质量比例阻尼层（ALID） |

限制：同一面只能有一种边界；不能与 `boundary.fixed_face` 相同；吸收层不能放在有激励的面上；管只能用 X 端面；吸收层长度不超过该方向尺寸的一半且不小于网格尺寸。激励若作用在被约束的位移分量上会在装配时报错（例如 Y 向载荷遇到 `ymax` 的 `symmetry`，此时应改用 `antisymmetry`）。

## 3. 数值实现

### 3.1 吸收层

节点阻尼系数 α(x) = α_max · (d/L)^p，d 为节点深入吸收层的距离，p 为渐变阶数。默认由设计衰减 A（dB）给出

α_max = (p + 1) · c_L · ln(10) · A / (20 L)，

即波以纵波速度穿过吸收层并返回时，幅值按 exp(−α_max L / ((p + 1) c_L)) 衰减。这是设计目标，不是实际反射系数；实际残余回波随频率、层长与渐变而变。

显式速度 Verlet 中阻尼项对速度隐式处理：

v⁺ = (v + Δt/2 · (a + (f_ext − f_int)/m)) / (1 + Δt·α/2)，a⁺ = (f_ext − f_int)/m − α v⁺，

因此稳定步长不变。隐式 Newmark 中阻尼进入残差 (M + Δt/2 C)(u⁺ − û) + Δt²/4 (f_int − f_ext + C(v + Δt/2 a)) 及预条件矩阵的对角项，原有的 ‖A⁻¹‖ ≤ 1 收敛判据仍成立。

能量记录增加耗散列：`energy_samples` 每行为 [t, 动能, 应变能, 外力功, 耗散]，能量收支检查改为 |K + W + D − 外力功| / 尺度 ≤ 1%。

测试中 60 mm 长、2 × 2 mm 截面铝杆，0.5 MHz 四周期脉冲，20 mm 三次渐变 60 dB 吸收层：谱元与四面体两套求解器上远端回波均低于自由端回波的 10%，直达波在 1e-6 相对精度内不变。

### 3.2 对称与反对称面

节点按坐标落在平面上判定（相对容差 1e-7）。测试用整板与半板对照：对称/反对称半模型的全部节点位移与镜像对称的整板模型一致到 1e-10（谱元）与 1e-9（四面体）相对精度。

### 3.3 多源叠加

线性对照下两个激励的响应等于各自单独响应之和（相对 1e-9 以内），由测试保证。

## 4. 结果文件

- 诊断新增：`excitation_model`（`legacy_single_source` 或 `multi_source_v2`）、`sources`、`constraints`、`damping`、`dissipated_energy_j`、`energy_columns`、`signal_peaks`。
- 谱元结果的 `loaded_faces` 改为所有三个节点都承受牵引的表面三角形，`excitation` 字段给出各激励的信号摘要。
- 版本 1 任务的结果文件内容不变。

## 5. 示例

`docs/examples/plate-v2-absorbing-comb.json`：60 mm 铝板半模型（`ymax` 对称），Tukey 窗梳状激励加一个高斯脉冲圆盘激励，两端 10 mm 吸收层。该文件可直接通过 `python -m solid.cli validate` 校验，也可作为三维页面的导入配置。

## 6. 测试

```sh
python -m unittest physics.test_signals physics.test_scene
python -m unittest spectral3d.test_excitation solid.test_excitation
```

新增 21 项测试，覆盖：信号与版本 1 公式一致、归一化与支撑、采样信号插值、v1 原样通过、v2 示例、各类拒绝、v2 无扩展时与 v1 一致、对称/反对称半模型、线性叠加、吸收层回波抑制与能量收支、梳状/高斯/圆盘面积、隐式通孔求解器的阻尼。原有 25 项求解器物理测试与 84 项 Node 测试全部通过。

修改前后各求解器在版本 1 场景上的逐位核对（板、缺口、管的显式谱元；通孔的隐式谱元；板、固定端管、缺口的四面体有限元）在本次提交的开发环境中执行，位移场、接收点与诊断字典完全相同。

## 7. 尚未包含

压力型随动载荷、体力、位移/速度型激励、周期边界、弹性支承、PML、各向异性材料、二维平面应变求解器，以及前端表单对这些字段的编辑界面。前端目前可以通过导入 JSON 配置提交 v2 场景。

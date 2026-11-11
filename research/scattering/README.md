# 缺口与通孔的散射

2026-11-11。核验离线工作台默认引擎“物理 AI · 冻结网络”计算导波遇到缺口和通孔时的散射。两类缺陷各用一个与软件无关的参考：

- **通孔：** 平面应力解析解（Pao & Mow 级数解），完全不经过软件；
- **矩形缺口：** 本仓库另一套二维平面应变谱元求解器 `spectral2d`，在比软件细 2–4 倍的网格上计算。它与网络不共用代码路径，自身收敛程度用两级网格核对。

缺口算例走工作台的同一条路径：请求校验 → `offline.worker` → 完整 float64 记录 → 结果页取波形用的 `offline.recording.traces`。通孔网格有 8.6 万个节点，超过工作台 3 万节点的显示上限，所以直接调用 worker 内部同一个网络推演函数（`graphnet.rollout`），记录仍用 `offline.recording.traces` 读取。

## 结论

NOTCH_SUMMARY

- **通孔（S0 入射，ka = 1.16）：** 孔边 5 个方位和 r = 2a 处 3 个方位，共 8 个接收点。
  - 软件的散射场与解析解相差 0.1–0.6%（相对 L2），散射峰值相差 ≤ 0.4%；
  - 总场相差 0.1–0.4%，其中最多约 0.2% 来自三维板中 S0 的真实频散，平面应力模型不含这部分。
- **只核验了线性响应。** 缺口和孔处的非线性散射（如二次谐波）没有独立参考，未作定量核验。

## 1. 通孔：平面应力解析解

**算例。** 铝板 48 × 48 × 1 mm，中心有半径 a = 4 mm 的通孔。

- **边界：** 两个 y 面设对称面，相当于沿 y 方向间隔 48 mm 的一列孔，入射波是严格的直峰 S0 平面波；
- **激励：** xmin 端面施加均匀轴向牵引，0.25 MHz、2 周期 Hann 脉冲，载频处 ka = 1.16；
- **软件计算：** h 2.5 mm、4 阶，86240 个节点，记录 20 μs，另算一个无孔板（32805 个节点）提供入射波；
- **接收点：** 中面上孔边 r = a + 0.02 mm 的 θ = 180°、135°、90°、45°、0°，以及 r = 2a 的 180°、90°、0°。θ 从 +x 量起，θ = 180° 是迎波面。

**解析解**（`scripts/hole_theory.py`）。低频 S0 在薄板中是二维纵波，常数取平面应力值 λ' = 2λμ/(λ + 2μ)。Diligent 等（2002）研究 S0 被通孔散射时，同样限于 S0 不频散的低频范围（据摘要）。在这一范围内，沿板厚的应力和位移分布简单，可以按二维平面应力处理。

- **级数形式：** 入射 φ = e^{iαx}，散射势 φ_s = Σ A_n H_n(αr) cos nθ，ψ_s = Σ B_n H_n(βr) sin nθ；
- **求解：** 孔边 σ_rr = σ_rθ = 0，逐阶解 2 × 2 方程（Pao & Mow 1973）。
- **自检：**
  - ka → 0 时孔边 90° 环向应力为 2.6550 σ₀，等于静力 Kirsch 解 3 − ν（直峰平面波带有 σ_yy = ν σ_xx，相当于双轴拉伸）；
  - 孔边牵引残差 9e-16；
  - 3 MHz、r = 2a 处，级数取 40 阶与 80 阶相差 1e-4。

**对照方法**（`scripts/analyze_hole.py`）。

- **入射波：** 取无孔板在孔中心处的 u_x，作为入射波；
- **预测：** 乘以解析传递函数，得到各接收点的总场和散射场；
- **软件散射场：** 有孔与无孔两次计算之差；
- **比较窗口：** 从 0 到 10.86 μs，即孔散射的波经 y 对称面反射（镜像孔）回到孔边之前；
- **入射波截断：** 入射波在远端反射到达孔中心（13.25 μs）之前平滑截断，频带取到 3 MHz，3 MHz 以上的频谱只占峰值的 6e-7。

**结果：**

| r (mm) | θ | 散射场 / 入射峰值 | 总场误差 | 散射场误差 | 散射峰值比（软件/解析） |
|---|---|---|---|---|---|
| 4.02 | 180° | 0.89 | 0.09% | 0.11% | 0.9992 |
| 4.02 | 135° | 0.71 | 0.16% | 0.28% | 0.9992 |
| 4.02 | 90° | 0.20 | 0.13% | 0.47% | 0.9961 |
| 4.02 | 45° | 0.53 | 0.37% | 0.37% | 1.0012 |
| 4.02 | 0° | 1.01 | 0.32% | 0.23% | 0.9978 |
| 8 | 180° | 0.63 | 0.20% | 0.09% | 0.9994 |
| 8 | 90° | 0.21 | 0.12% | 0.57% | 0.9956 |
| 8 | 0° | 0.64 | 0.12% | 0.27% | 0.9992 |

误差来源：

- **S0 频散：** 无孔板中 S0 的拟合相速度为 5427 m/s，平面应力值为 5433 m/s，两者相差 0.1%。这是 1 mm 板在 0.25 MHz 的真实频散，接收点离孔中心越远，这部分越明显；
- **镜像波：** r = 8 mm、θ = 90° 的接收点离镜像孔最近，窗口末尾约 0.5 μs 已混入镜像孔的散射波，因此误差稍大。

![通孔散射](figures/fig3-hole.png)

NOTCH_SECTION

## 4. 适用范围

- **只核验线性响应：** 缺口和孔处的二次谐波等非线性散射没有独立参考，未作定量核验。
- **通孔：** 只测了一个 ka（1.16）和一个板厚（1 mm）。平面应力模型只在低 fd 下成立，更高频率需要三维参考。
- **缺口：** 只测了贯穿宽度的矩形缺口（平面应变）。有限宽度的三维缺口、裂纹（闭合接触）不在本页范围。
- **文献：** 期刊全文在本环境无法读取，文献只用于物理规律（深度、宽度趋势）的定性对照，定量对照用的是上面的独立参考解。

## 5. 计算时间

同一台 4 核云机器，同时运行 3–5 个任务，所以以下是偏慢的数字：

| 计算 | 节点数 | 记录 | 三组控制总耗时 |
|---|---|---|---|
| 缺口，h 1 mm | 2.8 万 | 45 μs | 30–48 min |
| 缺口，h 1.5 mm | 1.3 万 | 45 μs | 12 min |
| 无缺口板，h 1 mm | 2.0 万 | 45 μs | 15 min |
| 通孔板，h 2.5 mm | 8.6 万 | 20 μs | 84 min |
| 无孔板，h 2.5 mm | 3.3 万 | 20 μs | 11 min |

## 参考文献

1. Y.-H. Pao, C.-C. Mow, *Diffraction of Elastic Waves and Dynamic Stress Concentrations*, Crane Russak (1973)，第 4 章：圆孔对平面 P 波的散射级数解（书目）。
2. O. Diligent, T. Grahn, A. Boström, P. Cawley, M. J. S. Lowe, *The low-frequency reflection and scattering of the S0 Lamb mode from a circular through-thickness hole in a plate: Finite Element, analytical and experimental studies*, J. Acoust. Soc. Am. 112(6), 2589–2601 (2002)；[DOI 10.1121/1.1512292](https://doi.org/10.1121/1.1512292)（摘要：限于 S0 不频散的低频范围，有限元、解析解（模态叠加）与实验三方对照）。
3. M. J. S. Lowe, O. Diligent, *Low-frequency reflection characteristics of the s0 Lamb wave from a rectangular notch in a plate*, J. Acoust. Soc. Am. 111(1), 64–74 (2002)（摘要：反射系数随缺口宽度周期变化，来自缺口两端反射的干涉）。
4. M. J. S. Lowe, P. Cawley, J.-Y. Kao, O. Diligent, *The low frequency reflection characteristics of the fundamental antisymmetric Lamb wave a0 from a rectangular notch in a plate*, J. Acoust. Soc. Am. 112(6), 2612–2622 (2002)；[DOI 10.1121/1.1512702](https://doi.org/10.1121/1.1512702)（摘要：A0 反射系数随缺口宽度呈余弦型周期变化，来自缺口两端反射的干涉；反射随频率和深度增大）。
5. G. Kirsch (1898)：无限板圆孔的静力应力集中（经典结果，用于解析解的静力极限自检）。

## 复现

```sh
export WORKAI_VERIFY_JOBS=/path/to/jobs          # 记录 0.6–1.9 GB/算例，放在仓库外
for n in plate d0.4-w1 d0.8-w1 d1.2-w1 d0.8-w2 d0.8-w4 d0.8-w6 d0.8-w9; do
  python research/scattering/scripts/notch.py reference $n      # 二维参考，h 0.5 与 0.25 mm
  python research/scattering/scripts/notch.py network $n 1      # 软件，h 1 mm
done
python research/scattering/scripts/notch.py network plate 1.5 && python research/scattering/scripts/notch.py network d0.8-w1 1.5
python research/scattering/scripts/analyze_notch.py
python research/scattering/scripts/hole.py run plate && python research/scattering/scripts/hole.py run hole
(cd research/scattering/scripts && python hole_theory.py && python analyze_hole.py)
python research/scattering/scripts/plots.py                     # 需要 matplotlib
```

结果在 [`results/notch.json`](results/notch.json) 与 [`results/hole.json`](results/hole.json)。参考数组（`reference/*.npy`）可重新生成，不入库。

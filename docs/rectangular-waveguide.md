# 有限宽铝截面的导波核对

本模块独立计算无限长、自由侧边、均匀矩形截面波导的线性特征模态，用来核对三维瞬态结果中的基频形状和传播相位。它使用与瞬态模型相同的密度与二阶弹性常数，不从瞬态波形拟合波数，也不修改既有分析时间窗。适用范围为无孔直板的矩形截面；管、孔、缺口、有限端面附近及任意 CAD 需要相应的模型。

## 方程与数值方法

采用半解析有限元（SAFE），沿 X 方向的约定为 `u = Re[φ(y,z) exp(iωt − ikx)]`。YZ 截面使用三角形一次单元、每节点三个位移分量，工程剪应变采用 `(xx, yy, zz, yz, xz, xy)` 排列。

```text
B(k) = B0 − i k B1
K(k) = K0 − i k G + k² K2
G = ∫(B0ᵀ C B1 − B1ᵀ C B0) dA
K(k) φ = ω² M φ
```

M 为一致质量矩阵。每个三角形三点积分可精确积分本实现的质量项和刚度项。稀疏 Hermitian 特征求解使用负的谱位移；该位移用于数值求解，不改变物理刚度或模态频率。最初用零位移时，长波挠曲近零特征值使独立长波极限测试不通过，已修正并重新生成全部色散证据。

从长波杆的伸缩分支开始，采用质量加权 MAC 连续跟踪到指定频率，再用频率求根定位 k。跟踪 MAC 低于 0.8 或相对特征残差高于 10⁻⁶ 时停止。群速度由该分支频率对 k 的中心差分计算。这里只跟踪一条分支；在 2f 频率上它可能与其他形状混合，不能替代全部接收模态的搜索。

截面尺寸 4 × 1 mm，密度 2700 kg/m³，λ=58.1 GPa、μ=26.1 GPa。三个独立截面网格的步距设定为 0.25、0.125、0.0625 mm，节点数为 85、297、1105。

| 频率 | 最细网格 k / rad·m⁻¹ | 相速度 / m·s⁻¹ | 群速度 / m·s⁻¹ | 最后两层 k 差异 |
| --- | --- | --- | --- | --- |
| 0.3 MHz | 375.0970945 | 5025.2471 | 4843.8446 | 0.00089122% |
| 0.6 MHz | 959.2152412 | 3930.2036 | 1630.3838 | 0.11817628% |

两频率均通过本研究 1% 的截面网格比较门槛。所跟踪同一分支的 Δk=209.0210522 rad/m，相干长度 π/|Δk|=15.03002985 mm。**这不是所有 2f 模态的最优相位匹配结果**：未枚举其他分支，也未计算非线性模态耦合积分。

## 与完整三维场比较

`tools/compare-waveguide-field.py` 首先核对完成研究、子任务、原始与压缩 float64 场的 SHA-256，要求同网格、时间采样和求解器源码。将每个接收 X 平面的整个三维截面用四面体重心插值映射到 SAFE 节点，独立用 NumPy 计算带加权均值移除的 Hann 复数频率系数。原有三个直达波窗保持不变。

基频使用线性控制；材料 2f 使用完整 Murnaghan 场减去几何非线性控制后的时域差，再投影频率。设截面复数场为 q，候选模态为 φ：

```text
a = (φᴴ M q) / (φᴴ M φ)
shape mismatch = ||q − aφ||M / ||q||M
```

这是单一截面形状的拟合残差，**不是模态能量或功率占比**。有限脉冲的受迫响应、其他模态和近场都可能使残差增加。相邻截面的 a 用来核对传播相位；只有间距小于该候选半波长时才通过无混叠筛查。基频候选标签还同时要求各截面形状和相位差异 ≤5%、截面网格 k 差异 ≤1%。它不声称纯模态分离或材料常数反演。

原始色散、模态向量 NPZ、源码快照及摘要位于 `/workspace/workai-remote/validation/20261023-rectangular-waveguide-v2`。初版证据保留，当前采用重新通过长波极限测试后的 v2。瞬态细化和最终截面比较见 [solid-propagation.md](solid-propagation.md)。

## 复现与引用边界

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /workspace/workai-remote/solid/venv/bin/python -m unittest waveguide.test_safe waveguide.test_field -v
/workspace/workai-remote/solid/venv/bin/python tools/run-waveguide-dispersion.py --help
/workspace/workai-remote/solid/venv/bin/python tools/compare-waveguide-field.py --root /workspace/workai-remote --study COMPLETED-STUDY --dispersion DISPERSION.json --output NEW-COMPARISON.json
node tools/link-waveguide-reference.mjs /workspace/workai-remote NEW-COMPARISON.json DISPERSION.json
```

六项独立测试覆盖一致质量与仿射应变能、刚体转动、Hermitian 性与稠密特征解、长波杆速度及旋转截面不变性、复数形状拟合、三维仿射场截面插值和解析频谱相位。这里只验证各自所述性质，不等于实验验证。

附加引用文件不修改原始求解结果。服务器核对完成研究摘要、原始色散记录、控制任务身份、几何材料、点位和原分析窗；前端仅在这些引用匹配时显示。离线沿程报告保留相同记录和范围说明。改变单点查看器的方向或窗口不会改变上方固定的沿程比较，也不会继承原来的离散精度标签。

安装只运行小规模物理测试；不自动启动新的瞬态研究、数据生成或训练。神经网络可使用守恒方程、本构、边界和初始条件，但不能把本案例未证明的单模态增长律强加给任意几何。

# 非线性纵向、横向运动方程与训练约束

这些约束已经接入 `graphwave/modal_calibration.py` 的实际优化目标和候选验收。它们约束三维运动、边界力和能量；物理残差合格不等于 5 ms 波形误差合格。独立 COMSOL 比较仍单独执行，未通过全部验收的权重不替换离线软件的默认权重。

## 三维运动与本构

使用参考构形坐标 (X_J)、参考密度 \(\rho_0\) 和第一 Piola 应力 \(P_{iJ}\)：

\[
\rho_0\ddot u_i=\sum_{J=1}^3\partial_{X_J}P_{iJ}+b_i,\qquad i=1,2,3.
\]

\[
G_{iJ}=u_{i,J},\quad F=I+G,\quad
\varepsilon=\tfrac12(G+G^T),\quad Q=\tfrac12G^TG,\quad E=\varepsilon+Q.
\]

代码使用最后一个表达式计算 Green 应变，避免微小位移下用 \(F^TF-I\) 相减造成精度损失。材料常数沿用场景中的铝 Murnaghan 参数，训练不修改其数值。明确采用本仓库的常数约定：

\[
a=n,\qquad b=m-\tfrac n2,\qquad c=l-m+\tfrac n2,
\]

\[
W(E)=\frac\lambda2(\operatorname{tr}E)^2+\mu\operatorname{tr}(E^2)
+\frac a3\operatorname{tr}(E^3)+b\operatorname{tr}E\operatorname{tr}(E^2)
+\frac c3(\operatorname{tr}E)^3,
\]

\[
S=\lambda\operatorname{tr}E\,I+2\mu E
+[b\operatorname{tr}(E^2)+c(\operatorname{tr}E)^2]I
+2b\operatorname{tr}E\,E+aE^2,\qquad P=FS.
\]

例如第一方向的完整运动方程为

\[
\rho_0\ddot u_1=\sum_{J,k}\left[u_{1,kJ}S_{kJ}
+(\delta_{1k}+u_{1,k})\partial_{X_J}S_{kJ}\right]+b_1.
\]

第二、第三方向分别替换自由指标 1 为 2、3。实现通过解析链式法则计算全部 \(u_{i,JK}\)，保留混合导数、剪切、泊松耦合和各方向乘积。

## 纵向与两个横向投影

在给定单位方向 \(e_L\) 与正交的 \(e_{T1},e_{T2}\) 下，分别约束

\[
r_L=e_L\cdot[\rho_0\ddot u-\operatorname{Div}P-b],\quad
r_{T1}=e_{T1}\cdot[\rho_0\ddot u-\operatorname{Div}P-b],\quad
r_{T2}=e_{T2}\cdot[\rho_0\ddot u-\operatorname{Div}P-b].
\]

三个方程共用完整三维应力，不能用互不耦合的一维波方程替代。这里的“纵向/横向”指所选坐标方向的投影；有限宽板中的投影不是自动识别出的纯传播模态。

同时分开检查五个本构分支。令 \(\mathcal L(A)=\lambda\operatorname{tr}A\,I+2\mu A\)，\(S_N\) 是上式中由 \(a,b,c\) 产生的部分：

\[
P_{\rm lin}=\mathcal L(\varepsilon),\quad
P_{\rm geo}=\mathcal L(Q)+G\mathcal L(E),\quad
P_{\rm mat}=(I+G)S_N(E),
\]

\[
P_{\rm StVK}=P_{\rm lin}+P_{\rm geo},\qquad
P_{\rm full}=P_{\rm lin}+P_{\rm geo}+P_{\rm mat}.
\]

每个分支均检查三个投影。几何和材料增量在**同一个制造位移场**上计算，使用各自的量级归一化；它们是本构算子检查，不能等同于三个不同瞬态解相减得到的响应增量。

一个单独验证的耦合例子是 \(u=(0,v(X_1),0)\)。线性纵向内力为零，但 StVK 几何纵向项为

\[
(\operatorname{Div}P_{\rm geo})_1=(\lambda+2\mu)v_{,1}v_{,11}.
\]

因此仅训练横向线性剪切方程会漏掉实际的纵向非线性驱动。测试直接检查了这个非零项。

## 实际进入训练的残差

`graphwave/nonlinear_constraints.py` 独立生成三次以内的制造场，包含纵向、两个横向、扭转、混合场和混合剪切六族。梯度幅值为 \(10^{-6}\)–\(10^{-4}\)，单元边长为 1.2–2.5 mm。时间函数

\[
u(X,t)=s(X)[1-\cos(\omega t)]
\]

给出解析速度、加速度和严格零初始位移、速度；频率取 0.18–0.28 MHz。独立解析本构与链式求导给出体力 \(b=\rho_0\ddot u-\operatorname{Div}P\) 和六面牵引 \(t=PN\)，不调用学习得到的空间导数构造标签。

每个场在 125 个 GLL 节点和六个面上形成以下约束：

| 约束组 | 实际检查量 | 每个场的验收分量数 |
| --- | --- | ---: |
| 内部运动平衡 | \(\rho_0\ddot u+(f_{NN}-f_{\partial\Omega})/V-b\) 在内部节点的 RMS | 5 分支 × 3 方向 |
| 边界运动平衡 | 相同残差在边界节点的 RMS | 5 × 3 |
| 边界牵引 | 学习应力的 \(PN\) 与解析牵引差 | 5 × 3 |
| 应变能 | 学习总势能与制造场本构能量的积分差 | 5 |
| 内功率 | \(\sum f_{NN}\cdot v-\int P:\nabla v\,dV\) | 5 |

数值积分使用声明的 GLL 规则；制造场检查包含其离散误差，不能据此宣布连续域处处精确。96 个训练场对应 5280 个分组、分方向验收约束，另用不同随机种子的 48 个场独立检查。训练目标同时包含真实 COMSOL 模态标签与上述残差：

\[
\mathcal J=\sum_q[w_q\,2\pi(0.005)(f_q^{NN}-f_q^{COMSOL})]^2
+30^2\sum_g\|\widehat r_g\|^2+10^{-4}\|\theta\|^2.
\]

每个物理分支、方向均使用解析标签的固定尺度归一化，并声明 5% 特征量下限，避免接近零的理论分量导致不适定相除。分量之间不能通过平均值互相掩盖。归一化物理残差与位移误差的单位、弱信号阈值不同，二者不混用。

v3 使用软惩罚，边界检查失败；v4 增加每个分量的 4% 不等式约束，但优化器线搜索未正常结束。v5 使用量纲一致的平方不等式 \(1-(R_g/0.035)^2\ge0\)，训练目标按初值固定缩放，仍按独立 5% 物理门槛验收。原始 v5 点略超训练约束，离线参数回退后的 `v5-feasible` 严格通过训练约束，且新的 48 个检查场最大残差为 3.6181%；原优化器失败状态仍保留。每轮配置、源代码快照、随机种子、合成场和完整失败记录均保留在 `training/`。

参数化同时严格保持常数/仿射导数一致性、反中心对称性、刚体运动客观性、转置配对的保守内力、原有质量与材料能量参数。时间推进仍严格实施零初始状态和固支位移/速度。推理阶段不读取 COMSOL 标签，也不加入参考解校正。

这里的完整滤波器投影从 v6 起实施。补充测试发现 v1–v5 仅约束了增量，起始滤波器约 1.52e-8 的仿射误差会被保留，导致有限刚体转动产生假应力。令 \(C=[\mathbf1,\xi]\)、\(B=[\mathbf0,\mathbf1]\)，v6 在训练前执行

\[
D_\star=D+(B-DC)(C^TC)^{-1}C^T,
\]

并保持反中心对称，使 \(D_\star C=B\) 在浮点舍入范围内成立。之后的可学习增量满足 \(\Delta D C=0\)。这属于离线权重约束，推理时不作投影或数值修正。最终候选在三轴、三个转角和三轴平移的 12 个状态中，分别检查 StVK 与完整 Murnaghan：最大假内力 1.96e-10 N、应变范数 4.10e-15，均通过预设绝对门槛。线性小应变本构不被错误要求在有限转角下保持客观性。

## 独立检查与适用范围

`tests/test_nonlinear_constraints.py` 检查五分支解析散度与复步长求导的一致性、能量梯度与 Piola 应力、角动量、刚体转动、横向到纵向的非线性耦合，并确认制造场能拒绝破坏运动平衡的候选滤波器。

制造场物理检查、COMSOL 频率拟合、5 ms 瞬态比较分别报告。COMSOL 网格频率变化超过长记录相位门槛时，即使训练残差下降，也不能把参考解或候选宣布为“误差在 5% 内”。最终仍需覆盖 Hann、Gaussian、矩形窗正弦和扫频 × 1、3、5、10、20 周期的真实长记录，以及线性、几何、材料分支的独立收敛参考。

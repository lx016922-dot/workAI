# 粗网格算子的范围核验：传播距离、板宽、激励形式与缺陷

粗网格算子（`docs/coarse-graph-operator.md`）是在 48 × 2 mm 半宽铝板、30 μs 记录、接收点距激励 6–21 mm 的算例上训练和测试的。学到的是所有单元共用的单元算子，本身不依赖板的尺寸；需要实测的是：误差随传播距离、板宽、激励形式和缺陷怎样变化。这里的每个算例都在同一个场景上算三种结果：

- **细网格参考**：冻结网络（等价于 GLL 谱元），4 阶、h 1 mm，自动时间层（相位误差上界 0.01 rad）；
- **粗网格 + 学到的算子**：两个随软件发布的模型，`p4-h3-harmonic`（快速，h 3 mm）和 `p4-h2-multisource`（精确，h 2 mm）；
- **同一粗网格 + GLL 算子**：不学习的基线。

粗网格的计算走离线软件快速引擎现在用的同一条通道（`coarsegraph.record` → `graphnet.rollout`）。

```bash
python -m coarsegraph.verify run --output research/coarse-range/results
python -m coarsegraph.verify table --results research/coarse-range/results
python -m coarsegraph.verify reference --output research/coarse-range/results --case distance-f0.4-t1.6 --h-mm 0.8
```

`results/<算例>.json` 是门槛结果和场景，`results/<算例>.npz` 是全部接收点的波形（单精度），可以不重算就重新统计。

## 1. 门槛

与训练时相同（`coarsegraph.evaluate.case_metrics`）：完整响应的波形相对 L2、f 幅值、2f 幅值，以及非线性增量（完整 − 线性）的 2f 幅值，每个接收点都在 5% 以内才算通过；只统计参考值不低于该组最大值 10% 的接收点。

有一处不同。训练用的记录只有 30 μs，波占满整段记录，所以门槛取整段记录。长板远处的接收点在 100 μs 的记录里大部分时间是静止的，整段记录上 Hann 加权的 2f 幅值主要是基频的泄漏。因此这里的门槛取在每组接收点的**到达时间窗**上：细网格参考超过其峰值 2% 的时间段，前后各加两个周期。整段记录的数值也存在结果文件里（`whole_record`）；两种口径下通过的组数见第 3 节。

接收点按到激励的距离分组（每组 4–6 个位置，各有顶面 z 向和中面 x 向两个接收点），门槛按组判定，所以长板算例给出的是门槛随距离的变化。

## 2. 算例

铝（训练材料），五周期 Hann 脉冲，1 MPa，0.05 μs 采样，两端 8 mm 吸收层（端面激励时只在远端）。

| 类别 | 场景 | 参数 | 接收点组 |
|---|---|---|---|
| distance 传播距离 | 200 × 2 mm 半宽板，100 μs，顶面高斯 σ 2 mm | 0.2 / 0.3 / 0.4 MHz × 板厚 1.6 / 2.4 mm | 距激励 10–25、45–60、85–100、125–140 mm |
| width 板宽 | 48 × 8 mm 半宽板，30 μs，顶面高斯 | 0.2 / 0.4 MHz × 1.6 / 2.4 mm | 对称面上，以及距对称面 6 mm 的一排 |
| end 端面激励 | 100 × 2 mm，50 μs，端面整面轴向 | 0.2 / 0.4 MHz × 1.6 / 2.4 mm | 距端面 20–35、60–75 mm |
| two-patch 双贴片 | 100 × 2 mm，60 μs，两个 σ 1.5 mm 贴片反相、延迟 1 μs | 0.25 MHz，2.0 mm | 10–25、45–60 mm |
| full-width 全宽 | 48 × 4 mm 全宽板（没有对称面），30 μs | 0.3 MHz，1.6 mm | 中线 |
| defect 缺陷 | 100 mm 板，60 μs，缺陷在 x = 50 mm | 见第 5 节 | 缺陷前（反射侧）、缺陷后（透射侧） |

<!-- RESULTS -->

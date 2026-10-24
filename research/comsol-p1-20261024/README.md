# COMSOL P1 数据任务（2026-10-24）

按用户上传的 [仿真要求](../../docs/comsol/comsol-data-request.md)、[工况表](../../docs/comsol/cases.csv)、[CSV 表头](../../docs/comsol/header-template.csv) 和 [元数据模板](../../docs/comsol/metadata-template.json) 执行。

目标是三个 P1 工况的 21 次求解：每个工况的 linear、stvk、murnaghan，以及 linear/murnaghan 的 mesh2 和 dt2 对照。P2、P3 和 amp2 未列入本批。

用户已明确允许忽略附件提到的外部 `coarsegraph.comsol` 校验器和谱元标签；本目录只交付真实 COMSOL 结果及其独立格式、物理限值和收敛检查。不进行后续模型训练。

## 当前状态

以 [campaign-status.json](campaign-status.json) 和 [data/validation.json](data/validation.json) 为准。`complete=false` 表示 21 次运行尚未交齐，不能当作完整数据集。

已通过 COMSOL 6.4 build 293 的单个四阶六面体均匀变形自检。三组第二 Piola 应力的最大绝对误差约为 7e-12 MPa，详见 [本构自检](data/material-check.json) 和 [原始证据](evidence/material-check)。

初次完整 A 工况发现变时间步下的默认 Jacobian 复用导致反复迭代；已准备每个时间步更新 Jacobian 的设置。未完成或失败的试算不会进入交付 CSV。

## 实现与结果

- 半宽三维模型，y=2 mm 仅约束 uy；所有接收点固定在参考构形。
- 总 Lagrange 弱式，严格使用附件定义的能量，P=F·S；linear、StVK、Murnaghan 分别控制几何和材料非线性。
- 顶面完整高斯、端面均匀牵引、双高斯反相及 1 μs 延迟；载荷按参考面积、固定方向施加。
- 吸收层为三个分量上的质量比例体力，alpha_max=22075615.536638446 /s；无附加数值或刚度阻尼。
- 四阶扫掠六面体，h≤0.8 mm；mesh2 的目标 h=0.56 mm；20 阶积分。
- generalized-alpha，rho_infinity=1，strict 输出；最大步长 0.01 μs，dt2 为 0.005 μs。时间相对容差 1e-6，未缩放绝对容差 1e-18 m，Newton 解变化容差因子 0.001。
- 导出所有 601 个时刻的 13 个接收点三分量位移；科学计数法保留 17 位小数。全部输出时刻执行 COMSOL MaxVolume 应变/J 检查。

`data/` 放置通过校验的交付 CSV、每个工况元数据、独立校验报告。`evidence/` 保留与服务器签名清单核对过的原始输入、输出、求解器日志、实际 COMSOL 完成回执及模型源代码。大体积 `.mph` 保存在云端原始运行目录，不是附件要求的必交文件。

连接使用既有配对客户端和平台 VPN/代理，不复制或上传私钥。每个作业使用唯一身份；POST 返回不确定时只查询同一 ID，不重复提交。

## 复用

完整队列及已经准备好的不可变输入保存在 `plan.json` 与 `inputs/`。云环境中的原配对身份可用、服务器旧试算结束后，可恢复队列并逐次上传结果：

```sh
python research/comsol-p1-20261024/campaign.py --push
```

队列每次先执行签名 GET 确认作业是否已经存在；已存在的作业只查询和收集。网络、求解器或校验错误会保留文件并停止队列，不会盲目重复 POST。云环境及 Windows API/COMSOL 服务必须持续运行；环境关闭后需要重新启动队列。

准备输入：

```sh
python research/comsol-p1-20261024/prepare.py \
  --case-id A-f0.40-T1.6 --laws 0 1 2 --output /workspace/p1/A-base
```

模型必须先通过原有签名 revision 接口安装并核对源代码 SHA-256。实际提交和收集使用仓库已有 `tools/collect-comsol-case.py`，依赖云环境中原配对身份。

交付格式转换和独立检查：

```sh
python research/comsol-p1-20261024/collect_dataset.py \
  --material-check /workspace/p1/material-check \
  --runs /workspace/p1/A-base --output research/comsol-p1-20261024/data
python research/comsol-p1-20261024/validate.py research/comsol-p1-20261024/data --partial
```

完整交付时删除 `--partial`，检查必须包含全部 21 个结果。`validation.json` 中的谐波收敛指标使用相同的 0–30 μs 矩形时间窗，只用于 COMSOL 对照诊断，不冒充原谱元校验结果。

## 有限的服务器停止辅助

现有远程接口没有取消作业入口。[Stop-OwnPilot.ps1](Stop-OwnPilot.ps1) 仅用于在 Windows 服务器上停止本次已识别的低效试算：它先核对作业 ID、worker PID、命令行和进程创建时间，再停止该 worker 的进程树，并将该作业标为失败。它保留原文件和 HTTP 服务；可以先加 `-WhatIf` 查看目标。

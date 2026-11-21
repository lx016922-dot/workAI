# 开发者指南

给第一次读这份代码的人。目标是让你在一两天内知道：一次计算从哪里进、从哪里出；哪些文件不能动；改完怎么确认没改坏；常见的几类改动各自要碰哪些地方。

模块清单见 [README 的模块地图](../README.md#模块地图)，方法和核验见那里的文档索引。这里只讲代码怎么组织。

## 1. 一次计算走过的路

桌面软件是一个本机网页服务加一个窗口。界面（`offline/web/`）把模型拼成一个 JSON 请求，交给服务（`offline/app.py`），服务起一个子进程算（`offline/worker.py`）。

| 步骤 | 做什么 | 代码 |
|---|---|---|
| 1. 请求 | `{'scene': …, 'mesh': 'auto', 'controls': …, 'record': …, 'time_stepping': …}` | `offline/web/app.js` 的 `requestBody` |
| 2. 校验 | 场景的每个字段、范围和相互关系；返回规范化的场景 | `offline/domain.py` `validate_request` → `physics/scene.py` `validate_scene` |
| 3. 提交前预览 | 网格大小、时间层、耗时、内存、磁盘、各种提示 | `offline/preview.py`、`offline/resources.py` |
| 4. 网格 | 板、管、孔、缺口的 GLL 六面体网格；有缺陷时网格线穿过缺陷边缘 | `spectral3d/mesh.py`（冻结）、`spectral3d/feature_mesh.py` |
| 5. 载荷和边界 | 每个激励的节点力、约束的位移分量、吸收层和材料阻尼的系数 | `spectral3d/loads.py`、`physics/loads.py`、`physics/boundaries.py`、`graphnet/inputs.py` |
| 6. 时间层 | 量出网格的最高固有频率，选最省的时间层 | `graphnet/timestep.py` |
| 7. 推进 | 逐层推进三组控制（线性、StVK、完整），分块写记录，可断点续算 | `graphnet/rollout.py` 调 `graphnet/recurrence.py`（每组控制一个线程）或 `graphnet/parallel.py`（每组控制铺满所有核） |
| 8. 单元消息 | 每一层里由位移算内力：冻结的图网络 | `graphwave/inference.py`、`graphwave/buffered_inference.py`（冻结） |
| 9. 结果 | 五种响应、显示用的网格、诊断量 | `offline/worker.py`、`offline/recording.py`、`spectral3d/result.py`、`spectral3d/display.py` |
| 10. 分析 | 读已保存的记录：波形、频谱、沿程累积、混频、f–k 谱、理论频散、收敛检查、参数扫描汇总 | `offline/nonlinear.py`、`offline/guided.py`、`offline/scan.py` |

几个贯穿全程的概念：

- **场景（scene）**：模型的全部物理定义。版本 1 是最初的单激励格式，由 `solid/schema.py`（冻结）校验；版本 2（`physics/scene.py`）在它外面加了信号、空间分布、多激励、对称面、吸收层、缺陷、阻尼。新字段都加在版本 2：在 `validate_scene` 里先把它从场景里取出来，再交给版本 1 的校验。
- **控制（control）**：同一个模型算三遍——`linear`、`stvk`（只有几何非线性）、`murnaghan`（完整）。界面上的五种“响应”是它们和它们的差（`offline/domain.py` 的 `RECIPES`）。非线性增量 = 完整 − 线性。
- **记录（record）**：`record/recording.json` 是清单，`record/fields/*.npz` 是分块的数据，`record/states/*.npz` 是断点。完整记录存每个节点每个采样点；精简记录只存接收点的完整波形、抽稀的显示帧和 4 个完整快照。
- **指纹（fingerprint）**：清单里对模型、设置和 `graphnet/*.py` 源码哈希的摘要。续算时指纹不符就拒绝。所以**改了 `graphnet/` 下的任何文件，旧的未完成计算都不能续算**——这是有意的。

## 2. 不能动的东西

**按哈希锁定的文件。** 冻结网络的运行时文件在 `offline/model/model.json` 和 `extension.json` 里记了 SHA-256，启动时核对，不符就拒绝运行：

- `graphwave/__init__.py`、`inference.py`、`predict.py`、`long_rollout.py`、`long_inference.py`、`buffered_inference.py`、`signals.py`
- `spectral3d/__init__.py`、`mesh.py`、`curved_mesh.py`
- `solid/__init__.py`、`schema.py`

要改它们的行为，在外面包一层（`graphnet/`、`physics/`、`spectral3d/feature_mesh.py` 都是这么来的），不要改文件本身，连空格和注释也不要改。

**逐位一致。** 下面这些性质有测试守着，改内核时必须保持：

- 没用到新功能时，结果与改动前逐位相同（例如没有阻尼时，加了阻尼的内核给出同样的位）。
- 每组控制一个线程（`recurrence.three_segments`）和铺满所有核（`parallel.parallel_segments`）给出同样的位，与线程数无关。两处的算式要按同样的顺序写：同一个量在两边的加法、乘法次序必须一样，求和要按节点、分量的同一顺序。
- 中断后续算与一次算完给出同样的位。所以续算需要的量都要在断点里（`rollout.py` 的 `STATE`）。

**诊断不能被绕开。** 能量收支（动能 + 应变能 + 耗散 = 外力功，误差 1% 以内）、最大应变 ≤ 10⁻³、J > 0。加新的力时，要把它做的功或耗散的能量记进去，让这个检查仍然有意义。

**求解器不能悄悄忽略它不支持的东西。** 仓库里有几个求解器（冻结网络的长记录、短记录，谱元参考解，二维平面应变，四面体）。新的场景字段如果某个求解器不支持，要让它报错并说明，不能当作没看见（例子：`physics/damping.py` 的 `only_mass`）。

## 3. 怎么确认没改坏

```sh
python tools/run-unit-tests.py --group core       # 场景、信号、频散、板理论、数据契约；几分钟
python tools/run-unit-tests.py --group solvers    # 参考求解器；需要 gmsh
python tools/run-unit-tests.py --group networks   # 冻结网络的输入层、记录、离线软件；最慢，半小时以上
npm test                                          # 界面的纯函数（tests/*.test.mjs）
python -m offline.smoke                           # 起服务、提交一个小算例、读回结果
```

- 每个测试模块在自己的进程里跑。新加的测试文件要登记到 `tools/run-unit-tests.py` 的 `GROUPS` 里，脚本会检查有没有漏的。
- CI（`.github/workflows/tests.yml`）在每次推送和每个合并请求上跑这四组加界面测试。
- `desktop.yml` 在 Windows 上打包并真的开窗口跑一遍；合并到 `main` 的提交信息里带 `[publish-release]` 时发布。版本号在 `offline/version.json`、`desktop/WorkAI.csproj`、`desktop/README-DESKTOP.txt`、`desktop/RELEASE-NOTES.md`、`README.md` 五处。
- 界面里与浏览器有关的部分（画图、下载）没有单元测试，改完要在浏览器里实际点一遍：`python -m offline.app --data <目录> --no-browser`，然后打开它打印的地址。

**物理上对不对，单元测试答不了。** 新的物理功能要有一份核验：与解析解、文献公式或独立求解器对照，写成 `research/<主题>/README.md`，脚本和结果文件放在旁边，能重跑。先写参考解并自检，再跑软件，最后比。已有的例子：`research/material-damping`（阻尼）、`research/transducers`（换能器）、`research/literature-verification`（波速、频散、二次谐波）。

## 4. 常见改动怎么做

### 加一种激励的空间分布

1. `physics/loads.py`：在 `PROFILES` 里加名字，在 `validate_profile` 里校验参数，在 `region_weights` 里给出权重。
2. 如果它不是“一个形状乘一个信号”（例如随位置变化的延迟或方向），在 `expand` 里把它写成几个普通激励之和。斜楔和压电片就是这样做的。
3. 界面：`offline/web/scene-model.js`（标签、默认值）、`offline/web/inspector.js` 的 `profileForm`（表单）。
4. 测试：`physics/test_transducers.py` 这一类；界面在 `tests/offline-workbench.test.mjs`。

### 加一个场景字段

1. `physics/scene.py` 的 `validate_scene`：取出、校验、放回。
2. 用到它的地方读它；**不用它的求解器要拒绝它**。
3. 界面：表单、模型树的摘要、参数扫描里能不能扫（`offline/web/scan.js` 的 `scanParameters`，后端 `offline/scan.py` 的 `ROOTS`）。
4. 提交前预览里说明它的效果（`offline/preview.py`）。

### 加一项结果分析

1. `offline/nonlinear.py` 里加一个函数 `名字(job, body)`：读记录（`series`）、算、返回 JSON。分析窗、点列的选取有现成的函数（`line_of`、`_windows`、`wavenumber`、`linear_fit`）。
2. `offline/app.py` 的路由里加上这个名字。
3. `offline/web/nonlinear.js`：请求体（`…Body`）、结果的 HTML（`…Markup`）、CSV，再在 `mountNonlinear` 里接上按钮。前三个是纯函数，写测试。
4. 文档：`docs/nonlinear-analysis.md`。

### 改时间推进

三处必须一起改，并保持第 2 节的逐位一致：`graphnet/recurrence.py`（`_segment`）、`graphnet/parallel.py`（`parallel_segments`）、`graphnet/timestep.py`（稳定极限和误差判据）。新的状态量加进 `rollout.py` 的断点。材料阻尼的提交是一个完整的例子。

### 加一种材料

自定义材料不用改代码：界面里写应变能密度 W(E)（`docs/materials.md`）。求解器取 W 在零应变处的二阶、三阶展开。

## 5. 代码风格

核心模块（`physics/`、`graphnet/`、`offline/`、`spectral3d/` 里没有锁定的文件、`spectral2d/`、`dispersion/`）用 `ruff format` 统一格式，设置在仓库根目录的 `ruff.toml`：

```sh
pip install ruff
ruff format            # 按 ruff.toml 里列出的目录
ruff format --check    # 只检查
```

- 一行一条语句，行宽 140。
- 注释和文档字符串写“为什么”和“这个量是什么”，不重复代码。
- 给用户看的报错用中文，说明原因和怎么办；代码里的注释用英文。
- 统一格式的那次提交只改排版（每个文件的语法树前后相同），记在 `.git-blame-ignore-revs` 里：`git config blame.ignoreRevsFile .git-blame-ignore-revs` 之后，`git blame` 会跳过它。
- 锁定的文件、`graphwave/`、`solid/` 和 `research/` 不参与格式化。它们保持原样：前者不能动，后者是历史记录。
- numba 编译的函数里只用标量循环和简单的数组运算；不要在调用的参数里写条件表达式（numba 不支持）。

## 6. 目录之外的约定

- **数据目录**：软件的计算记录在 `--data` 指定的目录下的 `jobs/` 和 `scans/`，不在仓库里。`*.npz` 不进版本库。
- **研究目录**（`research/`）：每个主题一个文件夹，`README.md` 是报告，`scripts/` 能重跑，`results/` 是脚本写出的 JSON，`figures/` 是图。大的计算记录放在环境变量指定的目录里，不进仓库。
- **文档**：`docs/` 是使用说明和方法，`research/` 是证据。结论要能从结果文件里查到数字。

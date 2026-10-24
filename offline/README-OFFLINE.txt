WorkAI Offline 0.2.0-recovery.1 / 恢复版

Windows 10/11 x64：
1. 把整个 ZIP 解压到可写目录，例如 D:\WorkAI。不要在压缩包内直接运行。
2. 双击 Start-WorkAI.bat。无需安装 Python、Node、COMSOL 或登录账号。
3. 浏览器会打开本机页面；若没有打开，请复制启动窗口中的 http://127.0.0.1:端口 地址。
4. 保留启动窗口；按 Ctrl+C 正常退出并停止进行中的任务。
   不要用任务管理器强制结束或直接关闭计算窗口。

Linux x86_64（glibc）：
在解压目录运行 sh start-workai.sh。需要本机图形浏览器。
系统无需预装 Python。Alpine/musl、macOS、ARM 不属于本包支持平台。

整个启动、预测及结果分析过程均使用本机文件，无需网络。
Windows 与 Linux 便携运行时均已完成真实网络预测及五种响应检查。
Windows 在 GitHub Actions 的原生 x64 runner 上实测；平台记录见 offline/model/software-validation.json。
推荐 8 GB 以上内存、2 GB 以上空闲磁盘；计算任务和缓存会继续占用空间。
首次计算会编译 Numba 网络。小板通常较快，孔结构可能耗时数十分钟。
任务上限 90 分钟。避免在同一个数据目录同时启动多个实例。

功能：
- 输入板、圆管、通孔与缺口的几何、牵引、频率、幅值及边界。
- 使用冻结图网络逐时预测，不执行预测后数值校正或远程求解。
- 查看完整、线性、StVK、几何增量、材料增量五种响应。
- 三维动画、任意实体内点位与有符号方向、时间窗、FFT、准确 f/2f 复数系数。
- 多点及沿程谐波分析、波形 CSV、配置 JSON、完整结果 ZIP。
- 内置实际计算的铝板示例，可先选计算记录查看。

模型与证据：
当前云环境只恢复到早期快照。图网络源码从原聊天的修改记录恢复，
按原种子 2026102401、材料和训练过程重新生成了 120 参数网络。
新权重 SHA-256 与原权重不同，明确不继承原版的 8 例验收结论。
本次检查记录在 offline/model/validation.json；模型来源在 model.json。
四类历史参考结构的对照不等于新的独立八例测试或任意 CAD 精度保证。
材料固定为文献多晶铝参数；不是实验校准的未知合金模型。
该网络仍依赖网格与逐时传播，未声称比数值求解更快。

所有用户任务在 data/jobs；运行缓存在 data/cache。
可用 --data 路径 或 WORKAI_DATA_DIR 环境变量指定另外的数据目录。
移动应用前退出程序；保留整个 data 文件夹即可保留记录。
日志在各任务目录 worker.log。ZIP 导出包含数值场与配置，不含运行环境。
启动失败时先完整重新解压，并确认目录可写、路径不是压缩包内部。
仅绑定本机回环地址，不提供公网服务，不需要云端配对身份。

校验文件：
Windows: runtime\python.exe verify-package.py
Linux: runtime/bin/python3 verify-package.py
SHA256SUMS.txt 用于校验下载的 ZIP，MANIFEST.json 列出每个交付文件。

第三方许可随 Python runtime 和各依赖的 dist-info 目录提供。
源码和构建说明：https://github.com/lx016922-dot/workAI

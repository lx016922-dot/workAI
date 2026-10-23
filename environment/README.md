# 环境备份（2026-10-23）

本次按用户选择，将源码、配置和小型验收文件上传至公开仓库 `lx016922-dot/workAI`。这是一份项目恢复材料，不是整台云环境的镜像。

## 内容与来源

- 仓库根目录：从环境中最后一版已校验的 `workai-website-20261023-curved-training-final.zip` 恢复的 207 个源文件，包含网站、求解与训练代码、测试、依赖声明和技术文档。原 Git 工作目录只有 `archives` 占位文件。
- `workai-remote/`：云端 PowerShell 客户端、启动脚本、固定校验值的原协议实现、恢复的研究源文件，以及小型验收摘要和测试记录。
- `attachments/`：启动脚本依赖的两份原始小型输入 ZIP，分别是体积协议补丁和研究源码。已检查内部文件并核对原有 SHA-256；不含配对私钥。
- `../.env.example` 和 `workai-remote/connection/server.example.json`：不含真实密码或密钥的配置示例。连接示例不能直接用来连接现有服务。
- `runtime-versions.json`：本次环境中的操作系统、Node、Python 和主要科学计算依赖版本，不包含环境变量或凭据。
- `snapshot-manifest.json`：本次纳入文件的大小与 SHA-256、来源及排除范围。

源码归档的 SHA-256 为 `2bf8bfe4ca5e27e08fd2f7616e2c3fc7f1b68527a4c8c2eda5855bcfdc1202a1`。逐文件原始记录保留于 `workai-remote/validation/20261023-curved-training/source-bundle-final-manifest.json`。根目录 README 新增了本说明的链接，`.gitignore` 增加了凭据与生成文件排除规则；其余归档源文件保持原始字节。

验收目录记录的是原计算和训练时的结果，其中仍有未通过的科学精度检查。上传备份不会改变这些结论，也不代表重新完成了远程计算、训练或全部浏览器验收。记录引用的原始场数据、权重和大型报告不在本次上传范围内。

## 未上传的内容

客户端私钥、真实连接报告与身份文件、VPN 或 GitHub 凭据、平台内部目录、虚拟环境、已安装的 PowerShell、下载缓存、大型有限元场文件、训练数据集、模型权重和大型 HTML/视频报告均保留在原云环境，没有加入 Git。原环境的私钥及其他保留文件未被修改或删除。

公开源文件中保留了历史任务编号、非秘密的校验值和原目录说明；它们不能替代运行数据。完整恢复现有配对关系还需要用户另行安全保管的原身份和连接文件。不要把这些文件提交到公开或私有 Git 仓库，也不要因文件缺失自动生成新身份。

## 在原环境中恢复和验证

仓库位于 `/workspace/workAI`，原有辅助目录为 `/workspace/workai-remote`。本次恢复源码后可运行现有的本地测试：

```sh
cd /workspace/workAI
npm ci --ignore-scripts --offline --no-audit --no-fund --cache /tmp/workai-npm-cache
npm test
npm run test:protocol
```

`npm test` 使用模拟计算端；协议测试调用本地原版校验器。这些命令不会提交远程 COMSOL 计算。新的本地验证结果保存在 `verification/`。

## 在新的环境中准备

准备 Linux、Node.js 22 或以上和 Python 3.12，并将仓库放在 `/workspace/workAI`。脚本保留了原有 `/workspace` 路径约定。

先将公开辅助文件和小型输入放回预期位置；`cp -an` 避免覆盖目的地已有文件：

```sh
cd /workspace/workAI
mkdir -p /workspace/workai-remote /workspace/attachments
cp -an environment/workai-remote/. /workspace/workai-remote/
cp -an environment/attachments/. /workspace/attachments/
chmod +x /workspace/workai-remote/bin/workai /workspace/workai-remote/bin/workai-pwsh
```

通过安全渠道恢复原有 `private/client-private.xml`、`connection/server-original.json`、`connection/server-paired-20261022.json`、`connection/client-identity.json` 和 `pairing/cloud-client-public.xml`，私钥目录权限保持 0700、私钥文件保持 0600。原客户端和启动脚本会核对固定的可信校验值；不要用配置示例替换原报告，也不要关闭验证来跳过缺失文件。

恢复平台 VPN 和网络配置后，运行已有准备脚本：

```sh
bash /workspace/workai-remote/prepare-website.sh
cd /workspace/workAI
npm start
```

准备脚本会检查保留输入、下载并校验 PowerShell、按依赖声明准备 Python 环境并运行本地检查；它不会自动重跑历史训练或 COMSOL 任务。完整浏览历史计算和训练结果，需要另行恢复本次排除的数据目录。网站启动默认仅监听回环地址；外部部署所需的 HTTPS 和密码通过安全运行配置提供。

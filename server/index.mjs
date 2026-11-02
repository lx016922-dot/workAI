import { readFile, access, lstat } from 'node:fs/promises';
import { constants } from 'node:fs';
import { join, resolve } from 'node:path';
import { createApp } from './app.mjs';
import { RemoteClient } from './remote.mjs';
import { JobStore } from './store.mjs';
import { ModelStore } from './models.mjs';
import { readTrainingStatus } from './training.mjs';
import { SolidService } from './solid.mjs';
import { SpectralService } from './spectral.mjs';
import { PlaneService } from './plane.mjs';

const root = resolve(process.env.WORKAI_REMOTE_ROOT || '/workspace/workai-remote');
const state = resolve(process.env.WORKAI_STATE_DIR || join(root, 'web/jobs'));
const host = process.env.WORKAI_HOST || '127.0.0.1';
const port = Number(process.env.PORT || 3000);
const password = process.env.WORKAI_WEB_PASSWORD || '';
const origin = (process.env.WORKAI_WEB_ORIGIN || '').replace(/\/$/, '');
try {
  if (!Number.isInteger(port) || port < 1 || port > 65535) throw new Error('PORT 必须是有效端口。');
  const local = ['127.0.0.1', 'localhost'].includes(host);
  if ((!local || origin) && (password.length < 16 || !/^https:\/\/[^/]+$/.test(origin))) {
    throw new Error('外部访问需要 HTTPS 网站地址 WORKAI_WEB_ORIGIN 和至少 16 字符的 WORKAI_WEB_PASSWORD。');
  }
  if (origin) {
    const parsed = new URL(origin);
    if (parsed.username || parsed.password || parsed.search || parsed.hash) throw new Error('WORKAI_WEB_ORIGIN 必须是不带凭据和路径的 HTTPS 源。');
  }
  await access(join(root, 'bin/workai'), constants.X_OK);
  await access(join(root, 'bin/workai-pwsh'), constants.X_OK);
  await access(join(root, 'Get-WorkAIArtifacts.ps1'));
  const key = await lstat(join(root, 'private/client-private.xml'));
  const keyDirectory = await lstat(join(root, 'private'));
  if (!key.isFile() || key.isSymbolicLink() || !keyDirectory.isDirectory() || keyDirectory.isSymbolicLink() ||
      (key.mode & 0o777) !== 0o600 || (keyDirectory.mode & 0o777) !== 0o700) throw new Error('请保持已配对私钥的 0600 / 0700 权限。');
  const info = JSON.parse((await readFile(join(root, 'connection/server-paired-20261022.json'), 'utf8')).replace(/^\uFEFF/, ''));
  const store = new JobStore(state, join(root, 'requests')); await store.init();
  const remote = new RemoteClient(root);
  const models = new ModelStore(join(state, '..', 'models'), remote); await models.init();
  const solid = new SolidService(root); await solid.init();
  const spectral = new SpectralService(root,solid); await spectral.init(); solid.spectral=spectral;
  const plane = new PlaneService(root,solid); await plane.init(); solid.plane=plane;
  const server = createApp({ remote, store, models, solid, spectral, plane, trainingReader: () => readTrainingStatus(root), password, origin, connection: {
    base_url: info.base_url, server_fingerprint: info.server_public_key_sha256,
    client_fingerprint: info.cloud_client_public_key_sha256,
  } });
  server.on('error', error => { console.error(`WorkAI 启动失败：${error.code || '服务错误'}`); process.exitCode = 1; });
  server.listen(port, host, () => console.log(`WorkAI 网站服务已启动（端口 ${port}，${host}）。`));
  const stop = () => server.close(() => process.exit(0));
  process.once('SIGTERM', stop); process.once('SIGINT', stop);
} catch (error) { console.error(`WorkAI 启动失败：${error.message}`); process.exitCode = 1; }

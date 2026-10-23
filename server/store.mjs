import { mkdir, readdir, readFile, writeFile, rename, lstat } from 'node:fs/promises';
import { join } from 'node:path';
import { randomUUID } from 'node:crypto';
import { JOB_ID, assertJobId, AppError } from './remote.mjs';

export class JobStore {
  constructor(directory, requestsDirectory) { this.directory = directory; this.requestsDirectory = requestsDirectory; }
  async init() {
    await mkdir(this.directory, { recursive: true, mode: 0o700 });
    let entries;
    try { entries = await readdir(this.requestsDirectory, { withFileTypes: true }); }
    catch (error) { if (error.code === 'ENOENT') return; throw error; }
    for (const entry of entries) {
      if (!entry.isDirectory() || !JOB_ID.test(entry.name) || await this.get(entry.name)) continue;
      const file = join(this.requestsDirectory, entry.name, 'submission.json');
      try {
        const info = await lstat(file);
        if (!info.isFile() || info.isSymbolicLink() || info.size > 65_536) continue;
        const request = JSON.parse(await readFile(file, 'utf8'));
        if (request.job_id !== entry.name) continue;
        await this.put({ job_id: entry.name, kind: request.kind, created_at: request.prepared_at_utc, origin: 'recovered', attempted: true });
      } catch (error) { if (error.code !== 'ENOENT') throw new AppError(500, '保留的任务记录无法读取，请检查工作目录。'); }
    }
  }
  async get(id) {
    assertJobId(id);
    try { return JSON.parse(await readFile(join(this.directory, `${id}.json`), 'utf8')); }
    catch (error) { if (error.code === 'ENOENT') return null; throw error; }
  }
  async put(record) {
    assertJobId(record.job_id);
    const target = join(this.directory, `${record.job_id}.json`);
    const temporary = `${target}.${randomUUID()}.tmp`;
    await writeFile(temporary, JSON.stringify(record, null, 2) + '\n', { mode: 0o600, flag: 'wx' });
    await rename(temporary, target);
    return record;
  }
  async update(id, fields) {
    const previous = await this.get(id);
    return this.put({ ...previous, ...fields, job_id: id });
  }
  async list() {
    const names = (await readdir(this.directory)).filter(name => JOB_ID.test(name.replace(/\.json$/, '')) && name.endsWith('.json'));
    const records = await Promise.all(names.map(name => this.get(name.slice(0, -5))));
    return records.sort((a, b) => String(b.created_at || '').localeCompare(String(a.created_at || '')));
  }
}

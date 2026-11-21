"""The Windows package copies source files by an explicit list (tools/build-offline.py). A module of the package that
imports a first-party file outside that list works from the repository and fails in the package: this test finds it."""
import ast
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FIRST_PARTY = {p.name for p in REPO.iterdir() if p.is_dir() and (p / '__init__.py').exists()}
# Imports reached only from code the application never runs would be named here, each with its reason: (importing file, imported file).
ALLOWED = set()


def packaged():
    """Source files of the package, relative to the repository: the lists of the build script and the whole of offline/."""
    tree = ast.parse((REPO / 'tools' / 'build-offline.py').read_text(encoding='utf-8'))
    files = next(ast.literal_eval(node.value) for node in ast.walk(tree) if isinstance(node, ast.Assign) and any(getattr(t, 'id', '') == 'files' for t in node.targets))
    out = {f'{module}/{name}' for module, names in files.items() for name in names}
    return out | {f'offline/{p.name}' for p in (REPO / 'offline').glob('*.py') if not p.name.startswith('test_')}


def imported(path):
    """First-party files a module imports: {repository-relative path}."""
    package = path.split('/')[0]; found = set()
    for node in ast.walk(ast.parse((REPO / path).read_text(encoding='utf-8'))):
        if isinstance(node, ast.ImportFrom):
            base = ([package] if node.level else []) + (node.module.split('.') if node.module else [])
            if not base or base[0] not in FIRST_PARTY: continue
            target = '/'.join(base)
            if (REPO / (target + '.py')).exists(): found.add(target + '.py')
            else:
                found.add(target + '/__init__.py')
                found.update(f'{target}/{alias.name}.py' for alias in node.names if (REPO / target / (alias.name + '.py')).exists())
        elif isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split('.')
                if parts[0] in FIRST_PARTY and (REPO / ('/'.join(parts) + '.py')).exists(): found.add('/'.join(parts) + '.py')
    return found


class Package(unittest.TestCase):
    def test_every_first_party_import_of_the_package_is_in_the_package(self):
        files = packaged(); self.assertIn('physics/damping.py', files); self.assertTrue(all((REPO / f).exists() for f in files), sorted(f for f in files if not (REPO / f).exists()))
        missing = sorted((source, target) for source in sorted(files) for target in imported(source) if target not in files and (source, target) not in ALLOWED and not target.split('/')[-1].startswith('test_'))
        self.assertEqual(missing, [], '打包清单（tools/build-offline.py）里缺这些被引用的文件：' + '；'.join(f'{a} → {b}' for a, b in missing))


if __name__ == '__main__': unittest.main()

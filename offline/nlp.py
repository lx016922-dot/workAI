"""Natural-language model editing, offline: Chinese or English sentences → operations on the scene.

    在 100 mm 处添加裂纹，深 0.5 mm；在 30、60 mm 处各加一个直径 1 mm 的通孔
    在 60 到 80 mm 之间添加分层，距上表面 0.8 mm；删除所有孔；板长改为 200 mm；频率 300 kHz
    add a closed crack at x = 40 mm, depth 0.4 mm; remove all defects; set thickness to 2 mm

A rule-based reader of this software's own vocabulary (defects, plate size, excitation, record length), not a
general language model: it runs without a network, returns the operations it understood with the values it
used (and which ones were defaults), lists any text it could not use, and never applies an operation the
scene validator rejects. The interface shows the result for confirmation before it changes the model.
Coordinates: x along the plate from the xmin end, y across the width, z up from the bottom face.
"""
import copy
import re
import unicodedata

from solid.schema import require

NUM = r'(-?\d+(?:\.\d+)?)'
UNIT = r'\s*(mm|毫米|cm|厘米|μm|um|微米|m(?![a-zA-Z])|米)?'
SCALE = {'mm': 1., '毫米': 1., 'cm': 10., '厘米': 10., 'μm': .001, 'um': .001, '微米': .001, 'm': 1000., '米': 1000.}
LINK = r'\s*(?:改为|改成|设为|设置为|变为|调整为|为|是|=|:|：|约|大约|of|is|to)?\s*'
KINDS = [
    ('contact', r'接触界面|接触面|接触|contact(?:\s+interface)?|kissing\s*bond'),
    ('delamination', r'分层|脱层|脱粘|层间开裂|delamination|disbond'),
    ('crack', r'裂纹|裂缝|裂口|cracks?'),
    ('notch', r'缺口|凹槽|刻槽|沟槽|槽|notch(?:es)?|groove|slot'),
    ('hole', r'通孔|圆孔|钻孔|横孔|侧钻孔|孔洞|孔|holes?|\bsdh\b|side[- ]drilled'),
]
LABEL = {'crack': '裂纹', 'delamination': '分层', 'contact': '接触界面', 'notch': '缺口', 'hole': '孔'}
REMOVE = r'删除|删掉|去掉|去除|移除|清除|清空|remove|delete|clear'
MOVE = r'移到|移动到|挪到|改到|移至|\bmove\b'
ADD = r'添加|增加|加入|加上|加个|加\d|加一|放置|插入|新建|\badd\b|insert|place|\bput\b|create'
GEOMETRY = [('length_mm', r'板长|长度|length'), ('thickness_mm', r'板厚|厚度|thickness'), ('width_mm', r'板宽|宽度|width')]
FROM_TOP = r'距(?:离)?(?:上表面|顶面|表面)|below\s+the\s+top(?:\s+surface)?|from\s+the\s+top(?:\s+surface)?'
FROM_BOTTOM = r'距(?:离)?(?:下表面|底面|背面)|above\s+the\s+bottom(?:\s+surface)?|from\s+the\s+bottom(?:\s+surface)?'
PARAMS = [r'深度|深|高度|高|depth|deep|height|tall', r'宽度|宽|width|wide', r'直径|diameter|φ|dia\.?', r'半径|radius', r'长度|长|length|long', FROM_TOP, FROM_BOTTOM, r'y\s*=', r'z\s*=']


def normalize(text):
    text = unicodedata.normalize('NFKC', text)
    for a, b in (('～', '~'), ('－', '-'), ('—', '-'), ('–', '-'), ('＝', '='), ('Φ', 'φ'), ('ø', 'φ'), ('Ø', 'φ')):
        text = text.replace(a, b)
    digits = {'一': '1', '两': '2', '二': '2', '三': '3', '四': '4', '五': '5', '六': '6'}
    return re.sub(r'([一两二三四五六])\s*(个|条|处|道)', lambda m: digits[m.group(1)] + m.group(2), text)


def _mm(value, unit):
    return float(value) * SCALE.get((unit or 'mm').lower(), SCALE.get(unit, 1.))


def _find(pattern, text):
    return re.search(pattern, text, re.I)


def _length(keys, text, before=True):
    """A length (mm) introduced by one of `keys` (深度 0.5 mm, depth of 0.5 mm) or followed by it (0.5 mm 深)."""
    m = _find(rf'(?:{keys}){LINK}{NUM}{UNIT}', text)
    if m: return _mm(m.group(1), m.group(2))
    if before:
        m = _find(rf'{NUM}{UNIT}\s*(?:的)?\s*(?:{keys})', text)
        if m: return _mm(m.group(1), m.group(2))
    return None


def _without_parameters(text):
    """The clause with its size/depth phrases removed, so their numbers are not read as positions."""
    for keys in PARAMS:
        text = re.sub(rf'(?:{keys}){LINK}{NUM}{UNIT}', ' ', text, flags=re.I)
        text = re.sub(rf'{NUM}{UNIT}\s*(?:的)?\s*(?:{keys})', ' ', text, flags=re.I)
    return text


def _positions(text, L):
    """x positions (mm) or one x range: 在 100 mm 处, x = 40, 30、60 mm 处, 从 60 到 80 mm, 距右端 20 mm, 中间."""
    text = _without_parameters(text)
    m = _find(rf'(?:从|x\s*=?\s*|between\s+|from\s+)?{NUM}{UNIT}\s*(?:到|至|~|-|and|to)\s*{NUM}{UNIT}', text)
    if m and (m.group(2) or m.group(4) or _find(r'从|之间|between|from|x\s*=', text)):
        unit = m.group(4) or m.group(2); a, b = _mm(m.group(1), unit), _mm(m.group(3), unit)
        return {'range': (min(a, b), max(a, b))}
    m = _find(rf'距(?:离)?\s*(?:右端|末端|另一端|xmax)\s*{NUM}{UNIT}', text) or _find(rf'{NUM}{UNIT}\s*from\s+the\s+(?:right|far)\s+end', text)
    if m: return {'x': [L - _mm(m.group(1), m.group(2))]}
    m = _find(rf'{NUM}{UNIT}(?:\s*(?:[、,，&]|和|及|与|and)\s*{NUM}{UNIT})+', text)
    if m and _find(r'处|位置|分别|各|\bat\b|x\s*=|在', text):
        numbers = re.findall(rf'{NUM}{UNIT}', m.group(0)); unit = next((u for _, u in reversed(numbers) if u), None)
        return {'x': [_mm(v, u or unit) for v, u in numbers]}
    for pattern in (rf'x\s*=\s*{NUM}{UNIT}', rf'(?:在|位于|位置|距(?:离)?(?:左端|起点|激励端|端面|板端|xmin)?|移到|移动到|挪到|移至|改到|\bat\b|\bto\b)\s*(?:x\s*=\s*)?{NUM}{UNIT}', rf'{NUM}{UNIT}\s*(?:处|的位置|位置)'):
        m = _find(pattern, text)
        if m: return {'x': [_mm(m.group(1), m.group(2))]}
    if _find(r'中间|中部|中央|中心|正中|middle|centre|center', text): return {'x': [L / 2]}
    return {}


def _size(scene):
    g = scene['geometry']; return g['length_mm'], g['width_mm'], g['thickness_mm']


def _defects_from_text(kind, text, scene):
    """(defects, defaults used) described by one clause."""
    from physics.defects import default_defect
    L, W, T = _size(scene); used = []; existing = scene.get('defects') or []
    base = default_defect(kind, scene, existing)
    where = _positions(text, L); depth = _length(PARAMS[0], text)
    for x in list(where.get('x') or []) + list(where.get('range') or []):
        require(0 < x < L, f'x = {x:g} mm 不在板上（板长 {L:g} mm）；可以先说“板长改为 {max(2 * L, (int(x / 50) + 1) * 50):g} mm”。')
    from_top = _length(FROM_TOP, text, before=False); from_bottom = _length(FROM_BOTTOM, text, before=False)
    closed = bool(_find(r'闭合|呼吸|紧贴|带接触|有接触|closed|breathing|kissing|with\s+contact', text))
    horizontal = kind == 'delamination' or (kind == 'contact' and _find(r'水平|层间|horizontal|between\s+layers', text))
    if horizontal:
        if 'range' in where: a, b = where['range']
        else:
            length = _length(PARAMS[4], text) or 10.
            if not _length(PARAMS[4], text): used.append(f'长度未指定，取 {length:g} mm')
            centre = (where.get('x') or [None])[0]
            if centre is None: centre = L / 2; used.append(f'位置未指定，取板长中点 {centre:g} mm')
            a, b = centre - length / 2, centre + length / 2
        d = {'type': 'delamination', 'closed': closed} if kind == 'delamination' else {'type': 'contact', 'orientation': 'horizontal'}
        d['x_start_mm'], d['x_end_mm'] = round(a, 4), round(b, 4)
        if from_top is not None: d['depth_mm'] = from_top
        elif from_bottom is not None: d['depth_mm'] = round(T - from_bottom, 4)
        elif depth is not None: d['depth_mm'] = depth
        else:
            d['depth_mm'] = round(T / 2, 4)
            if not _find(r'中面|板厚中间|厚度中间|mid[- ]?(?:plane|thickness)', text): used.append(f'深度未指定，取板厚中面（距上表面 {T / 2:g} mm）')
        return [d], used
    xs = where.get('x') or ([sum(where['range']) / 2] if 'range' in where else None)
    if not xs:
        xs = [base['x_mm']]; used.append(f'位置 x 未指定，取 {base["x_mm"]:g} mm')
    rows = []
    for x in xs:
        d = copy.deepcopy(base); d['x_mm'] = round(x, 4)
        if kind == 'crack':
            d['from'] = 'inside' if _find(r'内部|埋藏|内含|内嵌|inside|embedded|internal', text) else 'bottom' if _find(r'下表面|底面|背面|bottom|back\s*wall', text) else 'top'
            if depth is not None: d['depth_mm'] = depth
            elif not used or '深度' not in used[-1]: used.append(f'深度未指定，取 {d["depth_mm"]:g} mm')
            if d['from'] == 'inside':
                z = _length(r'z\s*=', text, before=False); d['z_mm'] = z if z is not None else round(T / 2, 4)
                if z is None: used.append('内部裂纹中心取板厚中面')
            d['closed'] = closed
        elif kind == 'notch':
            width = _length(PARAMS[1], text)
            if width is not None: d['width_mm'] = width
            else: used.append(f'宽度未指定，取 {d["width_mm"]:g} mm')
            if depth is not None: d['depth_mm'] = depth
            else: used.append(f'深度未指定，取 {d["depth_mm"]:g} mm')
            d['from'] = 'bottom' if _find(r'下表面|底面|背面|bottom', text) else 'top'
        elif kind == 'hole':
            diameter = _length(PARAMS[2], text); radius = _length(PARAMS[3], text)
            if diameter is not None: d['radius_mm'] = round(diameter / 2, 4)
            elif radius is not None: d['radius_mm'] = radius
            else: used.append(f'孔径未指定，取直径 {2 * d["radius_mm"]:g} mm')
            if _find(r'横孔|侧钻|side[- ]?drilled|\bsdh\b|贯穿板宽|沿宽度|axis\s*y', text):
                d['axis'] = 'y'; d.pop('y_mm', None); z = _length(r'z\s*=', text, before=False)
                if from_top is not None: z = T - from_top
                elif from_bottom is not None: z = from_bottom
                d['z_mm'] = round(z if z is not None else T / 2, 4)
                if z is None: used.append('横孔中心取板厚中面')
            else:
                y = _length(r'y\s*=', text, before=False)
                if y is not None: d['y_mm'] = y
        else:
            d = {'type': 'contact', 'orientation': 'vertical', 'x_mm': round(x, 4)}
        rows.append(d)
    return rows, list(dict.fromkeys(used))


def describe(d):
    """One line in Chinese for a defect."""
    from physics.defects import LABELS
    kind = d['type']
    if kind == 'crack':
        parts = [f'x = {d["x_mm"]:g} mm', {'top': '上表面开口', 'bottom': '下表面开口', 'inside': f'内部（中心 z = {d.get("z_mm", 0):g} mm）'}[d.get('from', 'top')],
                 f'{"高" if d.get("from") == "inside" else "深"} {d["depth_mm"]:g} mm'] + (['闭合（接触）'] if d.get('closed') else [])
    elif kind == 'delamination':
        parts = [f'x = {d["x_start_mm"]:g}–{d["x_end_mm"]:g} mm', f'距上表面 {d["depth_mm"]:g} mm'] + (['闭合（接触）'] if d.get('closed') else [])
    elif kind == 'contact':
        parts = [f'竖直截面 x = {d["x_mm"]:g} mm'] if d['orientation'] == 'vertical' else [f'水平 x = {d["x_start_mm"]:g}–{d["x_end_mm"]:g} mm', f'距上表面 {d["depth_mm"]:g} mm']
    elif kind == 'notch':
        parts = [f'x = {d["x_mm"]:g} mm', f'宽 {d["width_mm"]:g} mm', f'深 {d["depth_mm"]:g} mm', '上表面' if d.get('from', 'top') == 'top' else '下表面']
    else:
        axis = d.get('axis', 'z')
        parts = [f'{"通孔" if axis == "z" else "横孔"} 直径 {2 * d["radius_mm"]:g} mm', f'x = {d["x_mm"]:g} mm'] + \
                ([f'y = {d["y_mm"]:g} mm'] if axis == 'z' and 'y_mm' in d else [f'z = {d["z_mm"]:g} mm'] if 'z_mm' in d else [])
    return LABELS[kind] + '：' + '，'.join(parts)


def _validate_defects(scene):
    from physics.defects import validate_defects
    validate_defects(scene.get('defects') or [], scene)


class Plan:
    def __init__(self, scene):
        self.scene = copy.deepcopy(scene); self.operations = []; self.unparsed = []

    def record(self, text, summary, applied=True, defaults=()):
        row = {'text': text.strip(), 'summary': summary, 'applied': applied}
        if defaults: row['defaults'] = list(defaults)
        self.operations.append(row)

    def apply(self, text):
        """True when the clause was understood (applied or refused with a reason)."""
        scene = self.scene; kinds = [k for k, p in KINDS if _find(p, text)]
        if 'contact' in kinds and len(kinds) > 1: kinds.remove('contact')  # "裂纹（带接触）": a closed crack
        if _find(REMOVE, text): return self._remove(text, kinds)
        if kinds and _find(MOVE, text) and not _find(ADD, text): return self._move(text, kinds[0])
        if kinds:
            try:
                require(scene['geometry']['kind'] == 'plate', '缺陷目前只能加在平板上；请先把几何改为平板。')
                rows, defaults = _defects_from_text(kinds[0], text, scene)
                trial = copy.deepcopy(scene); trial['defects'] = list(trial.get('defects') or []) + rows; _validate_defects(trial)
            except ValueError as error:
                self.record(text, f'未执行：{error}', applied=False); return True
            self.scene = trial
            for d in rows: self.record(text, '添加 ' + describe(d), defaults=defaults)
            return True
        return self._settings(text)

    def _remove(self, text, kinds):
        defects = self.scene.get('defects') or []
        if not defects: self.record(text, '模型中没有缺陷可删除', applied=False); return True
        if not kinds: keep = []
        else:
            keep = [d for d in defects if d['type'] not in kinds]
            number = _find(r'第\s*(\d+)|(\d+)\s*号|#\s*(\d+)', text)
            if number and not _find(r'所有|全部|\ball\b', text):
                i = int(next(g for g in number.groups() if g)) - 1; target = [d for d in defects if d['type'] in kinds]
                keep = [d for d in defects if d is not target[i]] if 0 <= i < len(target) else defects
        removed = len(defects) - len(keep); self.scene['defects'] = keep
        self.record(text, f'删除 {removed} 个缺陷' if removed else '没有匹配的缺陷', applied=bool(removed)); return True

    def _move(self, text, kind):
        where = _positions(text, _size(self.scene)[0]); targets = [d for d in self.scene.get('defects') or [] if d['type'] == kind]
        if not targets or not where.get('x'):
            self.record(text, f'没有可移动的{LABEL[kind]}，或没有给出新位置', applied=False); return True
        trial = copy.deepcopy(self.scene); d = [e for e in trial['defects'] if e['type'] == kind][-1]; x = where['x'][0]
        if 'x_mm' in d: d['x_mm'] = x
        else: half = (d['x_end_mm'] - d['x_start_mm']) / 2; d['x_start_mm'], d['x_end_mm'] = x - half, x + half
        try: _validate_defects(trial)
        except ValueError as error: self.record(text, f'未执行：{error}', applied=False); return True
        self.scene = trial; self.record(text, '移动 → ' + describe(d)); return True

    def _settings(self, text):
        scene = self.scene
        for key, pattern in GEOMETRY:
            if _find(pattern, text) and not _find(r'吸收|absorb', text):
                value = _length(pattern, text)
                if value is None: continue
                trial = copy.deepcopy(scene); trial['geometry'][key] = value
                try: _validate_defects(trial)
                except ValueError as error: self.record(text, f'未执行：改变尺寸后 {error}', applied=False); return True
                self.scene = trial; self.record(text, f'{ {"length_mm": "板长", "thickness_mm": "板厚", "width_mm": "板宽"}[key] }改为 {value:g} mm'); return True
        m = _find(rf'(?:中心频率|频率|frequency){LINK}{NUM}\s*(mhz|khz|兆赫|千赫)', text) or _find(rf'{NUM}\s*(mhz|khz)\b', text)
        if m:
            value = float(m.group(1)) * (.001 if m.group(2).lower() in ('khz', '千赫') else 1.)
            scene['source']['frequency_mhz'] = value; self.record(text, f'主激励频率改为 {value:g} MHz'); return True
        m = _find(rf'(?:周期数|周期|cycles?){LINK}{NUM}', text)
        if m: scene['source']['cycles'] = float(m.group(1)); self.record(text, f'主激励周期数改为 {float(m.group(1)):g}'); return True
        m = _find(rf'(?:记录时长|记录|时长|仿真时间|计算时间|duration|record(?:\s+length)?){LINK}{NUM}\s*(μs|us|微秒|ms|毫秒)', text)
        if m:
            value = float(m.group(1)) * (1000. if m.group(2).lower() in ('ms', '毫秒') else 1.)
            scene['acquisition']['duration_us'] = value; self.record(text, f'记录时长改为 {value:g} μs'); return True
        m = _find(rf'(?:幅值|幅度|amplitude){LINK}{NUM}', text)
        if m: scene['source']['amplitude_mpa'] = float(m.group(1)); self.record(text, f'主激励幅值改为 {float(m.group(1)):g} MPa'); return True
        return False


def clauses(text):
    """Sentences, then comma pieces regrouped so that a piece without its own action keeps describing the previous one."""
    starts = rf'{REMOVE}|{MOVE}|{ADD}|' + '|'.join(p for _, p in KINDS) + '|' + '|'.join(p for _, p in GEOMETRY) + r'|频率|frequency|周期|cycles?|记录|时长|duration|幅值|amplitude'
    out = []
    for sentence in re.split(r'[。；;!?！？\n]+|然后|接着|另外|此外|\bthen\b', text):
        group = None
        for piece in re.split(r'[，,](?!\s*\d)', sentence):
            if not piece.strip(): continue
            parameter = _find(rf'^\s*(?:{"|".join(PARAMS)})', piece) and not _find(ADD + '|' + REMOVE, piece)
            if group is not None and (parameter or not _find(starts, piece)): group += '，' + piece
            else:
                if group is not None: out.append(group)
                group = piece
        if group is not None: out.append(group)
    return [c for c in (c.strip() for c in out) if c]


def interpret(text, scene):
    """{'scene', 'operations', 'unparsed', 'changed'}: `text` applied to a copy of `scene` (bad text never raises)."""
    require(isinstance(text, str) and 1 <= len(text.strip()) <= 2000, '请输入 1–2000 个字符的描述。')
    require(isinstance(scene, dict) and isinstance(scene.get('geometry'), dict) and isinstance(scene.get('source'), dict), '需要当前模型。')
    plan = Plan(scene)
    for clause in clauses(normalize(text)):
        try:
            if not plan.apply(clause): plan.unparsed.append(clause)
        except (ValueError, KeyError, TypeError) as error:
            plan.record(clause, f'未执行：{error}', applied=False)
    return {'scene': plan.scene, 'operations': plan.operations, 'unparsed': plan.unparsed, 'changed': any(op['applied'] for op in plan.operations)}

"""Automatic mesh: the software chooses the GLL degree and the element size; the user never sets a mesh.

The rule follows what this software has measured (research/long-record-accuracy, research/scattering):

* accuracy: nodes per shear wavelength at twice the highest −30 dB band edge of any source (the "2f resolution"
  shown in the messages). Target 10 for records shorter than 1 ms, 15 from 1 ms on (1 ms plate
  measurement: 11.5 → about 1 % at the end of the record, 15 → about 0.1 %). The hard floor stays 8;
* the element size stays in the network's validated range 0.8–2.5 mm, degree 4;
* defects need no extra rule: the defect mesher puts grid lines through every defect edge, so crack tips,
  delamination fronts, notch corners and hole patches are always exact element boundaries;
* budget: the machine, not the display. A model may use up to 80 % of the physical memory (`offline.resources`:
  about 1.3 kB per node) and the mesher's own size limit; the result animation draws a reduced display mesh and a
  large model keeps a compact record, so neither limits the mesh. A model too large for the target is coarsened step
  by step (and the report says so); one too large even at the coarsest admissible size is refused with advice.
"""
import math

DEGREE = 4
H_MIN, H_MAX = .8, 2.5
TARGET_SHORT, TARGET_LONG, MINIMUM = 10., 15., 8.
LONG_US = 1000.


def band_top_mhz(scene):
    """Highest second-harmonic frequency the mesh must resolve: twice the −30 dB band edge of every source (at least 2f).
    The same definition as the "2f resolution" in the preview messages, the long-record guidance and the time layer."""
    from graphnet.timestep import band_top_mhz as top
    return top(scene)


def shear_speed(scene):
    from physics.materials import slowest_shear_speed
    try: return slowest_shear_speed(scene['material'])
    except Exception:
        from solid.schema import MATERIAL
        return math.sqrt(MATERIAL['mu_pa'] / MATERIAL['rho_kg_m3'])


def target_for(scene):
    duration = (scene.get('acquisition') or {}).get('duration_us') or 0
    return TARGET_LONG if duration >= LONG_US else TARGET_SHORT


def accuracy_h(scene, degree=DEGREE):
    """Element size (mm) that reaches the target 2f resolution, before range and budget limits."""
    return shear_speed(scene) / (band_top_mhz(scene) * 1e6 * target_for(scene)) * 1000 * degree


def resolution(scene, h_mm, degree=DEGREE):
    return shear_speed(scene) / (band_top_mhz(scene) * 1e6) / (h_mm / degree * 1e-3)


def over_budget(nodes):
    """None, or why a mesh with this many nodes cannot run on this machine."""
    from .resources import memory_problem
    return memory_problem(nodes)


def _size_error(error):
    """True for a mesher's own size limit (the frozen meshers say 'budget'), false for every other mesh error."""
    return 'budget' in str(error) or '上限' in str(error)


def choose(raw, validate, degree=DEGREE):
    """(validated scene, report). `validate(raw, degree)` is the request validator for a scene with numerics.h_mm set."""
    import copy
    from spectral3d.feature_mesh import make_mesh
    wanted = accuracy_h(raw, degree)
    floor_h = shear_speed(raw) / (band_top_mhz(raw) * 1e6 * MINIMUM) * 1000 * degree   # coarsest size that keeps 8 nodes per 2f wavelength
    first = min(H_MAX, max(H_MIN, wanted)); last = max(first, min(H_MAX, floor_h))
    limited = 'accuracy' if H_MIN <= wanted <= H_MAX else ('h_max' if wanted > H_MAX else 'h_min')
    candidates, h = [], first
    while h < last - 1e-9: candidates.append(h); h *= 1.08
    candidates.append(last)
    from spectral3d.feature_mesh import has_defects, plate_node_count
    problem = None
    for i, h in enumerate(candidates):
        trial = copy.deepcopy(raw); trial.setdefault('numerics', {})['h_mm'] = math.floor(h * 1e4) / 1e4   # round down: never coarser than chosen
        scene = validate(trial, degree)
        if scene['geometry']['kind'] in ('plate', 'notch') and not has_defects(scene):   # the size is known before the mesh is built
            problem = over_budget(plate_node_count(scene, degree, scene['numerics']['h_mm'])[0])
            if problem: continue
        try: mesh = make_mesh(scene, degree, scene['numerics']['h_mm'])
        except ValueError as error:
            if not _size_error(error): raise
            problem = '超过网格生成上限的节点数（管和带孔板几何为 10 万，含缺陷的平板为 100 万）'; continue
        problem = over_budget(len(mesh['points']))
        if not problem:
            if i: limited = 'budget'
            break
    else:
        raise ValueError(f'模型太大：在保持 2f 分辨率 ≥ {MINIMUM:g} 的最粗网格（单元 {candidates[-1]:.2f} mm）下仍有 {problem}。'
                         '请缩短板长、改用半宽模型（y 最大面对称）、降低激励频率，或减少贯穿板厚方向的缺陷层次（裂纹深度、分层深度各自会增加一层单元）。')
    achieved = resolution(scene, scene['numerics']['h_mm'], degree); target = target_for(scene)
    reason = {'accuracy': f'按 2f 剪切波长 {target:g} 个节点选取', 'h_max': f'频率较低，取最大单元 {H_MAX} mm（分辨率已超过目标）',
              'h_min': f'频率较高，取最小单元 {H_MIN} mm', 'budget': '为满足本机内存或网格生成上限而放粗'}[limited]
    report = {'mode': 'auto', 'degree': degree, 'h_mm': scene['numerics']['h_mm'], 'node_spacing_mm': scene['numerics']['h_mm'] / degree,
              'nodes': int(len(mesh['points'])), 'elements': int(len(mesh['elements'])), 'min_node_spacing_mm': float(mesh['min_node_spacing']) * 1000,
              'band_top_mhz': band_top_mhz(scene), 'nodes_per_2f_shear_wavelength': achieved, 'target': target, 'limited_by': limited, 'reason': reason,
              'curved': str(mesh.get('mapping_kind', '')) == 'isoparametric', 'split_nodes': int(sum(mesh.get('split_nodes', []))), 'contact_pairs': int(len(mesh.get('contact_a', [])))}
    if achieved < target * (1 - 1e-3):
        report['warning'] = (f'2f 分辨率 {achieved:.1f}，低于目标 {target:g}' + ('（受本机内存或网格生成上限限制）' if limited == 'budget' else '') +
                             '；长记录末段误差会增大。可缩短板长、使用半宽模型或降低频率以提高分辨率。')
    return scene, report

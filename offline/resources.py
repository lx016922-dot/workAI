"""What a network computation needs and what this machine has: memory, disk and time.

The size of a model is limited by the machine, not by the result display: the animation uses a reduced
display mesh (`spectral3d.display`) and a large model keeps a compact record (`graphnet.rollout`).

* memory: about 0.3 kB per node for the mesh and element geometry plus 0.56 kB per node for each constitutive control
  (position, velocity, acceleration, external force and the temporaries of one layer): 2.0 kB with all three controls,
  0.86 kB with the complete response alone; a chunk of output is at most 512 MiB. Measured on 13 689 and 229 293
  nodes (docs/large-models.md 6.4): 0.25 kB + 0.51 kB per control; the constants add about 10 %.
  A model is refused when the estimate exceeds 80 % of the physical memory;
* disk: a full record is frames × nodes × 72 bytes; a compact one holds the receivers, the display animation, four
  full snapshots and two checkpoints;
* time: element-message evaluations × nodes / the rate measured on this machine by earlier computations
  (`throughput.json` in the data directory; a typical desktop value before the first one). The controls run one per
  thread, so with enough cores the complete (Murnaghan) control sets the time; with fewer cores than controls the
  others add to it (`relative_time`).
"""
import json
import math
import os
from pathlib import Path

BASE_BYTES = 250 * 1024 ** 2
FIXED_NODE_BYTES = 300
CONTROL_NODE_BYTES = 560
NODE_BYTES = FIXED_NODE_BYTES + 3 * CONTROL_NODE_BYTES   # all three controls
# CPU time of each control relative to the complete one, measured on whole records (docs/large-models.md 6.4): the element
# message alone is cheaper for the linear and StVK controls, the state update costs the same for all three.
CONTROL_COST = {'linear': .5, 'stvk': .55, 'murnaghan': 1.}
MEMORY_FRACTION = .8
FULL_RECORD_BYTES = 2 * 1024 ** 3      # 'auto' keeps every node of every sample up to this size
DEFAULT_RATE = 2.5e6                   # node × message evaluations per second of the slowest control (Murnaghan), one thread
OMEGA_SPACING = 3.6                    # ω_max ≈ 3.6 × fastest speed / smallest node spacing (upper edge of the meshes measured; estimates only)
RECORDS = ('auto', 'full', 'compact')
BROWSER_NODES, BROWSER_ENTRIES, BROWSER_TRIANGLES = 60000, 200000, 50000


def physical_memory():
    """Physical memory in bytes, or None when the platform does not say."""
    try:
        if os.name == 'nt':
            import ctypes

            class Status(ctypes.Structure):
                _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong), ('total', ctypes.c_ulonglong), ('available', ctypes.c_ulonglong),
                            ('total_page', ctypes.c_ulonglong), ('available_page', ctypes.c_ulonglong), ('total_virtual', ctypes.c_ulonglong),
                            ('available_virtual', ctypes.c_ulonglong), ('extended', ctypes.c_ulonglong)]
            status = Status(); status.length = ctypes.sizeof(Status)
            return int(status.total) if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)) else None
        return int(os.sysconf('SC_PAGE_SIZE')) * int(os.sysconf('SC_PHYS_PAGES'))
    except (AttributeError, ValueError, OSError):
        return None


def record_mode(requested, frames, nodes, controls=3):
    """'full' or 'compact' for a request's record setting."""
    if requested in ('full', 'compact'): return requested
    return 'full' if frames * nodes * 24 * controls <= FULL_RECORD_BYTES else 'compact'


def memory_bytes(nodes, record, display_nodes=0, controls=3, frames=None):
    """Estimated peak memory of the worker process: the network state plus one chunk of output.

    A full record holds a chunk of at most 512 MiB (less for a short record, when `frames` is given)."""
    if record == 'full': chunk = min(512 * 1024 ** 2, frames * nodes * 24 * controls) if frames else 512 * 1024 ** 2
    else: chunk = min(256 * 1024 ** 2, 256 * display_nodes * 12 * controls) + 2 * nodes * 24 * controls
    return BASE_BYTES + nodes * (FIXED_NODE_BYTES + controls * CONTROL_NODE_BYTES) + chunk


def relative_time(controls, threads=None):
    """Wall time of these controls relative to the complete control alone on one thread.

    One thread per control: with at least as many threads as controls the slowest control sets the time; with fewer
    the controls share the threads."""
    threads = max(1, min(len(controls), threads or os.cpu_count() or 1)); costs = [CONTROL_COST[name] for name in controls]
    return max(max(costs), sum(costs) / threads)


def memory_problem(nodes, record='compact', controls=3):
    """None, or why this machine cannot hold the model."""
    total = physical_memory(); need = memory_bytes(nodes, record, controls=controls)
    if total is None or need <= MEMORY_FRACTION * total: return None
    return f'{nodes} 个节点约需 {need / 1024 ** 3:.1f} GB 内存（本机物理内存 {total / 1024 ** 3:.1f} GB，上限为其 {MEMORY_FRACTION:.0%}）'


def throughput(data_directory):
    """Measured node × evaluations per second of this machine, or the default before the first computation."""
    try:
        value = json.loads((Path(data_directory) / 'throughput.json').read_text(encoding='utf-8'))['node_evaluations_per_s']
        if isinstance(value, (int, float)) and math.isfinite(value) and value > 0: return float(value), True
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return DEFAULT_RATE, False


def save_throughput(data_directory, nodes, evaluations_per_control, seconds, controls=('linear', 'stvk', 'murnaghan')):
    """Remember this machine's rate (of the complete control on one thread) from a finished computation.

    Compilation of the first run is included, so small runs are skipped."""
    if seconds < 20 or nodes * evaluations_per_control < 1e8: return
    path = Path(data_directory) / 'throughput.json'
    try:
        rate = nodes * evaluations_per_control * relative_time(controls) / seconds
        tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps({'node_evaluations_per_s': rate}), encoding='utf-8'); tmp.replace(path)
    except OSError:
        pass


def estimate(scene, mesh, time_stepping='auto', record='auto', data_directory=None, controls=('linear', 'stvk', 'murnaghan')):
    """Plan and cost of a network computation before it starts: record mode, layer, memory, disk and time.

    The layer uses ω_max ≈ 3.6 c / Δx instead of the measured value (the worker measures it), so the time is an estimate."""
    import shutil
    from graphnet.model import speed_bound
    from graphnet.rollout import storage_bytes
    from graphnet.timestep import plan
    from physics.materials import fastest_speed, is_custom
    from spectral3d.display import FRAMES
    acquisition = scene['acquisition']; frames = round(acquisition['duration_us'] / acquisition['sample_step_us']) + 1; nodes = int(len(mesh['points']))
    count = len(controls); mode = record_mode(record, frames, nodes, count); material = scene['material']
    speed = fastest_speed(material) if is_custom(material) else math.sqrt((material['lambda_pa'] + 2 * material['mu_pa']) / material['rho_kg_m3'])
    from .domain import MODEL
    composition = json.loads((MODEL / 'time-coefficients.json').read_text(encoding='utf-8'))['weights']   # the trained composition
    layers = plan(scene, mesh, composition, speed_bound(scene), OMEGA_SPACING * speed / float(mesh['min_node_spacing']), time_stepping)
    # A display mesh is at most the browser budget; the surface of a small mesh is smaller still.
    display_nodes = min(BROWSER_NODES, nodes); shown = min(FRAMES, frames)
    disk = storage_bytes(frames, nodes, mode, display_nodes, shown, len(scene['receivers']), controls=count)
    rate, measured = throughput(data_directory) if data_directory else (DEFAULT_RATE, False)
    free = shutil.disk_usage(data_directory).free if data_directory else None
    return {'record': mode, 'record_requested': record, 'frames': frames, 'time_stepping': time_stepping, 'scheme': layers['scheme'], 'time_order': layers['time_order'],
            'layer_step_us': layers['layer_step_us'], 'evaluations_per_control': layers['evaluations_per_control'],
            'frozen_evaluations_per_control': layers.get('frozen_evaluations_per_control', layers['evaluations_per_control']), 'layer_reason': layers['reason'],
            'memory_bytes': memory_bytes(nodes, mode, display_nodes, count, frames), 'physical_memory_bytes': physical_memory(), 'disk_bytes': int(disk), 'free_bytes': free,
            'seconds': nodes * layers['evaluations_per_control'] * relative_time(controls) / rate, 'rate_measured': measured, 'controls': list(controls),
            'cpu_relative_to_all': sum(CONTROL_COST[name] for name in controls) / sum(CONTROL_COST.values())}

"""Synchronous mode pairs for cumulative second-harmonic generation.

For material (and geometric) nonlinearity the second-harmonic forcing of any
Lamb mode is symmetric, so the secondary mode is taken from the S family by
default. A pair (m1 at f, m2 at 2f) is reported where the phase velocities
cross exactly, and where they stay within a tolerance over an fd interval
(for example S0–S0 at low fd). Group-velocity matching and the power-flux
coupling are reported as information, not used as filters.
"""
import numpy as np


def _interp(mode, fd):
    grid = np.array(mode['fd_mhz_mm'])
    if fd < grid[0] or fd > grid[-1]: return None, None
    return float(np.interp(fd, grid, mode['phase_velocity_m_s'])), float(np.interp(fd, grid, mode['group_velocity_m_s']))


def synchronous_pairs(curves, tolerance=.01, secondary_family='S', thickness_mm=None):
    thickness = thickness_mm or curves['thickness_mm']; rows = []
    secondaries = [m for m in curves['modes'] if m['family'] == secondary_family]
    for primary in curves['modes']:
        fd = np.array(primary['fd_mhz_mm']); cp1 = np.array(primary['phase_velocity_m_s']); cg1 = np.array(primary['group_velocity_m_s'])
        for secondary in secondaries:
            mismatch = np.full(len(fd), np.nan); cp2 = np.full(len(fd), np.nan); cg2 = np.full(len(fd), np.nan)
            for i, value in enumerate(fd):
                c, g = _interp(secondary, 2 * value)
                if c is not None: cp2[i] = c; cg2[i] = g; mismatch[i] = (c - cp1[i]) / cp1[i]
            valid = np.isfinite(mismatch)
            for i in range(len(fd) - 1):
                if valid[i] and valid[i + 1] and mismatch[i] * mismatch[i + 1] < 0:
                    w = mismatch[i] / (mismatch[i] - mismatch[i + 1]); fd_star = fd[i] + w * (fd[i + 1] - fd[i])
                    rows.append({'kind': 'exact_crossing', 'primary': primary['name'], 'secondary': secondary['name'], 'fd_mhz_mm': float(fd_star),
                                 'frequency_mhz': float(fd_star / thickness), 'phase_velocity_m_s': float(cp1[i] + w * (cp1[i + 1] - cp1[i])),
                                 'group_velocity_primary_m_s': float(cg1[i] + w * (cg1[i + 1] - cg1[i])), 'group_velocity_secondary_m_s': float(cg2[i] + w * (cg2[i + 1] - cg2[i])),
                                 'phase_velocity_mismatch': 0.})
            inside = valid & (np.abs(mismatch) <= tolerance); i = 0
            while i < len(fd):
                if inside[i]:
                    j = i
                    while j + 1 < len(fd) and inside[j + 1]: j += 1
                    best = i + int(np.argmin(np.abs(mismatch[i:j + 1])))
                    rows.append({'kind': 'within_tolerance', 'primary': primary['name'], 'secondary': secondary['name'], 'fd_start_mhz_mm': float(fd[i]), 'fd_end_mhz_mm': float(fd[j]),
                                 'fd_mhz_mm': float(fd[best]), 'frequency_mhz': float(fd[best] / thickness), 'phase_velocity_m_s': float(cp1[best]),
                                 'group_velocity_primary_m_s': float(cg1[best]), 'group_velocity_secondary_m_s': float(cg2[best]), 'phase_velocity_mismatch': float(mismatch[best]), 'tolerance': tolerance})
                    i = j + 1
                else: i += 1
    rows.sort(key=lambda r: r['fd_mhz_mm'])
    return {'schema': 'workai.lamb_synchronism.v1', 'thickness_mm': thickness, 'secondary_family': secondary_family, 'tolerance': tolerance,
            'note': 'Material-nonlinearity second harmonics are symmetric; phase matching is necessary, not sufficient: nonzero power flux and group-velocity matching decide the usable distance.',
            'pairs': rows}


def main():
    import argparse, json
    parser = argparse.ArgumentParser(description='List phase-matched (f, 2f) Lamb mode pairs from a dispersion JSON.')
    parser.add_argument('--curves', required=True); parser.add_argument('--tolerance', type=float, default=.01); parser.add_argument('--output')
    args = parser.parse_args(); result = synchronous_pairs(json.load(open(args.curves)), tolerance=args.tolerance)
    text = json.dumps(result, indent=1)
    if args.output: open(args.output, 'w', encoding='utf-8').write(text + '\n')
    else: print(text)


if __name__ == '__main__': main()

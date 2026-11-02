"""Excitation design for a chosen Lamb mode: wavelength, comb pitch, wedge angle, timing, symmetry."""
import math
import numpy as np
from .lamb import mode_at


def excitation_design(curves, mode_name, frequency_mhz, distance_mm=None, cycles=5., delay_us=0., wedge_speed_m_s=2730., bandwidth_db=6.):
    from physics.signals import bandwidth_mhz
    thickness = curves['thickness_mm']; fd = frequency_mhz * thickness; at = mode_at(curves, mode_name, fd)
    if at is None: raise ValueError(f'{mode_name} is not propagating (or not tracked) at fd = {fd:.4g} MHz·mm.')
    cp = at['phase_velocity_m_s']; cg = at['group_velocity_m_s']; wavelength_mm = cp / (frequency_mhz * 1e6) * 1e3; family = mode_name[0]
    burst_us = cycles / frequency_mhz
    source = {'frequency_mhz': frequency_mhz, 'cycles': cycles, 'delay_us': delay_us, 'phase_deg': 0.}
    upper = bandwidth_mhz(source, bandwidth_db); half_band = max(upper - frequency_mhz, 1e-9)
    design = {'mode': mode_name, 'frequency_mhz': frequency_mhz, 'fd_mhz_mm': fd, 'phase_velocity_m_s': cp, 'group_velocity_m_s': cg, 'wavelength_mm': wavelength_mm,
              'comb_pitch_mm': wavelength_mm, 'comb_tooth_width_mm': wavelength_mm / 2,
              'wedge_angle_deg': math.degrees(math.asin(wedge_speed_m_s / cp)) if wedge_speed_m_s < cp else None, 'wedge_speed_m_s': wedge_speed_m_s,
              'half_model_boundary': {'face': 'zmax', 'type': 'symmetry' if family == 'S' else 'antisymmetry', 'note': 'half-thickness model z in [0, d/2] isolates this family'},
              'burst_duration_us': burst_us, f'bandwidth_{int(bandwidth_db)}db_mhz': [max(frequency_mhz - half_band, 1e-6), frequency_mhz + half_band], 'second_harmonic_mhz': 2 * frequency_mhz}
    secondary = mode_at(curves, 'S0', 2 * fd); design['secondary_S0_phase_velocity_m_s'] = secondary['phase_velocity_m_s'] if secondary else None
    if distance_mm is not None:
        travel_us = distance_mm * 1e-3 / cg * 1e6; margin = .25 * burst_us
        low = mode_at(curves, mode_name, max(frequency_mhz - half_band, 1e-6) * thickness); high = mode_at(curves, mode_name, (frequency_mhz + half_band) * thickness)
        spread = abs(distance_mm * 1e-3 / low['group_velocity_m_s'] - distance_mm * 1e-3 / high['group_velocity_m_s']) * 1e6 if low and high else None
        design.update(distance_mm=distance_mm, group_arrival_us=delay_us + travel_us, direct_window_us=[delay_us + travel_us - margin, delay_us + travel_us + burst_us + margin + (spread or 0.)],
                      dispersive_spread_us=spread, propagation_wavelengths=distance_mm / wavelength_mm)
    return design


def main():
    import argparse, json
    parser = argparse.ArgumentParser(description='Excitation design for a Lamb mode from a dispersion JSON.')
    parser.add_argument('--curves', required=True); parser.add_argument('--mode', required=True); parser.add_argument('--frequency-mhz', type=float, required=True)
    parser.add_argument('--distance-mm', type=float); parser.add_argument('--cycles', type=float, default=5.); parser.add_argument('--wedge-speed', type=float, default=2730.)
    args = parser.parse_args()
    print(json.dumps(excitation_design(json.load(open(args.curves)), args.mode, args.frequency_mhz, args.distance_mm, args.cycles, wedge_speed_m_s=args.wedge_speed), indent=1))


if __name__ == '__main__': main()

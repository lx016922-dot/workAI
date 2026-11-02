"""Receiver points and receiver lines expanded into named sampling points (any dimension)."""
import numpy as np
from .scene import receiver_lines


def receiver_points(scene):
    """Scene receivers followed by the expanded receiver lines; line points carry group, index and distance."""
    rows = [dict(rx, group=None, index=None, distance_mm=None) for rx in scene['receivers']]
    for line in receiver_lines(scene):
        start = np.asarray(line['from_mm'], dtype=float); end = np.asarray(line['to_mm'], dtype=float)
        for i in range(line['count']):
            point = start + (end - start) * i / (line['count'] - 1)
            rows.append({'name': f"{line['name']}[{i}]", 'point_mm': point.tolist(), 'direction': line['direction'], 'group': line['name'], 'index': i, 'distance_mm': float(np.linalg.norm(point - start))})
    return rows

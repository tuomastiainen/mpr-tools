#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

from shapely import geometry
from shapely.geometry import Point, LinearRing
from shapely.geometry.polygon import Polygon
from shapely import affinity

import random
import time

from shapely.ops import cascaded_union

from data_utils import timing

from phases import phases_optimized_low_ront

ACCURACY = 2500


def get_xy(theta_r):
    theta, r = theta_r
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return (x, y)


def polar_to_cartesian(theta, r):
    return [get_xy(theta_r) for theta_r in zip(theta, r)]


def generate_signals(polygon,
                     angles_deg,
                     samples=1024,
                     cpm_points=None,
                     vertical_error=None,
                     horizontal_error=None,
                     angles_deg_errors=None):

    cpm_points = np.array(cpm_points)
    if not vertical_error:
        vertical_error = 0
    if not horizontal_error:
        horizontal_error = 0

    if angles_deg_errors:
        for index, angle in enumerate(angles_deg):
            angles_deg[index] += angles_deg_errors[index]

    angles = [((i / 360) * 2 * np.pi) for i in angles_deg]

    probe_distance = 1000
    probe_pos = [Point(probe_distance * np.cos(angle), probe_distance * np.sin(angle)) for angle in angles]
    probe_lines = [geometry.LineString([Point(0, 0), pos]) for pos in probe_pos]

    thetas = np.linspace(0, np.pi * 2, samples, endpoint=False)

    # initialize list of signals
    signals = [np.empty(len(thetas)) for _ in range(len(angles))]

    for theta_index, theta in enumerate(thetas):

        translated = affinity.translate(polygon, horizontal_error + cpm_points[theta_index][0],
                                        vertical_error + cpm_points[theta_index][1])
        rotated = affinity.rotate(translated, theta, use_radians=True, origin='centroid')

        for index, angle in enumerate(angles):
            line = probe_lines[index]

            intersection = line.intersection(rotated)
            dist = intersection.distance(probe_pos[index])

            signals[index][theta_index] = dist

    return signals


def create_unit_complex(phase):
    return np.cos(phase) + np.sin(phase) * 1j


def generate_reference_profile_100(profile_points, radius):
    from mpr_tools import get_roundness_profile
    phases = phases_optimized_low_ront
    fft = [0 + 0J] * (int(profile_points / 2))
    amplitude = -0.010  # 10 micrometers
    fourier_adjusted_amplitude = amplitude / (2 / profile_points)

    fd = np.array([create_unit_complex(phase) for phase in phases]) * fourier_adjusted_amplitude

    for index, value in enumerate(fd):
        fft[index] = value

    fft[0] = 0 + 0J
    fft[1] = 0 + 0J
    reverse_offset_fft = np.conjugate(np.flip(fft[1:]))

    fft = np.append(np.append(fft, [0]), reverse_offset_fft)

    roundness, theta, r = get_roundness_profile(fft, profile_points)
    imags = sorted([i.imag for i in np.fft.ifft(fft)])

    r = [i + radius for i in r]

    return theta, r


def generate_profile_signals(samples=1024,
                             angles=None,
                             cpm_points=None,
                             vertical_error=None,
                             horizontal_error=None,
                             angles_deg_errors=None):
    radius = 250
    polygon_points = ACCURACY
    theta, r = generate_reference_profile_100(polygon_points, radius)

    if not cpm_points:
        t = np.linspace(0, 2 * np.pi, samples)
        cpm_points = polar_to_cartesian(t, [0.00] * samples)

    points = polar_to_cartesian(theta, r)
    polygon = Polygon(points)
    if not angles:
        angles = [0, 38, 67, 180]

    signals = generate_signals(
        polygon,
        angles,
        samples=samples,
        cpm_points=cpm_points,
        vertical_error=vertical_error,
        horizontal_error=horizontal_error,
        angles_deg_errors=angles_deg_errors)

    profile = [radius, list(theta)[::-1], r]

    return signals, profile



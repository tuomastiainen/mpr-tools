#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################
# Multi probe roundness measurement algorithms #
# Tuomas Tiainen 2019                          #
# tuomas.tiainen@aalto.fi                      #
################################################

import time
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from math import sin, cos, sqrt, floor
from cmath import exp
import traceback
import warnings
import os

import json
from pprint import pprint
import uuid
import pickle

from mpl_toolkits.mplot3d import Axes3D

from generate_signals import generate_profile_signals, polar_to_cartesian

from data_utils import average, remove_radius
from plot_utils import polar_plot

warnings.filterwarnings('ignore')

SAMPLES_IN_ROUND = 1024
FILTERLEVEL = 100


def ozono_f_coeff(signals, angles=None):
    if not angles:
        angles = [0, 38, 67]

    a_1, a_2, a_3 = angles[0], angles[1], angles[2]

    a_1 = (angles[0] / 360) * (2 * np.pi)
    a_2 = (angles[1] / 360) * (2 * np.pi)
    a_3 = (angles[2] / 360) * (2 * np.pi)

    def equations(p):
        w_2, w_3 = p
        return (sin(a_1) + w_2 * sin(a_2) + w_3 * sin(a_3), cos(a_1) + w_2 * cos(a_2) + w_3 * cos(a_3))

    w_2, w_3 = fsolve(equations, (1, 1))

    arr = np.array([signals[0], w_2 * signals[1], w_3 * signals[2]])
    s = arr.sum(axis=0)

    fft_s = np.fft.fft(s, SAMPLES_IN_ROUND)

    ozono_coefficients = [0 + 0J] * SAMPLES_IN_ROUND

    for k, fft in enumerate(fft_s):
        alpha_k = (cos(k * a_1) + w_2 * cos(k * a_2) + w_3 * cos(k * a_3))
        beta_k = (sin(k * a_1) + w_2 * sin(k * a_2) + w_3 * sin(k * a_3))

        C_k = (2 * fft).real
        D_k = (2 * fft).imag

        A_k = (alpha_k * C_k - beta_k * D_k) / (alpha_k**2 + beta_k**2)
        B_k = (beta_k * C_k + alpha_k * D_k) / (alpha_k**2 + beta_k**2)

        ozono_coefficients[k] = (A_k + B_k * 1J) / 2

    coeff = np.array(ozono_coefficients)
    ecc = get_ecc(signals, angles)
    coeff[1] = ecc
    return coeff


def diameter_f_coeff(signals):
    deltar = 0.5 * np.add(signals[0], signals[1])
    fft = np.fft.fft(deltar, SAMPLES_IN_ROUND)
    return fft


def hybrid_merge(diameter, ozono):
    array = [0] * SAMPLES_IN_ROUND

    for index, item in enumerate(array):
        if (index % 2) == 0:
            array[index] = diameter[index]
        else:
            array[index] = ozono[index]

    return np.array(array)


def filter_fft(fft, filterlevel=FILTERLEVEL, include_ecc=False):
    fft = list(fft)

    # pad with zeros
    fft += ([0] * int(((SAMPLES_IN_ROUND / 2) - len(fft))))
    fft = np.array(fft)

    fft = fft[0:(int(SAMPLES_IN_ROUND / 2))]
    fft[0] = 0

    if not include_ecc:
        fft[1] = 0

    fft[filterlevel + 1:] = 0

    # the fourier is symmetric, use only first half, then mirror complex conjugates excluding first
    fft = np.append(fft, np.append([0], np.conj(fft[1:][::-1])))

    return fft


def get_roundness_profile(fft, samples=SAMPLES_IN_ROUND):
    theta = np.linspace(0, 2 * np.pi, samples, endpoint=False)
    r = np.fft.ifft(fft).real
    roundness = max(r) - min(r)
    return (roundness, theta, r)


def hybrid_f_coeff(signals, angles=None):

    if not angles:
        angles = [0, 38, 67]

    ozono = ozono_f_coeff([signals[0], signals[1], signals[2]], angles)
    diameter = diameter_f_coeff([signals[0], signals[3]])
    hybrid_fourpoint = hybrid_merge(diameter, ozono)
    return hybrid_fourpoint


def get_matrix_f(angles):
    def matrix(n):
        m = []
        for a in angles:
            m.append(np.array([cos(a), sin(a), exp(-1j * (n * a))]))

        return np.array(m)

    return matrix


def jansen_roundness_f_coeff(signals, angles=None, filterlevel=FILTERLEVEL):

    if not angles:
        angles = [0, 38, 67, 180]

    if len(angles) > len(signals):
        raise "Error: No signal for each angle."

    angles = [np.deg2rad(i) for i in angles]

    f = get_matrix_f(angles)
    fft_coefficients = [np.fft.fft(s, SAMPLES_IN_ROUND) for s in signals[0:len(angles)]]

    harmonics = np.arange(1, filterlevel + 1)
    r = []

    for harmonic in harmonics:
        h = f(harmonic)
        s_fourier_coefficients = [np.array(fft_coefficient[harmonic]) for fft_coefficient in fft_coefficients]

        ht = np.matrix.transpose(h)
        hth = np.matmul(ht, h)
        try:
            x_estimate = np.matmul(np.matmul(np.linalg.inv(hth), ht), s_fourier_coefficients)
        except np.linalg.linalg.LinAlgError:
            x_estimate = [0 + 0J] * 3

        r.append(x_estimate[2])

    fft = np.array([0 + 0J] * SAMPLES_IN_ROUND)

    for index, value in enumerate(r):
        fft[index + 1] = value

    fft = fft[0:(int(SAMPLES_IN_ROUND / 2))]
    fft[0] = 0 + 0J
    fft[1] = get_ecc(signals, angles)

    fft[filterlevel + 1:] = 0
    fft = np.append(fft, np.append([0], np.conj(fft[1:][::-1])))

    return fft


def get_ecc(signals, angles):
    """
    Calculate eccentric component based on average of probe signals
    """
    # option 1: use only S1 ecc component
    # f = np.fft.fft(signals[0])
    # return (f[1])

    # option 2: average all sensor ecc components
    phase_shift = [exp(-1j * -1 * i) for i in angles]
    fft_s = [np.fft.fft(s, SAMPLES_IN_ROUND)[1] for s in signals[0:len(angles)]]
    eccs = np.multiply(fft_s, phase_shift)
    ecc = np.mean(eccs)

    # TODO: option 3: determine ECC component by fitting circle in cpm

    return ecc


def plot_generated():
    s = 0
    cpm_points, cpm_data = r_cpm(s)
    angles_deg_errors = [0 for i in range(4)]
    vertical_error = 0
    horizontal_error = 0

    angles = [0, 38, 67, 180]
    signals, profile = generate_profile_signals(
        SAMPLES_IN_ROUND,
        angles=angles,
        cpm_points=cpm_points,
        vertical_error=vertical_error,
        horizontal_error=horizontal_error,
        angles_deg_errors=angles_deg_errors)

    signals = np.array(average(signals, SAMPLES_IN_ROUND))
    profile = remove_radius(profile)

    signals = [s * -1 for s in signals]
    angles = [0, 38, 67, 180]

    ozono = ozono_f_coeff(signals, angles[0:3])
    hybrid = hybrid_f_coeff(signals, angles=angles[0:3])
    jansen2 = jansen_roundness_f_coeff(signals, angles)

    filtered_ozono = filter_fft(ozono, include_ecc=False)
    filtered_hybrid = filter_fft(hybrid, include_ecc=False)
    filtered_jansen2 = filter_fft(jansen2, include_ecc=False)

    o = get_roundness_profile(filtered_ozono)
    h = get_roundness_profile(filtered_hybrid)
    ar2 = get_roundness_profile(filtered_jansen2)

    offset = 0.40
    r_max = 0.3
    step = 0.1

    f = 1
    offset = 0.40
    # polar_plot(profile, None, offset * f, True, None, f * r_max, f * 0.1, f, False)
    polar_plot(o, "Ozono", offset, False, None, r_max=r_max, step=step)
    polar_plot(h, "4 probe hybrid", offset, False, None, r_max=r_max, step=step)
    polar_plot(ar2, "4 probe LS", offset, False, None, r_max=r_max, step=step)
    plt.legend()
    plt.show()


def r_cpm(c=0.0, count=SAMPLES_IN_ROUND):
    """
    Function for generating random center point movement
    """
    t = np.linspace(0, 2 * np.pi, count)

    # cpm_points = polar_to_cartesian(t, [0.01] * 1024)
    # return cpm_points

    def rp():
        return np.random.random() * np.pi * 2

    def rd_cpm():
        rps = [rp() for i in range(5)]
        cpm = np.array([(np.sin(1 * i + rps[0]) + 0.8 * np.sin(2 * i + rps[1]) + 0.6 * np.sin(3 * i + rps[2]) +
                         0.4 * np.sin(4 * i + rps[3]) + 0.2 * np.sin(5 * i + rps[4])) for i in t])
        return [cpm, rps]

    rd = rd_cpm()
    cpm_x_points, cpm_x_phases = rd[0], rd[1]
    cpm_x = cpm_x_points * c

    rd = rd_cpm()
    cpm_y_points, cpm_y_phases = rd[0], rd[1]
    cpm_y = cpm_y_points * c

    return list(zip(cpm_x, cpm_y)), {"cpm_c": c, "cpm_x_phases": cpm_x_phases, "cpm_y_phases": cpm_y_phases}


def circle_cpm(amplitude=0.0, count=SAMPLES_IN_ROUND):
    t = np.linspace(0, 2 * np.pi, count)
    cpm_points = polar_to_cartesian(t, [amplitude] * SAMPLES_IN_ROUND)
    return cpm_points


def nr(mu, sigma):
    # normally distributed random number
    # mean and standard deviation
    return np.random.normal(mu, sigma, 1)[0]


if __name__ == "__main__":
    plot_generated()

#!/usr/bin/env python

import numpy as np
import lvm_read

import time

SAMPLES_IN_ROUND = 1024


def timing(f):
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        print('{:s} function took {:.3f} ms'.format(f.__name__, (time2 - time1) * 1000.0))
        return ret

    return wrap


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def RRMSE(S, S_e):
    S = S[2]
    S_e = S_e[2]

    squaresum = []

    for index, value in enumerate(S):
        squaresum.append((S[index] - S_e[index])**2)

    squaresum = sum(squaresum)

    return 1 / ((max(S) - min(S)) * np.sqrt((1 / len(S)) * squaresum))


def RPPE(S, S_e):
    S = S[2]
    S_e = S_e[2]
    return abs(max(S) - min(S) - (max(S_e) - min(S_e))) / (max(S) - min(S_e))


def remove_radius(profile):
    return [profile[0], profile[1], [i - profile[0] for i in profile[2]]]


def average(data, samples_in_round=SAMPLES_IN_ROUND, remove_first_round=False, rounds=None):
    # remove first round samples
    if remove_first_round:
        data = [signal[SAMPLES_IN_ROUND:] for signal in data]

    if rounds:
        start = int(samples_in_round * rounds[0])
        end = int(samples_in_round * rounds[1])
        data = [signal[start:end] for signal in data]

    for index, signal in enumerate(data):
        s_mean = np.mean(signal)
        data[index] = np.array(signal) - s_mean

    rounds = int(len(data[0]) / samples_in_round)

    for index, signal in enumerate(data):
        s_chunks = list(chunks(signal, samples_in_round))
        signal = [np.mean([val[i] for val in s_chunks]) for i in range(samples_in_round)]
        data[index] = signal

    # this data contains samples_in_round points averaged over each round
    return [np.array(s) for s in data]

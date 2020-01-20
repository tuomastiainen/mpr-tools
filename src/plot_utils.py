#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np


def polar_plot(data,
               label=None,
               offset=None,
               plot_circles=False,
               title=None,
               r_max=0.04,
               step=0.01,
               f=1,
               unitlabel=True):



    def round_nearest(x, a):
        return round(x / a) * a

    r_max = r_max

    roundness, theta, r = data
    r = [f * i for i in r]

    r = [i + abs(min(r)) for i in r]

    r = list(r)
    theta = list(theta)
    r.append(r[0])
    theta.append(theta[0])

    ax = plt.subplot(111, projection='polar')
    line, = ax.plot(theta, r, linewidth=2.5, label=label)

    if offset:
        ax.set_rorigin(-1 * offset)

    if plot_circles:
        ax.plot(theta, [max(r)] * len(theta), linewidth=2, color='r', ls='dashed')

    step = step
    ax.set_rticks(np.arange(0, round_nearest((r_max + step), step), step))
    ax.set_rlabel_position(62.5)  # get radial labels away from plotted line
    ax.grid(True)

    if not title:
        pass
    else:
        ax.set_title("Roundness profile {}".format(title), va='bottom')

    label_position = ax.get_rlabel_position()

    unitlabel = True
    if unitlabel:
        ax.text(
            np.radians(label_position + 5),
            ax.get_rmax() / .9,
            r'$\rm mm$',
            rotation=label_position,
            ha='center',
            va='center')


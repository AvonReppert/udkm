# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import udkm.tools.colors as colors


def offset_plot(x_list, y_list, label_list, offset=0.1, x_text=0, y_text=0, **kwargs):
    if not("index_array" in kwargs):
        index_array = np.arange(0, len(x_list), 1)
    else:
        index_array = kwargs["index_array"]

    gs = gridspec.GridSpec(1, 1, wspace=0, hspace=0)
    ax1 = plt.subplot(gs[:, 0])

    for counter, index in enumerate(index_array):
        color = colors.cmap_blue_red_3(counter/(len(index_array)-1))
        ax1.plot(x_list[index], y_list[index] + counter*offset,
                 label=label_list[index], lw=2, color=color)
        ax1.axhline((counter*offset), ls="--", color=color)
        ax1.annotate(label_list[index], xy=(x_text, counter*offset+y_text), xycoords='data',
                     xytext=(x_text, counter*offset+y_text), fontsize=8, color=color)


def list_plot(x_list, y_list, label_list, **kwargs):
    if not("index_array" in kwargs):
        index_array = np.arange(0, len(x_list), 1)
    else:
        index_array = kwargs["index_array"]
    gs = gridspec.GridSpec(1, 1, wspace=0, hspace=0)
    ax1 = plt.subplot(gs[:, 0])
    for counter, index in enumerate(index_array):
        color = colors.cmap_blue_red_3(counter/(len(index_array)-1))
        ax1.plot(x_list[index], y_list[index],
                 label=label_list[index], lw=1.5, color=color)

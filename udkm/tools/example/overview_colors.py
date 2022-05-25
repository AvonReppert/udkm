# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 21:01:12 2022

@author: Aleks
"""


import udkm.tools.colors as colors
import numpy as np
import matplotlib.pyplot as plt


""" multicolor colormaps """


gradient = np.linspace(0, 1, 256)
gradient = np.vstack((gradient, gradient))


def plot_color_gradients(category, cmap_list):
    # Create figure and adjust figure height to number of colormaps
    nrows = len(cmap_list)
    figh = 0.35 + 0.15 + (nrows + (nrows - 1) * 0.1) * 0.22
    fig, axs = plt.subplots(nrows=nrows + 1, figsize=(6.4, figh))
    fig.subplots_adjust(top=1 - 0.35 / figh, bottom=0.15 / figh,
                        left=0.2, right=0.99)
    axs[0].set_title(f'{category} colormaps', fontsize=14)
    counter = 0
    for ax, name in zip(axs, cmap_list):
        ax.imshow(gradient, aspect='auto', cmap=cmap_list[counter])
        ax.text(-0.01, 0.5, name_list[counter], va='center', ha='right', fontsize=10,
                transform=ax.transAxes)
        counter += 1

    # Turn off *all* ticks & spines, not just the ones with colormaps.
    for ax in axs:
        ax.set_axis_off()
    plt.savefig(category+".png", dpi=300, bbox_inches="tight")


""" multicolor lists """
cmap_list = [colors.cmap_1, colors.cmap_2, colors.cmap_3, colors.fireice()]
name_list = ["cmap_1", "cmap_2", "cmap_3", "fireice"]
plot_color_gradients('multicolor', cmap_list)

""" blue lists """
cmap_list = [colors.cmap_blue_1, colors.cmap_blue_2, colors.cmap_blue_3,
             colors.cmap_blue_4, colors.cmap_blue_5]
name_list = ["blue_1", "blue_2", "blue_3", "blue_4", "blue_5"]
plot_color_gradients('blue', cmap_list)

""" red lists """
cmap_list = [colors.cmap_red_1, colors.cmap_red_2, colors.cmap_red_3, colors.cmap_red_4]
name_list = ["red_1", "red_2", "red_3", "red_4"]
plot_color_gradients('red', cmap_list)

""" gold lists """
cmap_list = [colors.cmap_gold_1, colors.cmap_gold_2, colors.cmap_gold_3, colors.cmap_gold_4]
name_list = ["gold_1", "gold_2", "gold_3", "gold_4"]
plot_color_gradients('gold', cmap_list)

""" bipolar lists """
cmap_list = [colors.cmap_blue_red_1, colors.cmap_blue_red_2, colors.cmap_blue_red_3, colors.cmap_blue_red_4,
             colors.cmap_blue_red_5, colors.cmap_red_blue_1, colors.cmap_red_blue_2
             ]
name_list = ["blue_red_1", "blue_red_2", "blue_red_3", "blue_red_4",
             "blue_red_5", "red_blue_1", "red_blue_2"
             ]
plot_color_gradients('bipolar', cmap_list)

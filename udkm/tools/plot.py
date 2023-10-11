# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import udkm.tools.colors as colors
import cv2
import os
import imageio


def list_plot(x_list, y_list, label_list, **kwargs):
    """plots y_list versus x_list with legend entries from the label_list

    this function reqires that x_list and y_list contain arrays that are equally long for each index.
    It provides a way to quickly plot an overview where the line color is chosen from a continous colormap.
    Labels are taken from label_list, the order of plots can be specified by index_list and the colormap
    can be set using the cmap.

    Parameters
    -----------
    x_list : list of 1D numpy arrays
        list of x-values
    y_list : list of 1D numpy arrays
        list of y-values to plot
    label_list : list of strings
        labels for the functions
    **kwargs : dict , optional
        index_array : list
            list of indices that can be used to manually set the order of plots
        cmap : matplotlib linear segmented colormap
            user defined colormap - default is udkm.tools.colors.cmap_blue_red_3

    Returns
    --------
        ax1 : matplotlib figure axis

    Example
    --------
        >>> ax1 = list_plot(x_list,y_list,label_list,index_array = index_list)"""

    if not("index_array" in kwargs):
        index_array = np.arange(0, len(x_list), 1)
    else:
        index_array = kwargs["index_array"]

    if not("cmap" in kwargs):
        color_map = colors.cmap_blue_red_3
    else:
        color_map = kwargs["cmap"]

    gs = gridspec.GridSpec(1, 1, wspace=0, hspace=0)
    ax1 = plt.subplot(gs[:, 0])

    for counter, index in enumerate(index_array):
        color = color_map(counter/(len(index_array)-1))
        ax1.plot(x_list[index], y_list[index],
                 label=label_list[index], lw=1.5, color=color)
    return ax1


def offset_plot(x_list, y_list, label_list, offset=0.1, x_text=0, y_text=0, **kwargs):
    """plots y_list versus x_list with an offset and legend entries from the label_list adjacent to the curves

    this function reqires that x_list and y_list contain arrays that are equally long for each index.
    It provides a way to quickly plot an overview where the line color is chosen from a continous colormap.
    Labels are taken from label_list, the order of plots can be specified by index_list and the colormap.
    can be set using the cmap. The relative offset of the label-text can be set by adjusting x_text and y_text.

    Parameters
    -----------
    x_list : list of 1D numpy arrays
        list of x-values
    y_list : list of 1D numpy arrays
        list of y-values to plot
    label_list : list of strings
        labels for the functions that are written next to the curves
    x_text : float
        x-position of the label text in the x-coordinate system - default 0.0
    y_text : float
        y-position offset of the label text in the y-coordinate system - default 0.0

    **kwargs : dict , optional
        index_array : list
            list of indices that can be used to manually set the order of plots
        cmap : matplotlib linear segmented colormap
            user defined colormap - default is udkm.tools.colors.cmap_blue_red_3

    Returns
    --------
        ax1 : matplotlib figure axis

    Example
    --------
        >>> ax1 = list_plot(x_list,y_list,label_list,index_array = index_list)"""

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
    return ax1


def create_video_from_pngs(png_folder, output_file, frame_rate, output_format='avi'):
    """
    Create a video (AVI or GIF) from a series of numbered PNG files.

    Parameters
    ----------
    png_folder : str
        The path to the folder containing the numbered PNG files.
    output_file : str
        The path to the output video file (AVI or GIF).
    frame_rate : int
        The frame rate (frames per second) of the resulting video.
    output_format : str, optional
        The output video format ('avi' or 'gif'). Default is 'avi'.

    Returns
    -------
    None
    """
    # Ensure the folder paths have the correct format (forward slash).
    png_folder = os.path.normpath(png_folder)
    output_file = os.path.normpath(output_file)

    # Get the list of PNG files in the folder.
    png_files = [f for f in os.listdir(png_folder) if f.endswith('.png')]
    png_files.sort()  # Sort the files to ensure the correct order.

    # Check if there are PNG files to process.
    if not png_files:
        print("No PNG files found in the specified folder.")
        return

    # Load the first PNG image to get the dimensions.
    first_image = cv2.imread(os.path.join(png_folder, png_files[0]))
    height, width, layers = first_image.shape

    if output_format == 'avi':
        # Define the VideoWriter to create the AVI file.
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for AVI (XVID)
        out = cv2.VideoWriter(output_file, fourcc, frame_rate, (width, height))

        # Iterate through PNG files and add them to the AVI.
        for png_file in png_files:
            image_path = os.path.join(png_folder, png_file)
            frame = cv2.imread(image_path)
            out.write(frame)

        # Release the VideoWriter when done.
        out.release()

    elif output_format == 'gif':
        # Create a GIF using the imageio library.
        images = []
        for png_file in png_files:
            image_path = os.path.join(png_folder, png_file)
            frame = imageio.imread(image_path)
            images.append(frame)

        imageio.mimsave(output_file, images, duration=1/frame_rate)

    else:
        print("Invalid output format. Use 'avi' or 'gif'.")

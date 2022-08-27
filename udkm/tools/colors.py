# -*- coding: utf-8 -*-

import matplotlib


teststring = "Successfully loaded udkm.tools.colors"

##########################
# frequently used colors #
##########################

grey_1 = '#4b4b4bff'
grey_2 = '#7c7c7cff'
grey_3 = '#bfbfbfff'
grey_4 = '#dcdcdcff'


blue_1 = '#0048ccff'
blue_2 = '#aaccffff'
blue_3 = '#002f5eff'
blue_4 = '#0088b9ff'

red_1 = '#dc1700ff'
red_2 = '#fe8585ff'
red_3 = '#ff0000ff'
red_4 = '#ffa07aff'

orange_1 = '#ffd700ff'
orange_2 = '#ffa500ff'
orange_3 = '#d2691eff'
orange_4 = '#ffdeadff'

white = '#ffffff'
black = '#000000'
yellow = '#ffff00'
darkred = '#800000'


def hex_to_rgb(hx):
    """ returns an rgb tuple from a hexadecimal rgba input

    Parameters
    ----------
    hex : str
        hexadecimal code for the color usually extracted using the inkscape color picker format #0048ccff""

    Returns
    -------
    tuple : tuple of rgb values

    Example
    -------
    This example returns the rgb tuple of the hex code for the predefined blue_1

    >>> blue_1 = hex_to_rgb('#0048ccff')
    (0.0, 0.28125, 0.796875)

    """
    hx = hx.lstrip('#')
    return tuple(int(hx[i:i+2], 16)/256 for i in (0, 2, 4))


def make_colormap(seq):
    """ returns a matplotlib LinearSegmentedColormap from a sequence of rgb-tuples

    The floats that define the color spacing need to be strictly increasing and within the interval [0,1].

    Parameters
    ----------
    seq : list of tuples
        hexadecimal code for the color usually extracted using the inkscape color picker format #0048ccff""

    Returns
    -------
    tuple : tuple of rgb values

    Example
    -------
    This example shows how the cmap_blue_red_1 is created

    >>> make_colormap([hex_to_rgb(white), hex_to_rgb(blue_2), 0.15,
                       hex_to_rgb(blue_2), hex_to_rgb(blue_1), 0.4,
                       hex_to_rgb(blue_1), hex_to_rgb(blue_3), 0.6,
                       hex_to_rgb(blue_3), hex_to_rgb(grey_1), 0.8,
                       hex_to_rgb(grey_1), hex_to_rgb(grey_3)
                       ])

    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return matplotlib.colors.LinearSegmentedColormap('CustomMap', cdict)


########################
# multicolor colormaps #
########################

cmap_1 = make_colormap([hex_to_rgb(blue_2), hex_to_rgb(blue_1), 0.2,
                        hex_to_rgb(blue_1), hex_to_rgb(orange_1), 0.5,
                        hex_to_rgb(orange_1), hex_to_rgb(red_1)
                        ])

cmap_2 = make_colormap([hex_to_rgb(yellow), hex_to_rgb(red_1), 0.1,
                        hex_to_rgb(red_1), hex_to_rgb(orange_1), 0.4,
                        hex_to_rgb(orange_1), hex_to_rgb(blue_1), 0.8,
                        hex_to_rgb(blue_1), hex_to_rgb(blue_2)
                        ])

cmap_3 = make_colormap([(170/256, 204/256, 255/256), (0/256, 72/256, 204/256), 0.2,
                        (0/256, 72/256, 204/256), (220/256, 23/256, 0/256), 0.5,
                        (220/256, 23/256, 0/256), (255/256, 215/256, 0/256)])


def fireice():
    """Returns a self-defined analogue of the colormap fireice that can be called via colors.fireice()"""
    cdict = {'red':    ((0.0,  0.75, 0.75),
                        (1/6,  0, 0),
                        (2/6,  0, 0),
                        (3/6,  0, 0),
                        (4/6,  1.0, 1.0),
                        (5/6,  1.0, 1.0),
                        (1.0,  1.0, 1.0)),

             'green': ((0.0,  1, 1),
                       (1/6,  1, 1),
                       (2/6,  0, 0),
                       (3/6,  0, 0),
                       (4/6,  0, 0),
                       (5/6,  1.0, 1.0),
                       (1.0,  1.0, 1.0)),

             'blue':  ((0.0,  1, 1),
                       (1/6,  1, 1),
                       (2/6,  1, 1),
                       (3/6,  0, 0),
                       (4/6,  0, 0),
                       (5/6,  0, 0),
                       (1.0,  0.75, 0.75))}
    fireice = matplotlib.colors.LinearSegmentedColormap('fireice', cdict)
    return(fireice)

##################
# blue colormaps #
##################


cmap_blue_1 = make_colormap([hex_to_rgb(white), hex_to_rgb(blue_2), 0.15,
                             hex_to_rgb(blue_2), hex_to_rgb(blue_1), 0.4,
                             hex_to_rgb(blue_1), hex_to_rgb(blue_3), 0.6,
                             hex_to_rgb(blue_3), hex_to_rgb(grey_1), 0.8,
                             hex_to_rgb(grey_1), hex_to_rgb(grey_3)
                             ])

cmap_blue_2 = make_colormap([hex_to_rgb(blue_2), hex_to_rgb(blue_1), 0.5,
                             hex_to_rgb(blue_1), hex_to_rgb(blue_3)])

cmap_blue_3 = make_colormap([hex_to_rgb(blue_2), hex_to_rgb(blue_1), 0.33,
                             hex_to_rgb(blue_1), hex_to_rgb(blue_3), 0.66,
                             hex_to_rgb(blue_3), hex_to_rgb(grey_1)
                             ])

cmap_blue_4 = make_colormap([hex_to_rgb(white), hex_to_rgb(blue_2), 0.3,
                             hex_to_rgb(blue_2), hex_to_rgb(blue_1), 0.6,
                             hex_to_rgb(blue_1), hex_to_rgb(blue_3), 0.9,
                             hex_to_rgb(blue_3),  hex_to_rgb(grey_1)
                             ])

cmap_blue_5 = make_colormap([hex_to_rgb(white), hex_to_rgb(blue_2),
                             0.33, hex_to_rgb(blue_2), hex_to_rgb(blue_1), 0.66,
                             hex_to_rgb(blue_1), hex_to_rgb(orange_2)])

###############################
# yellow-orange-red colormaps #
###############################

cmap_gold_1 = make_colormap([hex_to_rgb(white), hex_to_rgb(yellow), 0.15,
                             hex_to_rgb(yellow), hex_to_rgb(orange_1), 0.25,
                             hex_to_rgb(orange_1), hex_to_rgb(orange_2), 0.4,
                             hex_to_rgb(orange_2), hex_to_rgb(orange_3), 0.6,
                             hex_to_rgb(orange_3), hex_to_rgb(red_2), 0.8,
                             hex_to_rgb(red_2), hex_to_rgb(red_1)
                             ])

cmap_gold_2 = make_colormap([hex_to_rgb(red_1), hex_to_rgb(orange_2), 0.5,
                             hex_to_rgb(orange_2), hex_to_rgb(orange_1)
                             ])

cmap_gold_3 = make_colormap([hex_to_rgb(red_1), hex_to_rgb(orange_3), 0.33,
                            hex_to_rgb(orange_3), hex_to_rgb(orange_2), 0.66,
                            hex_to_rgb(orange_2), hex_to_rgb(orange_1)
                             ])

cmap_gold_4 = make_colormap([hex_to_rgb(orange_1), hex_to_rgb(orange_2), 0.33,
                             hex_to_rgb(orange_2), hex_to_rgb(orange_3), 0.66,
                             hex_to_rgb(orange_3), hex_to_rgb(red_1)
                             ])

#################
# red colormaps #
#################

cmap_red_1 = make_colormap([hex_to_rgb(white), hex_to_rgb(orange_4), 0.2,
                            hex_to_rgb(orange_4), hex_to_rgb(orange_2), 0.6,
                            hex_to_rgb(orange_2), hex_to_rgb(red_1)
                            ])

cmap_red_2 = make_colormap([hex_to_rgb(white), hex_to_rgb(red_2), 0.15,
                            hex_to_rgb(red_2), hex_to_rgb(red_1), 0.4,
                            hex_to_rgb(red_1), hex_to_rgb(darkred), 0.6,
                            hex_to_rgb(darkred), hex_to_rgb(grey_1), 0.8,
                            hex_to_rgb(grey_1), hex_to_rgb(grey_3)
                            ])

cmap_red_3 = make_colormap([hex_to_rgb(white), hex_to_rgb(red_2), 0.3,
                            hex_to_rgb(red_2), hex_to_rgb(red_1), 0.6,
                            hex_to_rgb(red_1),  hex_to_rgb(darkred), 0.9,
                            hex_to_rgb(darkred), hex_to_rgb(grey_1)
                            ])

cmap_red_4 = make_colormap([hex_to_rgb(white), hex_to_rgb(red_2), 0.4,
                            hex_to_rgb(red_2), hex_to_rgb(red_1), 0.66,
                            hex_to_rgb(red_1),  hex_to_rgb(orange_2)
                            ])

#################################
# bipolar red to blue colormaps #
#################################

cmap_blue_red_1 = make_colormap([hex_to_rgb(blue_1), hex_to_rgb(blue_2), 1/3,
                                 hex_to_rgb(blue_2), hex_to_rgb(white), 1/2,
                                 hex_to_rgb(white), hex_to_rgb(red_2), 2/3,
                                 hex_to_rgb(red_2), hex_to_rgb(red_1)
                                 ])

cmap_blue_red_2 = make_colormap([hex_to_rgb(blue_1), hex_to_rgb(blue_2), 1/3,
                                 hex_to_rgb(blue_2), hex_to_rgb(orange_1), 1/2,
                                 hex_to_rgb(orange_1), hex_to_rgb(red_2), 2/3,
                                 hex_to_rgb(red_2), hex_to_rgb(red_1)
                                 ])

cmap_blue_red_3 = make_colormap([hex_to_rgb(blue_1), hex_to_rgb(blue_2), 1/4,
                                 hex_to_rgb(blue_2), hex_to_rgb(orange_1), 1/2,
                                 hex_to_rgb(orange_1), hex_to_rgb(red_2), 3/4,
                                 hex_to_rgb(red_2), hex_to_rgb(red_1)
                                 ])

cmap_blue_red_4 = make_colormap([hex_to_rgb(blue_1), hex_to_rgb(blue_2), 1/5,
                                 hex_to_rgb(blue_2), hex_to_rgb(orange_1), 2/4,
                                 hex_to_rgb(orange_1), hex_to_rgb(orange_2), 3/5,
                                 hex_to_rgb(orange_2), hex_to_rgb(red_2), 4/5,
                                 hex_to_rgb(red_2), hex_to_rgb(red_1)
                                 ])

cmap_blue_red_5 = make_colormap([(16/256, 73/256, 113/256), (1, 1, 1), 0.5, (1, 1, 1),
                                 (180/256, 16/256, 22/256)
                                 ])

cmap_red_blue_1 = make_colormap([hex_to_rgb(red_1), hex_to_rgb(red_2), 1/3,
                                 hex_to_rgb(red_2), hex_to_rgb(white), 1/2,
                                 hex_to_rgb(white), hex_to_rgb(blue_2), 2/3,
                                 hex_to_rgb(blue_2), hex_to_rgb(blue_1)
                                 ])

cmap_red_blue_2 = make_colormap([hex_to_rgb(red_1), hex_to_rgb(red_2), 1/3,
                                 hex_to_rgb(red_2), hex_to_rgb(orange_1), 1/2,
                                 hex_to_rgb(orange_1), hex_to_rgb(blue_2), 2/3,
                                 hex_to_rgb(blue_2), hex_to_rgb(blue_1)
                                 ])

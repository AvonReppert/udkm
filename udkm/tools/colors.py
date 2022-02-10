# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 21:29:02 2021

@author: Aleks
"""

import matplotlib 


teststring = "Successfully loaded udkm.tools.colors"

# Freuqently used colors 
blue1    = '#0048ccff'
red1     = '#dc1700ff'
orange1  = '#ffd700ff'
grey1    = '#4b4b4bff'

blue2    = '#aaccffff'
red2     = '#fe8585ff'
orange2  = '#ffa500ff'
grey2    = '#7c7c7cff'


blue3    = '#002f5eff'
red3     = '#ff0000ff'
orange3  = '#d2691eff'
grey3    = '#bfbfbfff'

blue4    = '#0088b9ff'
red4     = '#ffa07aff'
orange4  = '#ffdeadff'
grey4    = '#dcdcdcff'


white    = '#ffffff'
black    = '#000000'

yellow  = '#ffff00'
white   = '#ffffff'
darkred = '#800000'
black   = '#000000'
    
def hex2rgb(hx):
        hx = hx.lstrip('#')
        return tuple(int(hx[i:i+2], 16)/256 for i in (0, 2, 4))


def make_colormap(seq):
	"""Return a LinearSegmentedColormap
	seq: a sequence of floats and RGB-tuples. The floats should be increasing
	and in the interval (0,1).
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



def fireice():
    """Returns a self defined analog of the colormap fireice"""
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

#some colormaps used in dissertations 
dissColormap1       = make_colormap([(170/256,204/256,255/256), (0/256,72/256,204/256), 0.2, (0/256,72/256,204/256), (220/256,23/256,0/256), 0.5, (220/256,23/256,0/256), (255/256,215/256,0/256)])
dissColormap2       = make_colormap([(255/256,255/256,0/256),(255/256,215/256,0/256), 0.1, (255/256,215/256,0/256), (220/256,23/256,0/256), 0.4, (220/256,23/256,0/256), (0/256,72/256,204/256), 0.8, (0/256,72/256,204/256), (170/256,204/256,255/256) ])
dissColormapGold2   = make_colormap([(255/256,215/256,0/256), (255/256,165/256,0/256), 0.5, (255/256,165/256,0/256), (210/256,105/256,30/256)])
dissColormapBlue2   = make_colormap([(170/256,204/256,255/256),(0/256,72/256,204/256), 0.5, (0/256,72/256,204/256), (0/256,47/256,94/256)])
dissColormapGold3   = make_colormap([(255/256,215/256,0/256), (255/256,165/256,0/256), 0.33, (255/256,165/256,0/256), (210/256,105/256,30/256), 0.66, (210/256,105/256,30/256), (220/256,23/256,0/256)])
dissColormapGold3_r = make_colormap([(220/256,23/256,0/256), (210/256,105/256,30/256), 0.33,(210/256,105/256,30/256) ,(255/256,165/256,0/256) , 0.66,(255/256,165/256,0/256) , (255/256,215/256,0/256)])
dissColormapBlue3   = make_colormap([(170/256,204/256,255/256),(0/256,72/256,204/256), 0.33, (0/256,72/256,204/256), (0/256,47/256,94/256), 0.66, (0/256,47/256,94/256), (75/256,75/256,75/256)])
dissColormapGold    = make_colormap([(255/256,255/256,255/256),(255/256,255/256,0/256), 0.15, (255/256,255/256,0/256),(255/256,215/256,0/256), 0.25, (255/256,215/256,0/256), (255/256,165/256,0/256), 0.4, (255/256,165/256,0/256), (210/256,105/256,30/256), 0.6, (210/256,105/256,30/256), (220/256,23/256,0/256),0.8, (220/256,23/256,0/256), (254/256, 133/256, 133/256)])
dissColormapBlue    = make_colormap([(255/256,255/256,255/256),(170/256,204/256,255/256),0.15, (170/256,204/256,255/256),(0/256,72/256,204/256), 0.4, (0/256,72/256,204/256), (0/256,47/256,94/256), 0.6, (0/256,47/256,94/256), (75/256,75/256,75/256), 0.8, (75/256,75/256,75/256), (124/256,124/256,124/256)])
dissColormapRed2    = make_colormap([hex2rgb(white),hex2rgb(red2),0.15, hex2rgb(red2),hex2rgb(red1), 0.4, hex2rgb(red1), hex2rgb(darkred), 0.6, hex2rgb(darkred), hex2rgb(grey1), 0.8, hex2rgb(grey1), hex2rgb(grey3)])
dissColormapBlue2   = make_colormap([hex2rgb(white),hex2rgb(blue2),0.15, hex2rgb(blue2),hex2rgb(blue1), 0.4, hex2rgb(blue1), hex2rgb(blue3), 0.6, hex2rgb(blue3), hex2rgb(grey1), 0.8, hex2rgb(grey1), hex2rgb(grey3)])

dissColormapRed3    = make_colormap([hex2rgb(white),hex2rgb(red2), 0.3, hex2rgb(red2), hex2rgb(red1), 0.6, hex2rgb(red1),  hex2rgb(darkred), 0.9, hex2rgb(darkred), hex2rgb(grey1)])
dissColormapBlue3   = make_colormap([hex2rgb(white),hex2rgb(blue2),0.3, hex2rgb(blue2),hex2rgb(blue1), 0.6, hex2rgb(blue1), hex2rgb(blue3), 0.9,   hex2rgb(blue3),  hex2rgb(grey1)])

dissColormapRed4    = make_colormap([hex2rgb(white),hex2rgb(red2), 0.4, hex2rgb(red2), hex2rgb(red1), 0.66, hex2rgb(red1),  hex2rgb(orange2)])
dissColormapBlue4   = make_colormap([hex2rgb(white),hex2rgb(blue2),0.33, hex2rgb(blue2),hex2rgb(blue1), 0.66, hex2rgb(blue1), hex2rgb(orange2)])

dissColormapRed     = make_colormap([hex2rgb(white),hex2rgb(orange4), 0.2, hex2rgb(orange4), hex2rgb(orange2), 0.6, hex2rgb(orange2), hex2rgb(red1) ])

dissColormapGold2D  = make_colormap([hex2rgb(white),hex2rgb(yellow), 0.15, hex2rgb(yellow), hex2rgb(orange1), 0.25, hex2rgb(orange1), hex2rgb(orange2), 0.4, hex2rgb(orange2), hex2rgb(orange3), 0.6, hex2rgb(orange3), hex2rgb(red1),0.8, hex2rgb(red1), hex2rgb(red2)])
dissColormapGold1D  = make_colormap([hex2rgb(yellow),hex2rgb(orange1),2/7, hex2rgb(orange2),3/7, hex2rgb(orange3),4/7, hex2rgb(red1),5/7, hex2rgb(darkred),6/7,hex2rgb(grey1)])
dissColormapGold1D_r = make_colormap([hex2rgb(grey1),hex2rgb(darkred),2/7, hex2rgb(red1), 3/7, hex2rgb(orange3),4/7, hex2rgb(orange2),5/7, hex2rgb(orange1),6/7, hex2rgb(yellow)])

dissColormapGold1D2     = make_colormap([hex2rgb(orange1),hex2rgb(orange2),1/6, hex2rgb(orange3),2/6, hex2rgb(red1),3/6, hex2rgb(darkred),4/6,hex2rgb(grey1)])
dissColormapGold1D_r2   = make_colormap([hex2rgb(grey1),hex2rgb(darkred),1/6, hex2rgb(red1), 2/6, hex2rgb(orange3),3/6, hex2rgb(orange2),4/6,hex2rgb(orange1)])
dissColormapOrder       = make_colormap([hex2rgb(grey3),1/3,hex2rgb(orange1),2/3,hex2rgb(orange1),hex2rgb(blue2)])

dissColormapGold1D2         = make_colormap([hex2rgb(orange1),hex2rgb(orange2),1/6, hex2rgb(orange3),2/6, hex2rgb(red1),3/6, hex2rgb(darkred),4/6,hex2rgb(grey1)])
dissColormapGold1D_r3_old   = make_colormap([hex2rgb(grey2),hex2rgb(grey1),1/5, hex2rgb(red1), 2/5, hex2rgb(orange3),3/5, hex2rgb(orange2),4/5,hex2rgb(orange1)])

dissColormapGold1D4         = make_colormap([hex2rgb(orange1),hex2rgb(orange2),1/6, hex2rgb(orange3),2/6, hex2rgb(red1),3/6, hex2rgb(darkred)])
dissColormapGold1D_r4       = make_colormap([hex2rgb(grey1),hex2rgb(darkred),1/6, hex2rgb(red1), 2/6, hex2rgb(orange3),3/6, hex2rgb(orange2),4/6,hex2rgb(orange1)])

###corrected
dissColormapGold1D_3        = make_colormap([hex2rgb(orange1),hex2rgb(orange2), 1/8, hex2rgb(orange2), hex2rgb(orange3), 1/4, hex2rgb(orange3), hex2rgb(red1),2/4, hex2rgb(red1), hex2rgb(grey1),3/4,hex2rgb(grey1),hex2rgb(black)])
dissColormapGold1D_r3       = make_colormap([hex2rgb(black),hex2rgb(grey1), 1/8, hex2rgb(grey1), hex2rgb(red1), 1/4, hex2rgb(red1), hex2rgb(orange3),2/4, hex2rgb(orange3), hex2rgb(orange2),3/4,hex2rgb(orange2),hex2rgb(orange1)])
dissColormapGold1D_r2       = make_colormap([hex2rgb(red1), hex2rgb(orange3),1/3, hex2rgb(orange3), hex2rgb(orange2),2/3,hex2rgb(orange2),hex2rgb(orange1)])
dissColormapGold1D_2        = make_colormap([hex2rgb(orange1), hex2rgb(orange2),1/3, hex2rgb(orange2), hex2rgb(orange3),2/3,hex2rgb(orange3),hex2rgb(red1)])

dissColormapBlue1D_2        = make_colormap([hex2rgb(blue2), hex2rgb(blue1),1/2, hex2rgb(blue1), hex2rgb(blue3)])
dissColormapBlue1D_r2       = make_colormap([hex2rgb(blue3), hex2rgb(blue1),1/2, hex2rgb(blue1), hex2rgb(blue2)])

dissColormapRedBlue         = make_colormap([hex2rgb(blue1),hex2rgb(blue2),1/3, hex2rgb(blue2), hex2rgb(white),1/2, hex2rgb(white), hex2rgb(red2), 2/3, hex2rgb(red2), hex2rgb(red1)])
dissColormapBlueRed         = make_colormap([hex2rgb(red1),hex2rgb(red2),1/3, hex2rgb(red2), hex2rgb(white),1/2, hex2rgb(white), hex2rgb(blue2), 2/3, hex2rgb(blue2), hex2rgb(blue1)])

dissColormapRedBlue2        = make_colormap([hex2rgb(blue1),hex2rgb(blue2),1/3, hex2rgb(blue2), hex2rgb(orange1),1/2, hex2rgb(orange1), hex2rgb(red2), 2/3, hex2rgb(red2), hex2rgb(red1)])
dissColormapBlueRed2        = make_colormap([hex2rgb(red1),hex2rgb(red2),1/3, hex2rgb(red2), hex2rgb(orange1),1/2, hex2rgb(orange1), hex2rgb(blue2), 2/3, hex2rgb(blue2), hex2rgb(blue1)])

dissColormapRedBlue3        = make_colormap([hex2rgb(blue1),hex2rgb(blue2),1/4, hex2rgb(blue2), hex2rgb(orange1),1/2, hex2rgb(orange1), hex2rgb(red2), 3/4, hex2rgb(red2), hex2rgb(red1)])
dissColormapRedBlue4        = make_colormap([hex2rgb(blue1),hex2rgb(blue2),1/5, hex2rgb(blue2), hex2rgb(orange1),2/4, hex2rgb(orange1), hex2rgb(orange2), 3/5,hex2rgb(orange2), hex2rgb(red2), 4/5, hex2rgb(red2), hex2rgb(red1)])

    

# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math as math
from scipy.fft import fft
import matplotlib
import lmfit as lm
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from tkinter.filedialog import askopenfilename
import pickle


def round_it(num, k):
    rounded = round(num * 10.0 ** (k))
    rounded = rounded / 10 ** k
    return rounded


def plot_overview(line, parameters_example='verlauf', Fourier_Analyse=False, Fit_Analyse=False):
    parameter = pd.read_csv(r'parameters' +
                            '\\'+'parameters_'+parameters_example+'.txt', skiprows=1, sep='\t')
    date = parameter.date[line]
    time = parameter.time[line]
    infile = open(r'scan_export' +
                  '\\'+str(date)+'_'+str(time)+'.pickle', 'rb')
    new_dict = pickle.load(infile, encoding='bytes')
    wl = new_dict['wl']
    data = new_dict['data']
    delay = new_dict['delay']
    data_2D = new_dict['data_2D']
    ind = new_dict['ind']
    sliced = new_dict['sliced']

    title = parameter.name[line]
    fontSize = 24
    tickLabelFontSize = 10
    helpLineWidth = 0.5
    lineWidth = 2
    axLabelFontSize = 14
    fontSizelegend = 10

    # für slices oder nicht
    wlslicemin = 791
    wlslicemax = 801
    wlslice = 792
    wlslice2 = 796
    wlslice3 = 803

    wl_min = 780
    wl_max = 810

    yslicemin = np.abs(wlslicemin-wl).argmin()
    yslice = np.abs(wlslice-wl).argmin()
    yslice2 = np.abs(wlslice2-wl).argmin()
    yslice3 = np.abs(wlslice3-wl).argmin()
    yslicemax = np.abs(wlslicemax-wl).argmin()

    delay = np.abs(data[1:, 0])-np.abs(data[ind[0], 0])
    select = delay > 0.3  # since often the 0ps signal is saturated... fourier signal would be interfered
    delay_fourier = delay[select]

    # for 2D plot min and max
    delay_min = delay[0]
    delay_max = delay[len(delay)-1]
    intensity_slice_0ps = data[ind[0], 1:]
    if sliced == True:
        data_fourier = data_2D[:, yslice]/np.max(data_2D[:, yslice])
        data_fourier_2 = data_2D[:, yslice2]/np.max(data_2D[:, yslice2])
        data_fourier_3 = data_2D[:, yslice3]/np.max(data_2D[:, yslice3])
    else:
        data_fourier = data_2D[:, yslicemin:yslicemax]
        data_fourier_sum = np.sum(data_fourier, axis=1)/np.max(np.sum(data_fourier, axis=1))

    if sliced == True:
        fourier = np.abs(np.fft.rfft(data_fourier))
        dt = delay_fourier[1]-delay_fourier[0]
        df = 1/dt
        fmin = 1/(2*(max(delay_fourier)-min(delay_fourier)))
        fmax = 1/(2*(dt))
        fstep = fmin
        f = np.linspace(fmin, fmax, np.size(fourier))
        fourier_2D = np.array([f, fourier])
    else:
        try:
            fourier = np.zeros([math.ceil(len(data_fourier[:, 1])/2), len(data_fourier[1, :])])
            for i in range(len(data_fourier[1, :])):
                fourier[:, i] = np.abs(np.fft.rfft(data_fourier[:, i]))
        except:
            fourier = np.zeros([math.ceil(len(data_fourier[:, 1])/2+1), len(data_fourier[1, :])])
            for i in range(len(data_fourier[1, :])):
                fourier[:, i] = np.abs(np.fft.rfft(data_fourier[:, i]))
        dt = delay_fourier[1]-delay_fourier[0]
        df = 1/dt
        fmin = 1/(2*(max(delay_fourier)-min(delay_fourier)))
        fmax = 1/(2*(dt))
        fstep = fmin
        fouriersumme = np.sum(fourier, axis=1)
        f = np.linspace(fmin, fmax, np.size(fouriersumme))
        fourier_2D = np.array([f, fouriersumme])

    maximumfinder = int(10/fstep)
    # wei 10THz der maximal relevante bereich ist

    indfourier = np.unravel_index(np.argmax(fourier_2D[1, 0:maximumfinder], axis=None), np.array(fourier_2D).shape)
    # PLOT
    plt.figure(5, figsize=(12, 8), linewidth=2)
    gs = gridspec.GridSpec(2, 2,
                           width_ratios=[3, 1],
                           height_ratios=[1, 2],
                           wspace=0.0, hspace=0.0)
    X, Y = np.meshgrid(delay, wl)

    plt.suptitle(title, fontsize=fontSize)
    ax1 = plt.subplot(gs[0])
    if Fourier_Analyse == True:
        ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])
    ax4 = plt.subplot(gs[3])

    for ax in [ax1, ax3, ax4]:
        ax.tick_params('both', length=3, width=0.5, which='major', direction='in')
        ax.tick_params('both', length=1.5, width=0.5, which='minor', direction='in')
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.tick_params(axis='both', which='major', labelsize=tickLabelFontSize)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(helpLineWidth)
    if Fit_Analyse == True:
        # Initializes the fit model as the sum of a gaussian with a background
        model = lm.models.GaussianModel(prefix="g1_") + lm.models.LinearModel()
        parameters = lm.Parameters()
        xMin = -2
        xMax = 2
        # Name        StartValue  VaryParameter?     Min        Max
        parameters.add_many(('g1_center',     0,    True,     xMin,    xMax),
                            ('g1_sigma',              1,    True),
                            ('g1_amplitude',          1,    True),
                            ('slope',                 0,    True),
                            ('intercept',             0.5,    True))
        try:
            resultFit = model.fit(data_fourier, parameters, x=delay)
        except:
            resultFit = model.fit(data_fourier_sum[:], parameters, x=delay)
        sigmaFit = resultFit.values["g1_sigma"]
        ax1.plot(delay, resultFit.best_fit, 'r-', color="red", lw=lineWidth, label="fit Result")
        print(sigmaFit)

    if sliced == False:
        ax1.plot(delay, data_fourier_sum[:]/max(data_fourier_sum[:-1]), color='grey')
    else:
        ax1.plot(delay, data_fourier, color='yellowgreen')
        ax1.plot(delay, data_fourier_2, color='red')
        ax1.plot(delay, data_fourier_3, color='green')
    ax1.set_xlabel('delay [ps]', fontsize=axLabelFontSize)
    ax1.set_ylabel('intensity [a.u.]', fontsize=axLabelFontSize)
    ax1.xaxis.tick_top()
    ax1.xaxis.set_label_position("top")
    ax1.set_xlim(delay_min, delay_max)

    if Fourier_Analyse == True:
        ax2 = plt.subplot(gs[1])
        ax2.semilogy(f, fouriersumme, label=str(round_it(f[indfourier[1]+5], 2))+' THz at '+str(
            round_it(fouriersumme[np.argmax(fourier_2D[1, 5:maximumfinder])+5], 1)*0.001)+' [x 10$^3$ a.u.]')
        # ax2.semilogy(f,fourier/np.max(fourier[2:]),label=str(round_it(f[indfourier[1]+5],4)))
        # ax2.semilogy(f,fouriersumme/np.max(fouriersumme[2:]),label=str(round_it(f[indfouriersumme[1]+4],4)),color='red')
        ax2.legend(fontsize=fontSizelegend, loc='lower center')
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position("right")
        ax2.set_xlabel('frequency [THz]', fontsize=axLabelFontSize)
        ax2.set_ylabel('intensity [a.u.]', fontsize=axLabelFontSize)
        ax2.xaxis.tick_top()
        ax2.xaxis.set_label_position("top")
        # ax2.set_ylim(5e0,3e3)
        ax2.set_xlim(0.1, df/5)
        # ax2.text(0.6,1.3e3,'b)',fontsize=18)

    ax3 = plt.subplot(gs[2])
    ax3.axis([delay_min, delay_max, wl_min, wl_max])
    cmap = plt.get_cmap('jet')
    pl = ax3.pcolormesh(X, Y, np.transpose(data_2D),  norm=matplotlib.colors.LogNorm(vmin=data_2D.min(),
                        vmax=data_2D.max()), cmap=cmap)
    ax3.set_xlabel('delay [ps]', fontsize=axLabelFontSize)
    ax3.set_ylabel('wavelength [nm]', fontsize=axLabelFontSize)
    if sliced == True:
        ax3.hlines(wlslice, -10, 100, 'yellowgreen', '--', alpha=0.6)
        ax3.hlines(wlslice2, -10, 100, 'red', '--', alpha=0.6)
        ax3.hlines(wlslice3, -10, 100, 'green', '--', alpha=0.6)
    # ax3.hlines(wlslice,-10,45,'red','--',alpha=0.6)
    ax3.axvline(x=delay[ind[0]], color='grey', linewidth=1, linestyle="--")
    #ax3.axvline(x=delay[ind[0]-20], color ='grey', linewidth=1, linestyle="--")
    #ax3.axvline(x=delay[ind[0]+200], color ='grey', linewidth=1, linestyle="--")
    #ax3.axhline(y=794.2,color='grey' , linewidth=1,linestyle="--")
    #ax3.axhline(y=797.3,color='grey' , linewidth=1,linestyle="--")
    #ax3.axhline(y=791.7,color='grey' , linewidth=1,linestyle="--")

    axins3 = inset_axes(ax3,
                        width="60%",  # width = 10% of parent_bbox width
                        height="5%",  # height : 50%                   )
                        loc=4)
    ax3.add_patch(Rectangle((0.35, 0.018), 0.65, 0.17, edgecolor="none",
                  facecolor="white", alpha=0.90, transform=ax3.transAxes))
    cbar = plt.colorbar(pl, cax=axins3, orientation="horizontal")
    cbar.set_label('diffracted probe intensity (counts) ')
    # ax3.text(-2.9,807,'b)',color='white',fontsize=18)
    axins3.xaxis.set_ticks_position("top")
    axins3.xaxis.set_label_position("top")
    cl = plt.getp(cbar.ax, 'xmajorticklabels')
    plt.setp(cl, color="black")

    wavenumber = (1/794.2-1/wl)*10**7
    wn_min = (1/794.2-1/wl_min)*10**7
    wn_max = (1/794.2-1/wl_max)*10**7

    ax4 = plt.subplot(gs[3])
    ax4.plot(intensity_slice_0ps, wavenumber, color='grey')
    # ax4.plot(zunichtfourier2,wavenumber,color='grey')
    # ax4.plot(zunichtfourier3,wavenumber,color='grey')
    ax4.yaxis.tick_right()
    # ax4.text(1000,210,'c)',fontsize=18)
    ax4.yaxis.set_label_position("right")
    ax4.axis([0, np.max(intensity_slice_0ps), wn_min, wn_max])
    ax4.set_xlabel('intensity [counts]', fontsize=axLabelFontSize)
    ax4.set_ylabel('wavenumber [cm$^{-1}$]', fontsize=axLabelFontSize)

    #print('FWHM  = ' +str(sigmaFit*2.355) + '\t center = ' + str(muFit)  )
    print(str(round_it(f[indfourier[1]], 4)))
    print(str(round_it(fouriersumme[np.argmax(fourier_2D[1, 0:maximumfinder])], 4)))
    plt.savefig(r'plot_overview'+'\\'+str(title)+'.png', dpi=300)

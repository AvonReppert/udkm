import numpy as np
import matplotlib.pyplot as plt
import udkm.opp.functions as opp
import udkm.tools.functions as tools
import udkm.tools.colors as colors

import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Rectangle


parameter_file_name = "parameters/parameters_example_1.txt"

line = 0  # welchen Datensatz man auswerten möchte

params = opp.get_scan_parameter(parameter_file_name, line)

params["data_directory"] = "data\\"
params["probe_method"] = "transmission"

params["bool_force_reload"] = True

params["slice_wl"] = [1200, 2300]
params["slice_wl_width"] = [200, 200]

params["slice_delay"] = [1]
params["slice_delay_width"] = [0.5]


params["t0"] = -2008
params["exclude_loops"] = []
params["symmetric_colormap"] = False

scan = opp.load_data(params)

# %% clean data
pump_signal_errors = np.sum(scan["data_pump"], axis=0) <= 0
probe_signal_errors = np.sum(scan["data_probe"], axis=0) <= 0
errors = np.logical_or(pump_signal_errors, probe_signal_errors)
select_data = np.logical_not(errors)


scan["delay_raw_set"] = scan["delay_raw_set"][select_data]
scan["delay_raw_real"] = scan["delay_raw_real"][select_data]
scan["loop_array"] = scan["loop_array"][select_data]

scan["data_pump"] = scan["data_pump"][:, select_data]
scan["data_probe"] = scan["data_probe"][:, select_data]

scan["data_raw"] = (scan["data_pump"] - scan["data_probe"])/scan["data_probe"]

scan["data"] = np.zeros((len(scan["wavelength"]), len(scan["delay_raw_unique"])))

for i, t in enumerate(scan["delay_raw_unique"]):
    select_time = scan["delay_raw_set"] == t
    scan["data"][:, i] = np.mean(scan["data_raw"][:, select_time], axis=1)

# shift t0
scan["delay_set"] = scan["delay_raw_set"] - scan["t0"]
scan["delay_real"] = scan["delay_raw_real"] - scan["t0"]
scan["delay_unique"] = scan["delay_raw_unique"] - scan["t0"]

# generate wl slices
wl_slices = len(scan["slice_wl"])
if wl_slices > 0:
    scan["wl_slices"] = np.zeros((wl_slices, len(scan["delay_unique"])))
    scan["wl_labels"] = []
    for i, wl in enumerate(scan["slice_wl"]):

        wl_min = scan["slice_wl"][i] - scan["slice_wl_width"][i]
        wl_max = scan["slice_wl"][i] + scan["slice_wl_width"][i]
        t_min = np.min(scan["delay_unique"])
        t_max = np.max(scan["delay_unique"])
        _, _, _, scan["wl_slices"][i, :], _ = tools.set_roi_2d(scan["delay_unique"], scan["wavelength"], scan["data"],
                                                               t_min, t_max, wl_min, wl_max)
        scan["wl_labels"] = scan["wl_labels"].append(str(int(scan["slice_wl"][i])))

else:
    scan["wl_slices"] = np.zeros((wl_slices, 1))
    scan["wl_slices"][0, :] = np.sum(scan["data"], axis=0)
    scan["wl_labels"] = "integral"

# generate delay slices
t_slices = len(scan["slice_delay"])
if t_slices > 0:
    scan["delay_slices"] = np.zeros((t_slices, len(scan["wavelength"])))
    scan["delay_labels"] = np.array([])
    for i, t in enumerate(scan["slice_delay"]):

        wl_min = np.min(scan["wavelength"])
        wl_max = np.max(scan["wavelength"])
        t_min = scan["slice_delay"][i] - scan["slice_delay_width"][i]
        t_max = scan["slice_delay"][i] + scan["slice_delay_width"][i]
        _, _, _, _, scan["delay_slices"][i, :] = tools.set_roi_2d(scan["delay_unique"], scan["wavelength"], scan["data"],
                                                                  t_min, t_max, wl_min, wl_max)
        scan["delay_labels"] = scan["delay_labels"].append(str(int(scan["slice_delay"][i])))

else:
    scan["delay_slices"] = np.zeros((t_slices, 1))
    scan["delay_slices"][0, :] = np.sum(scan["data"], axis=1)
    scan["delay_labels"] = "integral"


# %%

f = plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(2, 2,
                       width_ratios=[3, 1],
                       height_ratios=[1, 2],
                       wspace=0.0, hspace=0.0)

ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])
ax3 = plt.subplot(gs[2])
ax4 = plt.subplot(gs[3])


## Plot 1) Top Left: Horizontal Profile #####################

for i, intensity in enumerate(scan["wl_slices"]):
    ax1.plot(scan["delay_unique"], intensity, lw=1,  label=str(scan["wl_labels"][i]) + r'$\,$nm')
    if not(scan["wl_labels"][i] == "integral"):
        ax3.axhline(y=scan["wl_slices"][i], lw=1, ls='--')
    ax1.axvline(x=0, ls="--", lw=0.5, color=colors.grey_3)
    ax1.axhline(y=0, ls="--", lw=0.5, color=colors.grey_3)


ax1.set_xlabel(r' ')
ax1.set_ylabel('rel. change ')

# ax1.set_ylim(signalStrengthSliceMin, signalStrengthSliceMax)
# ax1.set_xlim(tmin, tmax)

ax1.legend(loc=1, bbox_to_anchor=(1.175, 1.05))
ax1.axhline(y=0, lw=1, ls='--', color='grey')
ax1.axvline(x=0, lw=1, ls='--', color='grey')

ax1.xaxis.tick_top()
ax1.xaxis.set_label_position("top")

# ax1.set_title(dataLabel + ' | ' + str(i) + ' | ' + FileNameDate + ' | ' + FileNameTime + ' | ' +
#              str(T) + r'$\mathrm{K}$ ' + ' | ' + str(F) + r'$\mathrm{mJ/cm^2}$ ', fontsize=fontSize+2)


## Plot 3) Bottom Left: Colormap of the Profile#############

if params["symmetric_colormap"]:
    c_min = -1*np.max(np.abs(scan["data_averaged"]))
    c_max = 1*np.max(np.abs(scan["data_averaged"]))
else:
    c_min = np.min(scan["data_averaged"])
    c_max = np.max(scan["data_averaged"])

X, Y = np.meshgrid(scan["delay_unique"], scan["wavelength"])
plotted = ax3.pcolormesh(X, Y, 100*scan["data"], cmap=colors.fireice(), vmin=100*c_min, vmax=100*c_max)


# ax3.set_yticks(np.arange(wlMin, wlMax+wlStep, wlStep))
# ax3.axis([tmin, tmax, wlMin, wlMax])

ax3.set_xlabel(r'delay $t$ (ps)')
ax3.set_ylabel(r'wavelength $\mathrm{\lambda}$ (nm)')

cbaxes3 = f.add_axes([0.47, 0.65, 0.22, 0.015])
cbar = plt.colorbar(plotted, cax=cbaxes3, orientation='horizontal')

cbar.ax.tick_params('both', length=5, width=0.5, which='major', direction='in')
cl = plt.getp(cbar.ax, 'xmajorticklabels')
cbar.set_label(r' rel. change ', rotation=0)
cbaxes3.xaxis.set_ticks_position("top")
cbaxes3.xaxis.set_label_position("top")

ax2.axis('off')

## Plot 4) Bottom Right Slice at t = 0 #####################


for i, intensity in enumerate(scan["delay_slices"]):
    ax4.plot(intensity, scan["wavelength"],  lw=1,  label=str(scan["delay_labels"][i]) + r'$\,$ps')
    if not(scan["delay_labels"][i] == "integral"):
        ax3.avhline(y=scan["delay_slices"][i], lw=1, ls='--')
    ax4.axvline(x=0, ls="--", lw=0.5, color=colors.grey_3)
    ax4.axhline(y=0, ls="--", lw=0.5, color=colors.grey_3)


#ax4.semilogy(Data0/np.max(Data0),wave,'-',color = 'grey',lw = 2,label = 'spectrum')
ax4.set_ylabel(r'$\mathrm{\lambda}$ (nm)')
ax4.set_xlabel('I (a.u.)')

ax4.yaxis.tick_right()
ax4.yaxis.set_label_position("right")

ax4.legend(loc=1,  bbox_to_anchor=(1.05, 1.5))
ax4.axvline(x=0, lw=1, ls='--', color='grey')


#plt.savefig("plot_overview\\" + str(scan["id"]+".png"))

# opp.save_scan(scan)

# opp.plot_overview(scan)
# plt.show()

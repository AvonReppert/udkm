import numpy as np
import matplotlib.pyplot as plt
import udkm.opp.functions as opp
import udkm.tools.functions as tools
import udkm.tools.colors as colors

import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Rectangle


parameter_file_name = "parameters/parameters_example_3.txt"

line = 0  # welchen Datensatz man auswerten möchte

params = opp.get_scan_parameter(parameter_file_name, line)

params["data_directory"] = "data\\"
params["probe_method"] = "transmission"

params["bool_force_reload"] = True


params["t0"] = -2675.5
params["exclude_loops"] = []
params["symmetric_colormap"] = True
params["signal_level"] = 0.03


# parameters for overview plot
params["slice_wl"] = [500, 600,  700]
params["slice_wl_width"] = [10, 10, 10]
params["slice_delay"] = [2]
params["slice_delay_width"] = [0.5]


# values for dispersion correction
params["method"] = "max"
params["range_wl"] = [450, 740]
params["degree"] = 5
params["file"] = False


scan = opp.load_data(params)
plt.savefig("plot_standard\\" + scan["id"]+".png")


opp.plot_overview(scan)
plt.show()


scan = opp.frog_fit(scan)
plt.savefig("plot_fitfunction\\" + scan["id"]+".png")
plt.show()

scan = opp.frog_corr(scan)
plt.savefig("plot_frogcorr\\" + scan["id"]+".png")
plt.show()

# %%
opp.plot_overview(scan, data_key="frog_data")


opp.save_scan(scan)

# opp.plot_overview(scan)
#plt.savefig("plot_overview\\" + scan["id"]+".png")

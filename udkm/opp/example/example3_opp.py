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

params["delay_min"] = -1
params["delay_max"] = 5.5

params["wl_min"] = 1100
params["wl_max"] = 2500

# values for frog_fit
params["method"] = "max"
params["range_wl"] = [450, 740]
params["degree"] = 5
params["file"] = None


scan = opp.load_data(params)
plt.savefig("plot_standard\\" + scan["id"]+".png")
# opp.save_scan(scan)

opp.frog_fit(scan)
plt.savefig("plot_fitfunction\\" + scan["id"]+".png")

opp.frog_corr(scan)
plt.savefig("plot_frogcorr\\" + scan["id"]+".png")

# opp.plot_overview(scan)
#plt.savefig("plot_overview\\" + scan["id"]+".png")

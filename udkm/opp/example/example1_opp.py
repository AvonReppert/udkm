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

params["slice_wl"] = [1240, 2290]
params["slice_wl_width"] = [100, 100]

params["slice_delay"] = [1]
params["slice_delay_width"] = [0.5]


params["t0"] = -2008.75
params["exclude_loops"] = []
params["symmetric_colormap"] = False

params["delay_min"] = -1
params["delay_max"] = 4

params["wl_min"] = 1100
params["wl_max"] = 2500


scan = opp.load_data(params)
opp.save_scan(scan)

opp.plot_overview(scan)
plt.savefig("plot_overview\\" + scan["id"]+".png")

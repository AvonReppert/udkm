import numpy as np
import matplotlib.pyplot as plt
import udkm.opp.functions as opp
import udkm.tools.functions as tools
import udkm.tools.colors as colors

import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Rectangle


parameter_file_name = "parameters/parameters_example_3.txt"

line = 1  # welchen Datensatz man auswerten möchte

params = opp.get_scan_parameter(parameter_file_name, line)

params["data_directory"] = "data\\"
params["probe_method"] = "transmission"

params["bool_force_reload"] = True


params["t0"] = -2174.0
params["exclude_loops"] = []
params["symmetric_colormap"] = True
params["signal_level"] = 0.03

params["wl_min"] = 430
params["wl_max"] = 740


# parameters for overview plot
params["slice_wl"] = [500, 600,  700]
params["slice_wl_width"] = [10, 10, 10]
params["slice_delay"] = [2]
params["slice_delay_width"] = [0.5]


# values for dispersion correction
params["method"] = "max"
params["range_wl"] = [450, 740]
params["degree"] = 5


scan = opp.load_data(params)
plt.savefig("plot_standard\\" + scan["id"]+".png")


scan = opp.dispersion_fit(scan)
plt.savefig("plot_fitfunction\\" + scan["id"]+".png")
plt.show()

scan = opp.dispersion_corr(scan)
plt.savefig("plot_dispersioncorr\\" + scan["id"]+".png")
plt.show()

opp.plot_overview(scan, data_key="dispersion_data")
plt.show()

opp.save_scan(scan)


# Example usage:
delays = [1.0, 2.0, 3.0]
wavelengths = [500, 600, 700]
data = [
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6],
    [0.7, 0.8, 0.9]
]

# This section exports the data to the time explicit format used for import into glotaran
opp.save_time_explicit_format("HfN_15nm_Sapphire", scan["delay_unique"], scan["wavelength"],
                              scan["data"], scan["title_text"], scan["id"])

opp.save_time_explicit_format("HfN_15nm_Sapphire_dispersion", scan["delay_unique"], scan["wavelength"],
                              scan["dispersion_data"], scan["title_text"], scan["id"])

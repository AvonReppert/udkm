import numpy as np
import matplotlib.pyplot as plt
import udkm.opp.functions as opp
import h5py as h5

parameter_file_name = "parameters/parameters_example_2.txt"


line = 1  # choose dataset

params = opp.get_scan_parameter(parameter_file_name, line)

params["data_directory"] = "data\\"
params["probe_method"] = "transmission"

params["bool_force_reload"] = False

params["slice_wl"] = [475, 525, 600,  725]
params["slice_wl_width"] = [10, 10,  10, 10]

params["slice_delay"] = [5, 25]
params["slice_delay_width"] = [1, 1]

params["signal_level"] = 0.07
params["exclude_loops"] = []
params["symmetric_colormap"] = True

params["delay_min"] = -3
params["delay_max"] = 100

#params["wl_min"] = 1100
#params["wl_max"] = 2500

tst = 2
scan = opp.load_data(params)


opp.plot_overview(scan)
plt.savefig("plot_overview\\" + scan["id"]+".png")

opp.save_scan(scan)
scan2 = opp.load_scan(params["date"], params["time"], params["scan_directory"])

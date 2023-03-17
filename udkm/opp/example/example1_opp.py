
import udkm.opp.functions as opp
import matplotlib.pyplot as plt

parameter_file_name = "parameters/parameters_example_1.txt"

line = 0  # welchen Datensatz man auswerten möchte

params = opp.get_scan_parameter(parameter_file_name, line)

params["data_directory"] = "data\\"
params["probe_method"] = "transmission"

params["slice_wl"] = [1200, 2300]
params["slice_wl_width"] = [200, 200]
params["t0"] = -2008
params["exclude_loops"] = []
params["symmetric_colormap"] = False

scan = opp.load_data(params)
plt.savefig("plot_overview\\" + str(scan["id"]+".png"))

opp.save_scan(scan)

opp.plot_overview(scan)
plt.show()

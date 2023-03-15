
import udkm.opp.functions as opp
import matplotlib.pyplot as plt

parameter_file_name = "parameters/parameters_example_1.txt"

line = 0  # welchen Datensatz man auswerten möchte

params = opp.get_scan_parameter(parameter_file_name, line)
params["suffix"] = "_image"
params["plot_logarithmic"] = False
params["data_directory"] = "data\\"

scan = opp.load_data(params)

# beam.plot_overview(scan)
# plt.show()

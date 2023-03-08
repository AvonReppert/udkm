
import udkm.beamprofile.functions as beam
import matplotlib.pyplot as plt

parameter_file_name = "parameters/parameters_example_1.txt"

line = 0

params = beam.get_scan_parameter(parameter_file_name, line)
params["suffix"] = "_image"
params["plot_logarithmic"] = False
params["data_directory"] = "data\\"

params["x_min"] = 2200
params["x_max"] = 4000

params["y_min"] = 2100
params["y_max"] = 3000
params["good"] = True
scan = beam.load_data(params)

beam.plot_overview(scan)
plt.show()

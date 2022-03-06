import udkm.moke.functions as moke
import matplotlib.pyplot as plt
parameter_file_name = "parameters/parameters_example_1.txt"

line = 0

params = moke.get_scan_parameter(parameter_file_name, line)
params["bool_t0_shift"] = True
params["t0_column_name"] = "moke"
params["t_max"] = 1000
params["scan_path"] = "scan_export//"


scan = moke.load_overview_data(params)

moke.plot_overview(scan)
plt.show()
# %%
# Save scan dictionary into a pickle file.
moke.save_scan(scan)
# Load a scan from a pickle file
scan1 = moke.load_scan("20211119", "092027", "scan_export/")
moke.plot_overview(scan1, t_max=2000)

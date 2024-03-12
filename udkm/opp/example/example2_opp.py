import numpy as np
import matplotlib.pyplot as plt
import udkm.opp.functions as opp
import h5py as h5

parameter_file_name = "parameters/parameters_example_2.txt"


line = 0  # choose dataset

params = opp.get_scan_parameter(parameter_file_name, line)

params["data_directory"] = "data\\"
params["probe_method"] = "transmission"

params["bool_force_reload"] = True

params["slice_wl"] = [475, 525, 600,  725]
params["slice_wl_width"] = [10, 10,  10, 10]

params["slice_delay"] = [5, 25]
params["slice_delay_width"] = [1, 1]

params["signal_level"] = 0.07
params["exclude_loops"] = []
params["symmetric_colormap"] = True

params["wl_min"] = 450
params["wl_max"] = 800

#params["delay_min"] = -3
#params["delay_max"] = 100

#params["wl_min"] = 1100
#params["wl_max"] = 2500


# values for dispersion correction
params["method"] = "absMax"
params["range_wl"] = [[455, 480], [520, 550], [675, 745]]
params["degree"] = 3
params["file"] = False


scan = opp.load_data(params)
plt.savefig("plot_standard\\" + scan["id"]+".png")

opp.plot_overview(scan)
plt.savefig("plot_overview\\" + scan["id"]+".png")


scan = opp.dispersion_fit(scan)
plt.savefig("plot_fitfunction\\" + scan["id"]+".png")
plt.show()

scan = opp.dispersion_corr(scan)
plt.savefig("plot_dispersioncorr\\" + scan["id"]+".png")
plt.show()
opp.save_scan(scan)
# %%
opp.plot_overview(scan, data_key="dispersion_data")


# This section exports the data to the time explicit format used for import into glotaran
opp.save_time_explicit_format("AuNr_raw", scan["delay_unique"], scan["wavelength"],
                              scan["data"], scan["title_text"], scan["id"])

opp.save_time_explicit_format("AuNr_dispersion_corr", scan["delay_unique"], scan["wavelength"],
                              scan["dispersion_data"], scan["title_text"], scan["id"])

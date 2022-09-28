
import udkm.tools.functions as tools
import os


for root, _, files in os.walk("data"):
    for file in files:
        filename = os.path.join(root, file)
        print(filename)

        decimals = 0
        tools.reduce_file_size(filename, decimals, filename_suffix='')

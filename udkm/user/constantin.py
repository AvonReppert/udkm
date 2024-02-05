# -*- coding: utf-8 -*-
teststring = "Successfully loaded udkm.user.constantin"
####
import pandas as pd

####################
# Export functions #
####################
def export_field(series,name='data_field.dat'):
    export = pd.DataFrame(series['delay_list'][0], columns=['t'])
    for i,field in enumerate(series["fields"]):
        export[field] = series["signal_list"][i]
    export.to_csv(name,index=False)

def export_fluence(series,name='data_fluence.dat'):
    export = pd.DataFrame(series['delay_list'][0], columns=['t'])
    for i,fluence in enumerate(series["fluence"]):
        export[fluence] = series["signal_list"][i]
    export.to_csv(name,index=False)
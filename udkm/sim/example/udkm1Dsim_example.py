# -*- coding: utf-8 -*-
"""
Created on Fryday Jul 30 15:05:40 2021

@author: Max
"""
import numpy as np
import matplotlib.pyplot as plt
import udkm1Dsim as ud
u = ud.u
u.setup_matplotlib()

import udkm.sim.simhelpers as shel

import udkm.sim.materials.FeRh_001_1TM as FeRh_001_1TM
FeRh = FeRh_001_1TM.FeRh_001_1TM()

import udkm.sim.materials.MgO_001_1TM as MgO_001_1TM
MgO = MgO_001_1TM.MgO_001_1TM()

import udkm.sim.materials.W_001_1TM as W_001_1TM
W = W_001_1TM.W_001_1TM()

import udkm.sim.materials.Pt_111_1TM as Pt_111_1TM
Pt = Pt_111_1TM.Pt_111_1TM()

#%%

''' Put In the Simulation Parameter '''

#%%
''' Simulation Parameter '''
# Initialization of the sample
sample_name = 'FeRhP16c'
layers      = ['Pt','FeRh','W','MgO']
sample_dict = {'Pt':Pt,'FeRh':FeRh,'W':W,'MgO':MgO}
properties  = {'Pt':{'C' :Pt.prop['heat_capacity']},
               'FeRh':{'C' :FeRh.prop['heat_capacity']},
               'W':{'C' :W.prop['heat_capacity']},
               'MgO':{'C':MgO.prop['heat_capacity']}}

#Possible Excitation Conditions of Simulation
fluence_list = [1.4,2]
angle_list   = [39.7*u.deg]
peak_list    = ['FeRh']

#Simulated Excitation Conditions
peak_meas    = 'FeRh'
fluence_sim  = [fluence_list[0]]*u.mJ/u.cm**2
puls_width   = [0]*u.ps
pump_delay   = [0]*u.ps
multi_abs    = False
init_temp    = 0.1
heat_diff    = True
delays       = np.r_[-2:60:0.1]*u.ps

#Simulation Mode
static_exp      = False
el_pho_coupling = True

#Simulation Parameters
num_unit_cell    = [8,146,28,800]
heat_cond_fac    = [1,1,1,1]
el_pho_coup_list = [0.2*u.ps,0.6*u.ps,0.6*u.ps,0*u.ps]
el_grun_list     = [0.35,0.4,0.4,1]

#Analysis and Export
calc_energy_map = False
calc_stress_map = False
export_maps     = False
sim_name        = r'test'

export_name = peak_meas + 'peak_T' + str(init_temp) + 'K_F' + str(int(10*fluence_sim[0].magnitude)) + 'mJ_' + sim_name

plotting_sim    = [False,True,True,False]
color_list      = ['red','black','gray','blue']
data_list       = [0,0,0,0]

#%%

''' Build the sample structure from the initialized unit cells '''
#%% 

for l in range(len(layers)):
    prop_uni_cell = {}
    prop_uni_cell['a_axis']         = sample_dict[layers[l]].prop['a_axis']
    prop_uni_cell['b_axis']         = sample_dict[layers[l]].prop['b_axis']
    prop_uni_cell['sound_vel']      = sample_dict[layers[l]].prop['sound_vel']
    prop_uni_cell['lin_therm_exp']  = sample_dict[layers[l]].prop['lin_therm_exp']
    prop_uni_cell['heat_capacity']  = sample_dict[layers[l]].prop['heat_capacity']
    prop_uni_cell['therm_cond']     = heat_cond_fac[l]*sample_dict[layers[l]].prop['therm_cond']
    prop_uni_cell['opt_pen_depth']  = sample_dict[layers[l]].prop['opt_pen_depth']
    prop_uni_cell['opt_ref_index']  = sample_dict[layers[l]].prop['opt_ref_index']
    properties[layers[l]]['unit_cell'] = sample_dict[layers[l]].createUnitCell(layers[l],sample_dict[layers[l]].prop['c_axis'],prop_uni_cell)

S = ud.Structure(sample_name)
for l in range(len(layers)):
    S.add_sub_structure(properties[layers[l]]['unit_cell'],num_unit_cell[l])
S.visualize()
print(S)

_, _, distances = S.get_distances_of_layers()  

######### Get additional properties of the layers #########
for l in range(len(layers)):
    properties[layers[l]]['num_unit_cell'] = num_unit_cell[l]
    properties[layers[l]]['density']       = properties[layers[l]]['unit_cell'].density
    properties[layers[l]]['select_layer']  = S.get_all_positions_per_unique_layer()[layers[l]]
    properties[layers[l]]['thick_layer']   = properties[layers[l]]['num_unit_cell']*properties[layers[l]]['unit_cell'].c_axis
    properties[layers[l]]['C_layer']       = (properties[layers[l]]['C']*properties[layers[l]]['density']*properties[layers[l]]['thick_layer']).to('J/m**2/K').magnitude
    properties[layers[l]]['C_unit_cell']   = (properties[layers[l]]['C']*properties[layers[l]]['density']*properties[layers[l]]['unit_cell'].c_axis).to('J/m**2/K').magnitude

#%%

''' Determine the optical absorption from the excitation conditions '''

#%% Get the absorption
   
h = ud.Heat(S, True)
h.excitation = {'fluence':fluence_sim,'delay_pump':pump_delay,'pulse_width': puls_width,'multilayer_absorption':multi_abs,'wavelength':800*u.nm,'theta':angle_list[peak_list.index(peak_meas)]}

dAdzLB        = h.get_Lambert_Beer_absorption_profile()
dAdz, _, _, _ = h.get_multilayers_absorption_profile()
   
######### Plot the absorption profile #########
  
plt.figure()
plt.plot(distances.to('nm'), dAdz*1e-9*1e2, label='multilayer')
plt.plot(distances.to('nm'), dAdzLB*1e-9*1e2, label='Lambert-Beer')
plt.xlim(0,120)
plt.legend()
plt.xlabel('Distance (nm)')
plt.ylabel('Differnetial Absorption (%)')
plt.title('Laser Absorption Profile')
plt.show()

#%%

''' Get Temperature Map from the absorption profile including heat diffusion '''

#%% 

######### Calculate temperature map potentially including heat diffusion #########
h.save_data              = False
h.disp_messages          = True
h.heat_diffusion         = heat_diff
h.boundary_conditions    = {'top_type': 'isolator', 'bottom_type': 'isolator'}
temp_map, delta_temp_map = h.get_temp_map(delays, init_temp)

######### Artificially include el-pho coupling for stress rise time #########
if el_pho_coupling:
    temp_map = shel.IncludeElPhoCoupling(properties,layers,el_pho_coup_list,el_grun_list,temp_map,distances,delays,init_temp)

#%%
######### Plot generated temperature map #########
plt.figure(figsize=[6, 5])
plt.subplot(1, 1, 1)
plt.pcolormesh(distances.to('nm').magnitude,delays.to('ps').magnitude,temp_map,shading='auto',cmap='RdBu_r',vmin=np.min(temp_map), vmax=np.max(temp_map))
plt.colorbar()
plt.xlim(0,80)
plt.xlabel('Distance (nm)')
plt.ylabel('Delay (ps)')
plt.title('Temperature Map')
plt.tight_layout()
plt.show()

########## Plot the transient mean temperature ##########
mean_temp_list = []
for l in range(len(layers)):
    mean_temp_list.append(np.mean(temp_map[:, properties[layers[l]]['select_layer']], 1))
shel.ExportTransients('TransientTemp',mean_temp_list,delays,export_name)

plt.figure(figsize=[6, 0.68*6])
plt.subplot(1, 1, 1)
for l in range(len(layers)):
    if plotting_sim[l]:
        plt.plot(delays.to('ps'), mean_temp_list[l], label=layers[l],ls = '-',color = color_list[l])
plt.ylabel('Temperature (K)')
plt.xlabel('Delay (ps)')
plt.legend()
plt.tight_layout()
plt.show()

if calc_stress_map:
    stress_map  = shel.CalcStressFromTempMap(sample_dict,properties,layers,temp_map)

    plt.figure(figsize=[6, 5])
    plt.subplot(1, 1, 1)
    plt.pcolormesh(distances.to('nm').magnitude,delays.to('ps').magnitude,stress_map,shading='auto',cmap='RdBu_r',vmin=np.min(stress_map), vmax=np.max(stress_map))
    plt.colorbar()
    plt.xlim(0,80)
    plt.xlabel('Distance (nm)')
    plt.ylabel('Delay (ps)')
    plt.title('Stress Map')
    plt.tight_layout()
    plt.show()

#%%

''' Get the picosecond strain dynamics from the Temperature Map '''
    
#%% 

########## Calculate the strain map from the temp map ##########
pnum = ud.PhononNum(S, True,only_heat=static_exp)
pnum.save_data     = False
pnum.disp_messages = True

strain_map = pnum.get_strain_map(delays, temp_map, delta_temp_map)

######### Plot the strain map #########
plt.figure(figsize=[6, 0.68*6])
plt.pcolormesh(distances.to('nm').magnitude,delays.to('ps').magnitude,1e3*strain_map,shading='auto',cmap='RdBu_r',vmin=-np.max(1e3*strain_map), vmax=np.max(1e3*strain_map))
plt.colorbar()
plt.xlim(0, 105)
plt.xlabel('Distance (nm)')
plt.ylabel('Delay (ps)')
plt.title(r'Strain Map ($10^{-3}$)')
plt.tight_layout()
plt.show()

######### Plot comparison measurement and simulation ##########
mean_strain_list = []
for l in range(len(layers)):
    mean_strain_list.append(1e3*np.mean(strain_map[:,properties[layers[l]]['select_layer']], 1))
shel.ExportTransients('TransientStrain',mean_strain_list,delays,export_name)
#%%
plt.figure(figsize=[6, 0.68*6])
plt.subplot(1, 1, 1)
for l in range(len(layers)):
    if plotting_sim[l]:
        plt.plot(delays.to('ps'), mean_strain_list[l],label=layers[l],ls = '-',color = color_list[l])
for l in range(len(layers)):
    if np.sum(data_list[l]) != 0:
        plt.plot(data_list[l][:,5], data_list[l][:,6],'o',label=layers[l],color = color_list[l])
        
plt.xlim(np.min(delays),np.max(delays))
plt.ylabel('strain ($10^{-3}$)')
plt.xlabel('Delay (ps)')
plt.legend()
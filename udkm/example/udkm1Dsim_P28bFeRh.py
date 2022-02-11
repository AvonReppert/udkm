# -*- coding: utf-8 -*-
"""
Created on Fryday Jul 30 15:05:40 2021

@author: Max
"""
import helpers
hel = helpers.helpers()
import numpy as np
import matplotlib.pyplot as plt
import lmfit as lm
import udkm1Dsim as ud
import copy
u = ud.u
u.setup_matplotlib()

import FeRh_001_1TM as FeRh_001_1TM
FeRh = FeRh_001_1TM.FeRh_001_1TM()

import MgO_001_1TM as MgO_001_1TM
MgO = MgO_001_1TM.MgO_001_1TM()


#%%
def IncludeElPhoCoupling(DictProperties,ListLayers,ListCoupling,ListElGruneisen,TempMap,Distances,Delays,InitTemp):
    Indext0   = np.argmin(abs(Delays))
    Indextmax = np.argmin(abs(Delays.to('ps').magnitude-20))
    for i in np.arange(Indext0,Indextmax,1):
        for l in range(len(ListLayers)):
            SelectLayer = DictProperties[ListLayers[l]]['select_layer']
            TempMap[i,SelectLayer] = (TempMap[i,SelectLayer]-InitTemp)*(1-ListElGruneisen[l]*np.exp(-Delays[i]/ListCoupling[l]))+InitTemp
    return TempMap

def ExportTransients(quantity,dataList,delays,fileName):
    exportData      = np.zeros((len(delays),len(dataList)+1))
    exportData[:,0] = delays.to('ps').magnitude
    for i in range(len(dataList)):
        exportData[:,i+1] = dataList[i]
    np.savetxt('SimulationResults/' + quantity + '_' + fileName + '.dat',exportData,header='delay ps \t Strain of layers (10-3)')
    
def GetEnergyDistribution(DicProp,LayerName,TempMap):
    """This function extracts from the given TempMap the time- and depth dependent
    energy distribution and the time-dependent total energy density per layer. This
    prepares the redistribution of energy density in the case of magnetic excitations
    conserving the heat transport.
    
    Parameters:
    -----------
    DictProperties: dictionary 
        the properties of the layers of the sample like: 
        'select_layer': containing unit cells; 
        'C_layer': heat capacity of the whole layer [J*nm/kg*K]
        
    LayerName: string
        short name of the layer that is part of the sample and dictionary
        
    TempMap: 2D array
        the increased temperature after excitation of all layers for each delay step
    
    Returns:
    --------       
    ProfileMatrix: 2D array
        normed temperature profiles of the layer 'LayerName' for each delay step
        
    Qlayer: 1D array
        heat density stored in the layer for each delay step [J/m**2]
    
    Example:
    --------
    The layer 'Au' contains 4 unit cells and 2 delays
    >>> np.array([[0.5,0.5,0,0],[0.33,0.33,0.33,0]]), [17,16] = GetEnergyDistribution(Properties,'Au',[[50,50,0,0],[40,40,40,0]])"""
    
    Tsum          = np.sum(TempMap[:,DicProp[LayerName]['select_layer']],1)
    Qlayer        = np.mean(TempMap[:,DicProp[LayerName]['select_layer']],1)*DicProp[LayerName]['C_layer']
    ProfileMatrix = TempMap[:,DicProp[LayerName]['select_layer']]/Tsum[:,None]
    ProfileMatrix[np.isnan(ProfileMatrix)] = 1/np.sum(DicProp[LayerName]['select_layer'])
    return ProfileMatrix, Qlayer

def GetMagneticExcitation(DicSample,DicProp,MagLayerName,Timet0,InitTemp,MagParameters,RemainSat):
    """This function calculates the energy distribution between phononic and magnetic system in
    the magnetic layer 'Layers[MagLayerNum]' following the extracted spatial profile. The properties 
    'Q_spin', 'Q_pho', 'T_spin', 'saturation' and 'T_pho' are added. The magnetic excitations are 
    modelled by an instantaneous and a slow component with the amplitudes and rise time given by
    'MagParameters[0 to 2]'. Additionally, the function includes the saturation of the magnetic 
    excitations if the maximum amounf of energy is deposited (given by 'InitTemp'and 
    'MagParameters[3]'). The local saturation affects the global 'Q_spin' and the full phase 
    transition determines the remagnetization behaviour by 'RemainSat'.
    
    Parameters:
    -----------
    DicSample: dictionary
        the material classes of the sample
    
    DicProp: dictionary 
        the properties of the layers of the sample like: 
        'Q_new': time-dependent energy density [J/m**2]; 
        'C_unit_cell': heat capacity of unit cell [J/K*m**2];
        'mag_profiles': spatial distribution magnetic excitations;
        'profiles': spatial distribution phonon excitations
        
    MagLayerName: string
        short name of the magnetic layer in the sample
        
    Timet0: 1D array
        the delays t > t0
        
    InitTemp: float
        start temperature of the simulation
        
    MagParameters: list
        parameters of the magnetic excitation: [0]:AmpFast; [1]:AmpSlow; 
        [2]: TauSlow; [3]: SatFac
        
    RemainSat: boolean
        does a full demagnetization remains [True] or is there a remagnetization [False]
    
    Example:
    --------
    The sample consists of an yttrium and a dysprosium layer where dysprosium is magnetic
    >>> GetMagneticExcitation({'Y':Y,'Dy':Dy},DicProp,'Dy',[0,1,2],100,[0.2,0.2,1*u.ps,0.7],True)"""
    #time dependent Energy in Spin system
    DicProp[MagLayerName]["Q_spin"] = DicProp[MagLayerName]["Q_new"]*(MagParameters[0]+MagParameters[1]*(1-np.exp(-1*Timet0/MagParameters[2])))
    DicProp[MagLayerName]["Q_pho"]  = DicProp[MagLayerName]["Q_new"]*(1-(MagParameters[0]+MagParameters[1]*(1-np.exp(-1*Timet0/MagParameters[2])))) 
        
    #calculate Spin Temperature profile
    DicProp[MagLayerName]["T_spin"] = -1*DicProp[MagLayerName]["mag_profiles"]*DicProp[MagLayerName]["Q_spin"][:,None]/DicProp[MagLayerName]["C_unit_cell"]
    
    #implement saturation effect 
    DicProp[MagLayerName]["T_spin_sat"] = copy.deepcopy(DicProp[MagLayerName]["T_spin"])
    SelectSaturation = DicProp[MagLayerName]["T_spin_sat"] < ReturnTSpinMax(DicSample,DicProp,MagLayerName,InitTemp,MagParameters[3])
    if RemainSat:
        for c in range(len(SelectSaturation[0,:])):
            if np.sum(SelectSaturation[:,c] > 0):
                IndexSat = np.argmax(SelectSaturation[:,c])
                DicProp[MagLayerName]["saturation"][IndexSat:,c] = True*np.ones(len(SelectSaturation[IndexSat:,0]))
    else:
        DicProp[MagLayerName]["saturation"] = SelectSaturation
    DicProp[MagLayerName]["saturation"] = DicProp[MagLayerName]["saturation"] > 0.1
    DicProp[MagLayerName]["T_spin_sat"][DicProp[MagLayerName]["saturation"]] = ReturnTSpinMax(DicSample,DicProp,MagLayerName,InitTemp,MagParameters[3])    
    
    # return the energy above the saturation to phonon system
    DeltaQSpin = np.sum(DicProp[MagLayerName]["T_spin_sat"] - DicProp[MagLayerName]["T_spin"],1)*DicProp[MagLayerName]["C_unit_cell"]
    DicProp[MagLayerName]["Q_pho"]  = DicProp[MagLayerName]["Q_pho"] + DeltaQSpin
    DicProp[MagLayerName]["Q_spin"] = DicProp[MagLayerName]["Q_spin"] - DeltaQSpin 
    DicProp[MagLayerName]["Q_new"]  = DicProp[MagLayerName]["Q_pho"] + DicProp[MagLayerName]["Q_spin"]        
    DicProp[MagLayerName]["T_spin"] = DicProp[MagLayerName]["T_spin_sat"] 
    DicProp[MagLayerName]["T_pho"]  = DicProp[MagLayerName]["profiles"]*DicProp[MagLayerName]["Q_pho"][:,None]/DicProp[MagLayerName]["C_unit_cell"]
    
def ReturnTSpinMax(DicSample,DicProp,MagLayerName,InitTemp,SaturationFac):
    """This function returns the maximum spin temperature for the given strat temperature concerning the magnetic heat capacity. 
    
    Parameters:
    -----------    
    Tstart: number
        the start temperature of the measurement
        
    saturationFactor: number
        factor to reduce the used fraction of the magnetic heat capacity
            
    Returns:
    --------
    TspinMax: number
        the maximum spin temperature calculated with heat capacity

    Example:
    --------
    The magnetic layer 'Dy' is excited at 80K
    >>> 100 = ReturnTSpinMax({'Y':Y,'Dy':Dy},DicProp,'Dy',80,0.7)"""
    DeltaQMagMax = DicSample[MagLayerName].getQMag(InitTemp)
    TspinMax     = (-1*DeltaQMagMax/DicSample[MagLayerName].prop['molar_mass']/DicProp[MagLayerName]['C']).to('K').magnitude*SaturationFac
    return TspinMax

def UpdateAllLayers(DicSample,DicProp,Layers,MagLayerNum,Timet0,InitTemp,MagParameters,RemainSat,Qtot,FractionList):
    """This function adjust the phonon energy in the sample taking into account the energy 
    deposited to the magnetic excitations in the magnetic layer. The remaining phonon energy
    is distributed to the layers accoring its energy fraction 'FractionList' without magnetic
    excitations. Based on the new energy in the magnetic layer the distribution between
    phonons and spins is calculated again. The new magnetic excitations are compared with
    the old values to iteratively minimize the difference.
    
    Parameters:
    -----------
    DicSample: dictionary
        the material classes of the sample
    
    DicProp: dictionary 
        the properties of the layers of the sample like: 
        'Q_new': time-dependent energy density [J/m**2]; 
        'C_unit_cell': heat capacity of unit cell [J/K*m**2];
        'mag_profiles': spatial distribution magnetic excitations;
        'profiles': spatial distribution phonon excitations
        
    Layers: list
        short names of the layers in the sample
    
    MagLayerNum: integer
        position of the magnetic layer in the sample
        
    Timet0: 1D array
        the delays t > t0
    
    InitTemp: float
        start temperature of the simulation
        
    MagParameters: list
        parameters of the magnetic excitation: [0]:AmpFast; [1]:AmpSlow; 
        [2]: TauSlow; [3]: SatFac
        
    RemainSat: boolean
        does a full demagnetization remains [True] or is there a remagnetization [False]
    
    Qtot: float
        total deposited energy density to the sample [J/m**2]
    FractionList: list
        time-dependent energy density fraction stored in all layers without magnetic excitations
    
    Returns:
    --------
    Deviation: float
        difference between 'Q_spin' before and after adjustment of phonons [J/m**2]

    Example:
    --------
    The sample consists of an yttrium and a dysprosium layer where dysprosium is magnetic
    >>> 0.5 = UpdateAllLayers({'Y':Y,'Dy':Dy},DicProp,['Y','Dy'],1,[0,1],100,[0.2,0.2,1*u.ps,0.7],True,20.2,[[0.3,0.2],[0.7,0.8]])"""
    
    QSpinDummy = DicProp[Layers[MagLayerNum]]["Q_spin"] 
    QtotPho    = Qtot - DicProp[Layers[MagLayerNum]]["Q_spin"]
    for l in range(len(Layers)):
        if l == MagLayerNum:
            DicProp[Layers[l]]["Q_new"] = FractionList[l]*QtotPho + DicProp[Layers[l]]["Q_spin"]
        else:
            DicProp[Layers[l]]["Q_new"] = FractionList[l]*QtotPho 
            
    GetMagneticExcitation(DicSample,DicProp,Layers[MagLayerNum],Timet0,InitTemp,MagParameters,RemainSat)
    
    Difference = DicProp[Layers[MagLayerNum]]["Q_spin"] - QSpinDummy
    Deviation  = np.sum(Difference)
    return Deviation

def UpdateTempMaps(DicSample,DicProp,Layers,MagLayerNum,Time,Distances,ListCoupling,ListElGruneisen):
    """This function generates from the 'Q_new' of all non-magnetic layers and 
    'T_pho' of the magnetic layer from 'UpdateAllLayers' a phonon map. The 'T_spin'
    of the magnetic layer determines the magnetic temperature map that is sclaed by 
    the scaling factor between magnetic and phononic Gruneisen constant. The sum of
    subsytems temperature maps serves as source for strain dynamics.
   
    Parameters:
    -----------
    DicSample: dictionary
        the material classes of the sample
    
    DicProp: dictionary 
        the properties of the layers of the sample like: 
        'Q_new': time-dependent energy density [J/m**2]; 
        'C_unit_cell': heat capacity of unit cell [J/K*m**2];
        'T_pho': phonon temperature per layer [K];
        'T_spin': spin temperature of magnetic layer [K]
        
    Layers: list
        short names of the layers in the sample
    
    MagLayerNum: integer
        position of the magnetic layer in the sample
        
    Time: 1D array
        simulated delays including delays before excitation 
    
    Distances: 1D array
        simulated depth axis of the sample
    
    ListCoupling: list
        phonon stress rise time of all layers
        
    ListElGruneisen: list
        1- fraction of instantaneous stress
    
    Returns:
    --------
    TempMapPhonon: 2D array
        the phonon temperature of all layers with shape 'Time' and 'Distances'
    
    TempMapSpin: 2D array
        the spin temperature (scaled by Gruneisen) of all layers with shape 'Time' and 'Distances'
            
    TempMapTotal: 2D array
        the "total" temperature of all layers with shape 'Time' and 'Distances'      
    
    Example:
    --------
    The sample consists of an yttrium and a dysprosium layer where dysprosium is magnetic
    >>> Pho, Spin, Total = UpdateTempMaps({'Y':Y,'Dy':Dy},DicProp,['Y','Dy'],1,[-1,0,1],[0.5,1,1.6,2.2],[1*u.ps,2*u.ps],[0.6,0.5])"""
    #Determine the temperature maps
    for l in range(len(Layers)):
        if l != MagLayerNum:
            DicProp[Layers[l]]["T_pho"]  = DicProp[Layers[l]]["profiles"]*DicProp[Layers[l]]["Q_new"][:,None]/DicProp[Layers[l]]["C_unit_cell"]
      
    #Initialize the arrays to export the temperature Maps   
    TempMapPhonon = np.zeros((np.size(Time),np.size(Distances)))
    TempMapSpin   = np.zeros((np.size(Time),np.size(Distances)))
    TempMapTotal  = np.zeros((np.size(Time),np.size(Distances)))
    iStart        = np.argmin(abs(Time))
    
    for l in range(len(Layers)):
        TempMapPhonon[iStart:,DicProp[Layers[l]]["select_layer"]]  = DicProp[Layers[l]]["T_pho"]  
    TempMapPhonon = IncludeElPhoCoupling(DicProp,Layers,ListCoupling,ListElGruneisen,TempMapPhonon,Distances,Time[iStart:],0)

    MagGrunFac = DicSample[Layers[MagLayerNum]].prop['Grun_mag_c_axis']/DicSample[Layers[MagLayerNum]].prop['Grun_c_axis']
    TempMapSpin[iStart:,DicProp[Layers[MagLayerNum]]["select_layer"]] = MagGrunFac*DicProp[Layers[MagLayerNum]]["T_spin"]
    
    TempMapTotal = TempMapSpin + TempMapPhonon
    return TempMapPhonon, TempMapSpin, TempMapTotal  

def CalcStressFromTempMap(DicSample,DicProp,Layers,TempMap):
    StressMap = np.zeros((len(TempMap[:,0]),len(TempMap[0,:])))
    for l in range(len(Layers)):
        LinExp   = DicSample[Layers[l]].prop['lin_therm_exp']
        ElastC33 = DicSample[Layers[l]].prop['elastic_c33']
        StressMap[:,DicProp[Layers[l]]['select_layer']] = TempMap[:,DicProp[Layers[l]]['select_layer']]*LinExp*ElastC33
    return StressMap

def PlotMagEnergyDistribution(DicProp,Layers,MagLayerNum,Timet0,ColorList):
    Qtot = np.zeros(len(Timet0))
    plt.figure(figsize=[6, 0.68*6])
    plt.subplot(1, 1, 1)
    for l in range(len(Layers)):
        if l == MagLayerNum:
            plt.plot(Timet0,DicProp[Layers[l]]['Q_spin'],lw = 2,ls = '-.',color=ColorList[l],label=Layers[l] + ' spin')
            plt.plot(Timet0,DicProp[Layers[l]]['Q_pho'],lw = 2,ls = '--',color = ColorList[l],label = Layers[l] + ' pho')
            plt.plot(Timet0,DicProp[Layers[l]]['Q_pho']+DicProp[Layers[l]]['Q_pho'],lw = 2,ls = '-',color=ColorList[l],label = Layers[l] + ' tot')
            Qtot = Qtot+DicProp[Layers[l]]['Q_new']
        else:
            plt.plot(Timet0,DicProp[Layers[l]]['Q_new'],lw = 2,ls = '-',color=ColorList[l],label = Layers[l])
            Qtot = Qtot+DicProp[Layers[l]]['Q_new']
    plt.plot(Timet0,Qtot,lw = 2,ls = '-',color = "black" ,label = r"$\mathrm{Q_{tot}}$")
    plt.axhline(y= 0 ,ls = '--',color = "gray",lw = 1)
    plt.axvline(x= 0 ,ls = '--',color = "gray",lw = 1)
   
    plt.ylabel(r'$\mathrm{Q}$' + r' $\mathrm{\left(\frac{J}{m^2}\right)}$')
    plt.xlabel(r'$\mathrm{t}$' + r' $\mathrm{(ps)}$')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def PlotMap(Title,Map,ZeroCrossing):
    plt.figure(figsize=[6, 5])
    plt.subplot(1, 1, 1)
    if ZeroCrossing:
        plt.pcolormesh(distances.to('nm').magnitude,delays.to('ps').magnitude,Map,shading='auto',cmap='RdBu_r',vmin=-np.max(abs(Map)), vmax=np.max(abs(Map)))
    else:
        plt.pcolormesh(distances.to('nm').magnitude,delays.to('ps').magnitude,Map,shading='auto',cmap='viridis',vmin=np.min(Map), vmax=np.max(Map))
    plt.colorbar()
    plt.xlim(0,80)
    plt.xlabel('Distance (nm)')
    plt.ylabel('Delay (ps)')
    plt.title(Title)
    plt.tight_layout()
    plt.show()

#%%

''' Put In the Simulation Parameter '''

#%%
''' Simulation Parameter '''
# Initialization of the sample
sample_name = 'FeRhP28b'
layers      = ['FeRh','MgO']
sample_dic  = {'FeRh':FeRh,'MgO':MgO}
properties  = {'FeRh':{'C' :FeRh.prop['heat_capacity']},
               'MgO':{'C':MgO.prop['heat_capacity']}}

#Possible Excitation Conditions of Simulation
fluence_list = [5.5,4.6,1.35] #5.5=2mJ; 4.6=1.6mJ; 1.35=0.5mJ
angle_list   = [39.7*u.deg]
peak_list    = ['FeRh']

#Simulated Excitation Conditions
peak_meas    = 'FeRh'
fluenz_sim   = [1.1]*u.mJ/u.cm**2
puls_width   = [0]*u.ps
pump_delay   = [0]*u.ps
multi_abs    = True
init_temp    = 0.1
heat_diff    = True
delays       = np.r_[-2:50:0.1]*u.ps

#Simulation Mode
static_exp      = False
el_pho_coupling = True

#Simulation Parameters
num_unit_cell    = [44,700]
heat_cond_fac    = [1,1]
el_pho_coup_list = [0.6*u.ps,0*u.ps]
el_grun_list     = [0.4,1]

#Analysis and Export
calc_energy_map = False
calc_stress_map = False
export_maps     = False
sim_name        = r'subthreshold'

export_name = peak_meas + 'peak_T' + str(init_temp) + 'K_F' + str(int(10*fluenz_sim[0].magnitude)) + 'mJ_' + sim_name

plotting_sim    = [True,False]
color_list      = ['red','black','gray','blue','blue']
data_list       = [np.genfromtxt('Measurements/Strain05mJ.dat'),0]

#%%

''' Build the sample structure from the initialized unit cells '''
#%% 

for l in range(len(layers)):
    prop_uni_cell = {}
    prop_uni_cell['a_axis']         = sample_dic[layers[l]].prop['a_axis']
    prop_uni_cell['b_axis']         = sample_dic[layers[l]].prop['b_axis']
    prop_uni_cell['sound_vel']      = sample_dic[layers[l]].prop['sound_vel']
    prop_uni_cell['lin_therm_exp']  = sample_dic[layers[l]].prop['lin_therm_exp']
    prop_uni_cell['heat_capacity']  = sample_dic[layers[l]].prop['heat_capacity']
    prop_uni_cell['therm_cond']     = heat_cond_fac[l]*sample_dic[layers[l]].prop['therm_cond']
    prop_uni_cell['opt_pen_depth']  = sample_dic[layers[l]].prop['opt_pen_depth']
    prop_uni_cell['opt_ref_index']  = sample_dic[layers[l]].prop['opt_ref_index']
    properties[layers[l]]['unit_cell'] = sample_dic[layers[l]].createUnitCell(layers[l],sample_dic[layers[l]].prop['c_axis'],prop_uni_cell)

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
h.excitation = {'fluence':fluenz_sim,'delay_pump':pump_delay,'pulse_width': puls_width,'multilayer_absorption':multi_abs,'wavelength':800*u.nm,'theta':angle_list[peak_list.index(peak_meas)]}

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
    temp_map = IncludeElPhoCoupling(properties,layers,el_pho_coup_list,el_grun_list,temp_map,distances,delays,init_temp)

phase_transition = False
temp_map_dummy   = temp_map
tau = 7.8*u.ps
if phase_transition:
    index_t0 = np.argmin(abs(delays.to('ps').magnitude))
    select_FeRh = distances < 13.2*u.nm
    for i in range(len(delays)-index_t0):
        for j in range(len(distances[select_FeRh])):
            if temp_map[index_t0+i,j] > 50:
                temp_map_dummy[index_t0+i,j] = (temp_map[index_t0+i,j]-50)*(0.62+0.38*np.exp(-delays[index_t0+i]/tau))+50-10*(1-np.exp(-delays[index_t0+i]/tau))#+206*(1-np.exp(-delays[index_t0+i]/tau))
                #temp_map_dummy[index_t0+i,j] = (temp_map[index_t0+i,j]-50)*(0.62+0.38*np.exp(-delays[index_t0+i]/tau))+50-10*(1-np.exp(-delays[index_t0+i]/tau))#+280*(1-np.exp(-delays[index_t0+i]/tau))
            else:
                temp_map_dummy = temp_map

    
temp_map =  temp_map_dummy      
######### Re-calculate delte_temp_map ########
for i in range(len(delays)):
    if i >=1:
        delta_temp_map[i,:] = temp_map[i,:] - temp_map[i-1,:] 
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
ExportTransients('TransientTemp',mean_temp_list,delays,export_name)

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
    stress_map  = CalcStressFromTempMap(sample_dic,properties,layers,temp_map)

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
ExportTransients('TransientStrain',mean_strain_list,delays,export_name)

plt.figure(figsize=[6, 0.68*6])
plt.subplot(1, 1, 1)
for l in range(len(layers)):
    if plotting_sim[l]:
        plt.plot(delays.to('ps'), mean_strain_list[l],label=layers[l],ls = '-',color = color_list[l])
for l in range(len(layers)):
    if np.sum(data_list[l]) != 0:
        plt.plot(data_list[l][:,0], data_list[l][:,2],'o',label=layers[l],color = color_list[l])
        
plt.xlim(np.min(delays),np.max(delays))
plt.ylabel('strain ($10^{-3}$)')
plt.xlabel('Delay (ps)')
plt.legend()

#%%

''' Run Dynamical X-ray Simulation '''

#%%

dyn = ud.XrayDyn(S, True)
dyn.disp_messages = True
dyn.save_data = False

dyn.energy = np.r_[8047]*u.eV  # set two photon energies
dyn.qz = np.r_[4.05:4.35:0.001]/u.angstrom  # qz range

strain_vectors = pnum.get_reduced_strains_per_unique_layer(strain_map,N=1000)
R_seq = dyn.inhomogeneous_reflectivity(strain_map, strain_vectors, calc_type='sequential')

FWHM = 0.04/1e-10  # Angstrom
sigma = FWHM/2.3548

handle = lambda x: np.exp(-((x)/sigma)**2/2)

R_seq_conv = np.zeros_like(R_seq)
for i, delay in enumerate(delays):
    R_seq_conv[i, 0, :] = dyn.conv_with_function(R_seq[i, 0, :], dyn._qz[0, :], handle)


#%%
plt.figure(figsize=[6, 8])
plt.subplot(2, 1, 1)
plt.semilogy(dyn.qz[0, :].to('1/nm'), R_seq_conv[0, 0, :], label=np.round(delays[0]))
plt.semilogy(dyn.qz[0, :].to('1/nm'), R_seq_conv[100, 0, :], label=np.round(delays[10]))
plt.semilogy(dyn.qz[0, :].to('1/nm'), R_seq_conv[-1, 0, :], label=np.round(delays[-1]))

plt.xlabel('$q_z$ [nm$^{-1}$]')
plt.ylabel('Reflectivity')
plt.legend()
plt.title('Dynamical X-ray Convoluted')

plt.subplot(2, 1, 2)
plt.pcolormesh(dyn.qz[0, :].to('1/nm').magnitude, delays.to('ps').magnitude, np.log10(R_seq[:, 0, :]), shading='auto')
plt.ylabel('Delay [ps]')
plt.xlabel('$q_z$ [nm$^{-1}$]')

plt.tight_layout()
plt.show()
#%%
model = lm.models.GaussianModel(prefix = "g_")
pars  = lm.Parameters()                                                       
pars.add_many(('g_center',     41.3, True),
              ('g_sigma',      0.6, True),
              ('g_amplitude',  0.1,  True))


COM       = np.zeros(len(delays))
centerFit = np.zeros(len(delays))


for i in range(len(delays)):
    qzROI, IntensitiesROI = hel.setROI1D(dyn.qz[0, :].to('1/nm').magnitude,  R_seq_conv[i, 0, :],40.5,42.7)
    COM[i], a, b = hel.calcMoments(qzROI,IntensitiesROI)
    resultFit     = model.fit(IntensitiesROI, pars, x = qzROI)
    centerFit[i]  = resultFit.values["g_center"]

selectt0  = delays.to('ps').magnitude <=-0.5
strainCOM = -1e3*hel.relChange(COM, np.mean(COM[selectt0]))
strainFit = -1e3*hel.relChange(centerFit, np.mean(centerFit[selectt0]))

hel.writeColumnsToFile(r'SimulationResults/StrainUXRD' + export_name + '.dat', 'delays (ps) \t strain Dy (10^-3) \t strain Nb (10^-3)', 
                       delays.to('ps').magnitude,strainCOM,strainFit)
    
plt.figure(figsize=[6, 0.68*6])
plt.subplot(1, 1, 1)
plt.plot(data_list[0][:,0], data_list[0][:,1],'o',ms = 3,label=layers[0] + ' Data',color = color_list[0])
#plt.plot(delays.to('ps'), mean_strain_list[0],label=layers[0] + 'mean strain',ls = '-',color = color_list[0])
#plt.plot(delays.to('ps'), 1.08*strainCOM,label=layers[0] + 'UXRD',ls = '--',color = color_list[0])
plt.plot(delays.to('ps'), strainFit,label=layers[0] + 'UXRD',ls = ':',color = color_list[0])

plt.xlim(np.min(delays),np.max(delays))
plt.ylabel('strain ($10^{-3}$)')
plt.xlabel('Delay (ps)')
plt.legend()

# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 19:47:19 2021

@author: mamattern
"""

import numpy as np
import matplotlib.pyplot as plt
import udkm1Dsim as ud
import copy
u = ud.u
u.setup_matplotlib()

import udkm.tools.helpers as hel

class simhelpers:
    '''
    This class provides methods to extend the simulation of the udkm1Dsim toolbox to
    magnetic stress contributions and supports/shortens the analysis and simulation.
    The methods base on a dictionary created in the 'udkm1Dsim.py' script that contains
    the properties of the layers of the sample.
    The class contains the following methods:
        
    CalcStressFromTempMap: calculate the stress from a given temperature map
    ExportTransients: create and export a matrix containin the transients for all layers
    GetEnergyProfile: extract the spatio-temporal profile and total amount of energy
    
    
    '''
    def __init__(self):
        self.version = '0_1'
        
    def CalcStressFromTempMap(self,DicSample,DicProp,Layers,TempMap):
        """This method calculates the stress map from the given 'TempMap' using
        the linear expansion and c33 elastic constant of the layers of the sample
        given in the dictionary containing the material methods 'DicSample'
        
        Parameters:
        -----------
        DicSample: dictionary
            the material classes of the sample containing literature values like:
            'lin_therm_exp': expansion coefficient out-of-plane [1/K]
            'elastic_c33': elastic constant c33 [GPa]
            
        DictProperties: dictionary 
            the properties of the layers of the sample like: 
            'select_layer': containing unit cells
            
        Layers: list
            short name of the layers of the sample
            
        TempMap: 2D array
            the increased temperature after excitation of all layers for each delay step
        
        Returns:
        --------       
        StressMap: 2D array
            laser-induced stress for all layers [GPa]
        
        Example:
        --------
        The sample consists of an yttrium and a dysprosium layer
        >>> [[0.1,0.11],[0.09,0.13]]= CalcStressFromTempMap({'Y':Y,'Dy':Dy},DicProp,['Y','Dy'],np.array(([10,9],[9,10])))"""
        
        StressMap = np.zeros((len(TempMap[:,0]),len(TempMap[0,:])))
        for l in range(len(Layers)):
            LinExp   = DicSample[Layers[l]].prop['lin_therm_exp'].to('1/K').magnitude
            ElastC33 = DicSample[Layers[l]].prop['elastic_c33'].to('GPa').magnitude
            StressMap[:,DicProp[Layers[l]]['select_layer']] = TempMap[:,DicProp[Layers[l]]['select_layer']]*LinExp*ElastC33
        return StressMap  
    
    def ExportTransients(self,Quantity,DataList,Delays,FileName):
        """This method creates a matrix containing the transient 'Quantity' from 
        'DataList' for all layers and exports it under 'FileName' in the folder
        'SimulationResults'.
        
        Parameters:
        -----------
        Quantity: string
            name of the exported quantity
            
        DataList: list 
            list of teh transient quantity for all layers
            
        Delays: 1D array
            delays of teh transient quantity
            
        FileName: string
            name of the exported file that also contains 'Quantity'
        
        Example:
        --------
        The sample consists of two layers
        >>> ExportTransients('Temperature',[[10,9,8],[6,7,8]],[0,1,2],'F20mJ_T250K')"""
    
        ExportData      = np.zeros((len(Delays),len(DataList)+1))
        ExportData[:,0] = Delays.to('ps').magnitude
        for i in range(len(DataList)):
            ExportData[:,i+1] = DataList[i]
        hel.makeFolder('SimulationResults')
        np.savetxt('SimulationResults/' + Quantity + '_' + FileName + '.dat',ExportData,header='delay ps \t ' + Quantity + ' of layers (10-3)')
        
    def GetEnergyProfile(self,DicProp,LayerName,TempMap):
        """This method extracts from the given TempMap the time- and depth dependent
        energy distribution and the time-dependent total energy density of the layer
        'LayerName'. This prepares the redistribution of energy density in the case 
        of magnetic excitations conserving the heat transport.
        
        Parameters:
        -----------
        DicProp: dictionary 
            the properties of the layers of the sample like: 
            'select_layer': containing unit cells; 
            'C_layer': heat capacity of the whole layer [J/K*m**2]
            
        LayerName: string
            short name of the layer that is part of the sample and dictionary
            
        TempMap: 2D array
            the increased temperature after excitation of all layers for each delay step
        
        Returns:
        --------       
        ProfileMatrix: 2D array
            normed temperature profiles of the layer 'LayerName' for each delay step
            
        Qlayer: 1D array
            heat density stored in the layer 'LayerName' for each delay step [J/m**2]
        
        Example:
        --------
        The layer 'Au' contains 4 unit cells and 2 delays
        >>> np.array(([0.5,0.5,0,0],[0.33,0.33,0.33,0])), [17,16] = GetEnergyProfile(Properties,'Au',np.array([50,50,0,0],[40,40,40,0]))"""
        
        Tsum          = np.sum(TempMap[:,DicProp[LayerName]['select_layer']],1)
        Qlayer        = np.mean(TempMap[:,DicProp[LayerName]['select_layer']],1)*DicProp[LayerName]['C_layer']
        ProfileMatrix = TempMap[:,DicProp[LayerName]['select_layer']]/Tsum[:,None]
        ProfileMatrix[np.isnan(ProfileMatrix)] = 1/np.sum(DicProp[LayerName]['select_layer'])
        return ProfileMatrix, Qlayer
   
    def GetMagneticEnergy(self,DicSample,DicProp,MagLayerName,Timet0,InitTemp,MagParameters,RemainSat):
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
            the material classes of the sample containing meathods like: getQMag(InitTemp)
        
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
        SelectSaturation = DicProp[MagLayerName]["T_spin_sat"] < self.ReturnTSpinMax(DicSample,DicProp,MagLayerName,InitTemp,MagParameters[3])
        if RemainSat:
            for c in range(len(SelectSaturation[0,:])):
                if np.sum(SelectSaturation[:,c] > 0):
                    IndexSat = np.argmax(SelectSaturation[:,c])
                    DicProp[MagLayerName]["saturation"][IndexSat:,c] = True*np.ones(len(SelectSaturation[IndexSat:,0]))
        else:
            DicProp[MagLayerName]["saturation"] = SelectSaturation
        DicProp[MagLayerName]["saturation"] = DicProp[MagLayerName]["saturation"] > 0.1
        DicProp[MagLayerName]["T_spin_sat"][DicProp[MagLayerName]["saturation"]] = self.ReturnTSpinMax(DicSample,DicProp,MagLayerName,InitTemp,MagParameters[3])    
        
        # return the energy above the saturation to phonon system
        DeltaQSpin = np.sum(DicProp[MagLayerName]["T_spin_sat"] - DicProp[MagLayerName]["T_spin"],1)*DicProp[MagLayerName]["C_unit_cell"]
        DicProp[MagLayerName]["Q_pho"]  = DicProp[MagLayerName]["Q_pho"] + DeltaQSpin
        DicProp[MagLayerName]["Q_spin"] = DicProp[MagLayerName]["Q_spin"] - DeltaQSpin 
        DicProp[MagLayerName]["Q_new"]  = DicProp[MagLayerName]["Q_pho"] + DicProp[MagLayerName]["Q_spin"]        
        DicProp[MagLayerName]["T_spin"] = DicProp[MagLayerName]["T_spin_sat"] 
        DicProp[MagLayerName]["T_pho"]  = DicProp[MagLayerName]["profiles"]*DicProp[MagLayerName]["Q_pho"][:,None]/DicProp[MagLayerName]["C_unit_cell"]

    def GetMagStressSingle(self,DicSample,DicProp,Layers,NumMagLayer,TempMap,InitTemp,Time,Distances,MagParams,RemainSat,StressRise,ElPhoCoupling,ElGrun,ColorList,CalcStress,ExportMaps,ExportName):    
        TempMapDiff = TempMap - InitTemp*np.ones((len(Time),len(Distances)))
        Timet0      = Time[Time>=0]
        TempMapDiff = TempMapDiff[Time>=0,:]
        # Get the time-dependent energy distribution in the sample
        for l  in Layers:
            DicProp[l]["profiles"], DicProp[l]["Q"] = self.GetEnergyDistribution(DicProp,l,TempMapDiff)
            DicProp[l]["Q_new"] = copy.deepcopy(DicProp[l]["Q"])
        
        #Initialize quantities of magnetic layer
        DicProp[Layers[NumMagLayer]]["mag_profiles"], _ = self.GetEnergyDistribution(DicProp,Layers[NumMagLayer],TempMapDiff)
        DicProp[Layers[NumMagLayer]]["Q_pho"]           = copy.deepcopy(DicProp[Layers[NumMagLayer]]["Q"])
        DicProp[Layers[NumMagLayer]]["Q_spin"]          = np.zeros(len(Timet0))
        DicProp[Layers[NumMagLayer]]["saturation"]      = np.zeros((len(Timet0),len(DicProp[Layers[NumMagLayer]]["select_layer"])))   
        self.GetMagneticEnergy(DicSample,DicProp,Layers[NumMagLayer],Timet0,InitTemp,MagParams,RemainSat)
    
        ########## Adjust Phonon Temp in YTop and Bottom layer to achieve continous #############
        Deviations = 1000; MaxIterations = 50; Counter = 0
        
        Qtot = 0
        for l in Layers:
            Qtot = Qtot + DicProp[l]['Q_new']
        FracList = []
        for l in range(len(Layers)):
            FracList.append(DicProp[Layers[l]]['Q_new']/Qtot)
        
        self.PlotMagEnergyDistribution(DicProp,Layers,NumMagLayer,Timet0,ColorList)
        
        Counter=0
        while (np.abs(Deviations)> 0.01) and (Counter<MaxIterations):
            Deviations = self.UpdateAllLayers(DicSample,DicProp,Layers,NumMagLayer,Timet0,InitTemp,MagParams,RemainSat,Qtot,FracList)   
            Counter+=1
            
        TempMapPho, TempMapSpin, TempMap = self.UpdateTempMaps(DicSample,DicProp,Layers,NumMagLayer,Time,Distances,StressRise,ElPhoCoupling,ElGrun)
        
        StressMapPho  = np.zeros((len(Time),len(Distances)))
        StressMapSpin = np.zeros((len(Time),len(Distances)))
        StressMap     = np.zeros((len(Time),len(Distances)))
        
        if CalcStress:
            StressMapPho  = self.CalcStressFromTempMap(DicSample,DicProp,Layers,TempMapPho)
            StressMapSpin = self.CalcStressFromTempMap(DicSample,DicProp,Layers,TempMapSpin)
            StressMap     = self.CalcStressFromTempMap(DicSample,DicProp,Layers,TempMap)       
            np.savetxt(r'SimulationMaps/StressMap_Pho_' + ExportName + '.dat',StressMapPho)
            np.savetxt(r'SimulationMaps/StressMap_Spin_' + ExportName + '.dat',StressMapSpin)
            np.savetxt(r'SimulationMaps/StressMap_Tot_' + ExportName + '.dat',StressMap)
            
        if ExportMaps:
            np.savetxt(r'SimulationMaps/TempMap_Pho_' + ExportName + '.dat',TempMapPho)
            np.savetxt(r'SimulationMaps/TempMap_Spin_' + ExportName + '.dat',TempMapSpin)
            np.savetxt(r'SimulationMaps/TempMap_Tot_' + ExportName + '.dat',TempMap)
        return TempMap, TempMapPho, TempMapSpin, StressMap, StressMapSpin, StressMap
        
    def IncludeFiniteStressRise(self,DicProp,Layers,ListRiseTime,ListInstAmp,TempMap,InitTemp,Distances,Delays):
        Indext0   = np.argmin(abs(Delays))
        Indextmax = np.argmin(abs(Delays.to('ps').magnitude-20))
        for i in np.arange(Indext0,Indextmax,1):
            for l in range(len(Layers)):
                SelectLayer = DicProp[Layers[l]]['select_layer']
                TempMap[i,SelectLayer] = (TempMap[i,SelectLayer]-InitTemp)*(1-ListInstAmp[l]*np.exp(-Delays[i]/ListRiseTime[l]))+InitTemp
        return TempMap
        
    def PlotMagEnergyDistribution(self,DicProp,Layers,MagLayerNum,Timet0,ColorList):
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
        
    def PlotMap(self,Title,Distances,Time,Map,ZeroCrossing):
        plt.figure(figsize=[6, 5])
        plt.subplot(1, 1, 1)
        if ZeroCrossing:
            plt.pcolormesh(Distances.to('nm').magnitude,Time.to('ps').magnitude,Map,shading='auto',cmap='RdBu_r',vmin=-np.max(abs(Map)), vmax=np.max(abs(Map)))
        else:
            plt.pcolormesh(Distances.to('nm').magnitude,Time.to('ps').magnitude,Map,shading='auto',cmap='viridis',vmin=np.min(Map), vmax=np.max(Map))
        plt.colorbar()
        plt.xlim(0,80)
        plt.xlabel('Distance (nm)')
        plt.ylabel('Delay (ps)')
        plt.title(Title)
        plt.tight_layout()
        plt.show()    
    
    
    

    
    def ReturnTSpinMax(self,DicSample,DicProp,MagLayerName,InitTemp,SaturationFac):
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
    

    
    def UpdateAllLayers(self,DicSample,DicProp,Layers,MagLayerNum,Timet0,InitTemp,MagParameters,RemainSat,Qtot,FractionList):
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
                
        self.GetMagneticExcitation(DicSample,DicProp,Layers[MagLayerNum],Timet0,InitTemp,MagParameters,RemainSat)
        
        Difference = DicProp[Layers[MagLayerNum]]["Q_spin"] - QSpinDummy
        Deviation  = np.sum(Difference)
        return Deviation
    
    def UpdateTempMaps(self,DicSample,DicProp,Layers,MagLayerNum,Time,Distances,StressRise,ListCoupling,ListElGruneisen):
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
        if StressRise:
            TempMapPhonon = self.IncludeElPhoCoupling(DicProp,Layers,ListCoupling,ListElGruneisen,TempMapPhonon,Distances,Time[iStart:],0)
    
        MagGrunFac = DicSample[Layers[MagLayerNum]].prop['Grun_mag_c_axis']/DicSample[Layers[MagLayerNum]].prop['Grun_c_axis']
        TempMapSpin[iStart:,DicProp[Layers[MagLayerNum]]["select_layer"]] = MagGrunFac*DicProp[Layers[MagLayerNum]]["T_spin"]
        
        TempMapTotal = TempMapSpin + TempMapPhonon
        return TempMapPhonon, TempMapSpin, TempMapTotal  
    
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
    hel.makeFolder('SimulationResults')
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
    
#def PlotMap(Title,Map,ZeroCrossing):
#    plt.figure(figsize=[6, 5])
#    plt.subplot(1, 1, 1)
 #   if ZeroCrossing:
#        plt.pcolormesh(distances.to('nm').magnitude,delays.to('ps').magnitude,Map,shading='auto',cmap='RdBu_r',vmin=-np.max(abs(Map)), vmax=np.max(abs(Map)))
#    else:
#        plt.pcolormesh(distances.to('nm').magnitude,delays.to('ps').magnitude,Map,shading='auto',cmap='viridis',vmin=np.min(Map), vmax=np.max(Map))
#    plt.colorbar()
#    plt.xlim(0,80)
#    plt.xlabel('Distance (nm)')
#    plt.ylabel('Delay (ps)')
#    plt.title(Title)
#    plt.tight_layout()
#    plt.show()
        

    
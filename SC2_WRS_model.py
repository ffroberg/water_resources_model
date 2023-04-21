import os
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
import math
from pyomo.environ import *
#import pyomo.environ as pyo
from pyomo.opt import SolverFactory


datafolder=os.path.relpath(r'Data')

scatch_char = pd.read_excel(os.path.join(datafolder,'Subcatchments_CPY_input.xlsx')) # EXCEL File with subcatchment characteristics
prec = pd.read_excel(os.path.join(datafolder,'CRU_PRE_CPY_1991_2020.xlsx'))# EXCEL File with rainfall time series per subcatchment in mm/month
pet = pd.read_excel(os.path.join(datafolder,'CRU_PET_CPY_1991_2020.xlsx'))
grun = pd.read_excel(os.path.join(datafolder, 'G-RUN_CPY_1990_2019.xlsx'))
connectivity = pd.read_excel(os.path.join(datafolder, 'CPY_catch_connectivity.xlsx'))
assets_char = pd.read_excel(os.path.join(datafolder, 'Assets_CPY_input.xlsx'))


def Kc(tstamp): # Monthly crop coefficient values - modify this if you want time variable Kc
    return {
        1: 1, # Jan value
        2: 1, # Feb value
        3: 1, # etc
        4: 1,
        5: 1,
        6: 1,
        7: 1,
        8: 1,
        9: 1,
        10: 1,
        11: 1,
        12: 1
    }.get(tstamp.month, 1)

# These data items can be changed by the user
per_cap_ind_demand = 2.777E9/69625582 # m3/cap/yr from Aquastat
per_cap_dom_demand = 2.739E9/69625582 # m3/cap/yr from Aquastat
saltLR = 0.3 # Salt leaching ratio dimensionless
basineff = 0.7 # Basin efficiency dimensionless
WTPag = 0.05*37 # Thai Baht (THB) / m3 or million THB per million m3 1 Euro ~= 37 THB
WTPInd = 0.5*37 # THB / m3 or million THB per million m3
WTPDom = 0.3*37 # THB / m3 or million THB per million m3
WTPPow = 50*37 # THB /MWh
ThaChinDiv = 0.5 #ThaChin diversion, i.e. the fraction of the flow downstream of Upper Chao Phraya catchment that is diverted into Tha Chin. Fraction (dimensionless)


savepath = r'Data' #adust this path to write results in specific folder on your system

# Catchment data
ncatch = scatch_char['ID'].astype(int).tolist() # Catchment IDs; Note the use of the python dictionary data type for data items
nassets = assets_char['ID'].astype(int).tolist()                   # Reservoir IDs - transform to list
scatch_areas = scatch_char.set_index('ID').to_dict()['Area (km2)'] # subcatchment areas in km2. Keys are specific to the EXCEL input file!
scatch_people = scatch_char.set_index('ID').to_dict()['Number of people'] # subcatchment population. Keys are specific to the EXCEL input file!
scatch_airr =  scatch_char.set_index('ID').to_dict()['Area equipped for irrigation (ha)'] # subcatchment area equipped for irrigation. Keys are specific to the EXCEL input file!
scatch_dslink = connectivity.set_index('CATCHID').to_dict()['DSCATCHID'] # downstream link of each catchment. Keys are specific to the EXCEL input file!
scatch_uslink1 = connectivity.set_index('CATCHID').to_dict()['USCATCHID1'] # upstream link 1 of each catchment. Keys are specific to the EXCEL input file!
scatch_uslink2 = connectivity.set_index('CATCHID').to_dict()['USCATCHID2'] # upstream link 2 of each catchment. Keys are specific to the EXCEL input file!
scatch_uslink3 = connectivity.set_index('CATCHID').to_dict()['USCATCHID3'] # upstream link 3 of each catchment. Keys are specific to the EXCEL input file!
scatch_reservoir = connectivity.set_index('CATCHID').to_dict()['Reservoir'] # reservoir for each catchment; -1 if no reservoir. Keys are specific to the EXCEL input file!
runoff_rate = dict() # Create empty runoff dictionary
prec_rate = dict() # Create empty precipitation dictionary
pet_rate = dict() # Create empty PET dictionary
for c in ncatch: # Loop through all catchments
    runoff_rate[c] = grun.set_index('Month').to_dict()[c] # runoff rate in mm per day. One dictionary per catchment 'Month' key is specific to the EXCEL input file!
    prec_rate[c] = prec.set_index('Month').to_dict()[c] # precipitation rate in mm per month. One dictionary per catchment 'Month' key is specific to the EXCEL input file!
    pet_rate[c] = pet.set_index('Month').to_dict()[c] # potential ET rate in mm per day. One dictionary per catchment 'Month' key is specific to the EXCEL input file!
ntimestamps = list(runoff_rate[1].keys())[12:] # 29 years, 1991-2019 get them from catchid 1; This is the period for which precip, PET and runoff data overlap
ntimes = np.arange(1,len(ntimestamps)+1,1) # array of time step indicators ranging 1,2...,n-1,n
ntsdic = dict(zip(ntimes, ntimestamps)) # dictionary relating time step indicators to timestamps
IndDem = dict() # Create empty dictionary for industrial demand
DomDem = dict() # Create empty dictionary for domestic demand
AgDem = dict() # Create empty dictionary for agricultural demand as dictionary with double index n - for use in pyomo
AgDempl = dict() # Create empty dictionary for agricultural demand as nested dictionary - for plotting
RO = dict() # Create empty dictionary for runoff as dictionary with double index n - for use in pyomo
ROpl = dict() # Create empty dictionary for runoff as nested dictionary - for plotting
ROindividual = dict ()
for c in ncatch:
    IndDem[c] = scatch_people[c]*per_cap_ind_demand/12/1E6 # industrial demand million m3 per month = number of people times per capita demand
    DomDem[c] = scatch_people[c]*per_cap_dom_demand/12/1E6 # domestic demand million m3 per month = number of people times per capita demand
    AgDempl[c]=dict() # create one empty sub-dictionary per catchment
    ROpl[c]=dict() # create one empty sub-dictionary per catchment
    ROindividual[c] = dict()
    for t in ntimes: # loop through all time steps
        tstamp = ntsdic[t] # look up the time stamp corresponding to the time step
        irrigation_rate = (Kc(tstamp)*pet_rate[c][tstamp]*365/12 - prec_rate[c][tstamp])*(1+saltLR)/basineff #Calculate the irrigation rate
        if irrigation_rate < 0: # if rainfall exceeds crop water demand
            irrigation_rate = 0 # set irrigation rate to zero
        AgDem[(c,t)] = irrigation_rate/1000*scatch_airr[c]/100 # ag. demand dependent on time in million m3 per month
        RO[(c,t)] = runoff_rate[c][tstamp]*365/12/1000*scatch_areas[c] # Runoff generated in each catchment in million m3 per month
        AgDempl[c][t] = irrigation_rate/1000*scatch_airr[c]/100 # same values stored in nested dictionary
        ROpl[c][t] = runoff_rate[c][tstamp]*365/12/1000*scatch_areas[c] # same values stored in nested dictionary
        ROindividual[c][t] = runoff_rate[c][tstamp]*365/12/1000 # m/month

# Reservoir data
## add flood safety storage to reservoir (floods will not be visible on mnth time-scale) ##
Aname = assets_char.set_index('ID').to_dict()['Name'] # Asset/Reservoir name; dictionary relating name to ID
Aname2 = {y:x for x,y in Aname.items()} # invert the dictionary, i.e. produce a dictionary that gives the ID for each name       
Aweq = assets_char.set_index('ID').to_dict()['Estimated water-energy equivalent (kWh/m3)'] # Water energy equivalent for each reservoir
for ires in Aweq: # catch any missing values
    if math.isnan(Aweq[ires]):
        Aweq[ires]=0
AResCap = assets_char.set_index('ID').to_dict()['Storage Capacity (million m3)'] # Reservoir capacity in million m3
for reskey in AResCap.keys(): # catch any missing values
    try:
        check=AResCap[reskey]/2
    except TypeError:
        AResCap [reskey]=0
for ires in AResCap: # catch any missing NaNs
    if math.isnan(AResCap[ires]):
        AResCap[ires]=0
AResini=dict() # Initial reservoir storage in million m3
for reskey in AResCap.keys(): # Set to half of maximum storage and catch any missing values
    try:
        AResini[reskey]=AResCap[reskey]/2
    except TypeError:
        AResini[reskey]=0
AResTCapm3 = assets_char.set_index('ID').to_dict()['Turbine capacity (million m3/month)'] # Turbine capacity in million m3 per month
for reskey in AResTCapm3.keys(): # catch any missing values
    try:
        check=AResTCapm3[reskey]/2
    except TypeError:
        AResTCapm3 [reskey]=0
for ires in AResTCapm3:
    if math.isnan(AResTCapm3[ires]):
        AResTCapm3[ires]=0
for c in scatch_reservoir: # Replace reservoir name with reservoir ID in scatch_reservoir
    if scatch_reservoir[c] != -1:
        scatch_reservoir[c] = Aname2.get(scatch_reservoir[c],-1)
scatch_reservoir2 = {y:x for x,y in scatch_reservoir.items()} # invert the dictionary - can be used to look up catchment belonging to each reservoir
del scatch_reservoir2[-1] # delete key -1

#######
# model and opt here
#######
#------------------------------------------------------------------------------------------
# Create and run the pyomo model
#----------------------------------------------------------------------------------------------
# Define decision variables and parameters
model = ConcreteModel() # define the model

model.ncatch = Set(initialize=ncatch) # define catchment index, set values to ncatch
model.ntimes = Set(initialize=ntimes) # define time index, set values to ntimes
model.nres = Set(initialize=nassets) # define reservoir index, set values to nassets

# Declare decision variables - decision variable values will be provided by the optimizer
model.Aag  = Var(model.ncatch, model.ntimes, within=NonNegativeReals) # note double index. Agricultural allocation per time step and per subcatchment, million m3 or MCM
model.Aind  = Var(model.ncatch, model.ntimes, within=NonNegativeReals) # note double index. Industrial allocation per time step and per subcatchment, MCM
model.Adom  = Var(model.ncatch, model.ntimes, within=NonNegativeReals) # note double index. Domestic allocation per time step and per subcatchment, MCM
model.Qds  = Var(model.ncatch, model.ntimes, within=NonNegativeReals) # note double index. Outflow per time step and per sc, MCM
model.Rel   = Var(model.nres, model.ntimes, within=NonNegativeReals) # note double index. One release per month and per reservoir. turbined power, MCM
model.Spill   = Var(model.nres, model.ntimes, within=NonNegativeReals) #  note double index. One spill per month and per reservoir. Water flowing past the turbines, MCM
model.Send   = Var(model.nres, model.ntimes, within=NonNegativeReals) # note double index. One end storage per month and per reservoir. MCM

# Declare parameters
model.endtime = Param(initialize = ntimes[-1]) # find end time step of the model
model.dslink = Param(model.ncatch, within=Integers, initialize = scatch_dslink) # Set downstream catchment for all catchments; static, only catchment index
model.uslink1 = Param(model.ncatch, within=Integers, initialize = scatch_uslink1) # Set first upstream catchment for all catchments; static, only catchment index
model.uslink2 = Param(model.ncatch, within=Integers, initialize = scatch_uslink2) # Set second upstream catchment for all catchments; static, only catchment index
model.uslink3 = Param(model.ncatch, within=Integers, initialize = scatch_uslink3) # Set 3rd upstream catchment for all catchments; static, only catchment index
model.resyn = Param(model.ncatch, within=Integers, initialize = scatch_reservoir) # Set reservoir ID for each catchment; static, only catchment index
model.catch4res = Param(model.nres, within=Integers, initialize = scatch_reservoir2) # Set catchment ID for each reservoir; static, only reservoir index, tells the catchment for each reservoir
model.RO   = Param(model.ncatch, model.ntimes, within=NonNegativeReals,initialize = RO) # Set runoff; time variable, 2 indeces, month and catchment
model.AgDem   = Param(model.ncatch, model.ntimes,within=NonNegativeReals,initialize = AgDem) # Set agricultural demand; time variable, 2 indeces, month and catchment, unit MCM
model.IndDem  = Param(model.ncatch,within=NonNegativeReals,initialize = IndDem) # Set industrial demand; assumed constant in time here, only catchment index, unit MCM
model.DomDem  = Param(model.ncatch,within=NonNegativeReals,initialize = DomDem) # Set domestic demand; assumed constant in time here, only catchment index, unit MCM
model.WTPag  = Param(within=NonNegativeReals,initialize = WTPag) # Set WTP for agricultural water allocation; assumed constant in time and uniform across catchments here, therefore no index, THB/m3
model.WTPInd  = Param(within=NonNegativeReals,initialize = WTPInd) # Set WTP for industrial water allocation; assumed constant in time and uniform across catchments here, therefore no index, THB/m3
model.WTPDom  = Param(within=NonNegativeReals,initialize = WTPDom) # Set WTP for domestic water allocation; assumed constant in time and uniform across catchments here, therefore no index, THB/m3
model.WTPPow = Param(within=NonNegativeReals,initialize = WTPPow) # Set WTP for electical power, assumed constant in time and uniform across catchments here, therefore no index, THB/MWh
model.Resweq = Param(model.nres, within=NonNegativeReals,initialize = Aweq,default=0) # Set water-energy equivalent for all reservoirs; varies from reservoir to reservoir, therefore 1 index, kWh/m3
model.ResCap = Param(model.nres, within=NonNegativeReals,initialize = AResCap,default=0) # Set reservoir capacity for all reservoirs; varies from reservoir to reservoir, therefore 1 index, MCM
model.ResTCapm3 = Param(model.nres, within=NonNegativeReals,initialize = AResTCapm3,default=0) # Set turbine capacity for all reservoirs; varies from reservoir to reservoir, therefore 1 index, MCM
model.ResSini = Param(model.nres, within=NonNegativeReals,initialize = AResini, default=0) # Set initial reservoir storage for all reservoirs; varies from reservoir to reservoir, therefore 1 index, MCM
model.ThaChin = Param(within=NonNegativeReals,initialize =ThaChinDiv) # ThaChin diversion in percent of flow downstream of upper Chao Phraya; Just one number, therefore no index, dimensionless, fraction
#Set up the model
#Objective function: Sum benefit over all users, all time steps and all subcatchments
def obj_rule(model):
    global ag_ben, ind_ben, dom_ben, pow_ben
    ag_ben = sum(model.WTPag*model.Aag[c,t] for c in model.ncatch for t in model.ntimes)
    ind_ben = sum(model.WTPInd*model.Aind[c,t]  for c in model.ncatch for t in model.ntimes)
    dom_ben = sum(model.WTPDom*model.Adom[c,t]  for c in model.ncatch for t in model.ntimes)
    pow_ben = sum(model.WTPPow*model.Resweq[r]*model.Rel[r,t]/1000 for r in model.nres for t in model.ntimes)
    return ag_ben + ind_ben + dom_ben + pow_ben

model.obj = Objective(rule=obj_rule, sense = maximize)

# Agricultural demand constraint per catchment. Active for every time step and catchment, thus two indices
def wd_ag_c(model, nc, nt):
    return model.Aag[nc,nt] <= model.AgDem[nc,nt]
model.wd_ag = Constraint(model.ncatch, model.ntimes, rule=wd_ag_c)

# Industrial demand constraint per catchment. Active for every time step and catchment, thus two indices
def wd_ind_c(model, nc, nt):
    return model.Aind[nc, nt] <= model.IndDem[nc]
model.wd_ind = Constraint(model.ncatch, model.ntimes, rule=wd_ind_c)

# Domestic demand constraint per catchment. Active for every time step and catchment, thus two indices
def wd_dom_c(model, nc, nt):
    return model.Adom[nc,nt] <= model.DomDem[nc]
model.wd_dom = Constraint(model.ncatch, model.ntimes, rule=wd_dom_c)


# Catchment water balance per catchment.. Active for every time step and catchment, thus two indices
# Downstream flow is equal to inflow plus runoff minus use for all catchments, except 33 and 28, where inflow is partitioned according to model.ThaChin
def wb_c(model, nc, nt):
    Inus = 0.
    if  model.uslink1[nc] != -1:
        resindex = model.resyn[model.uslink1[nc]]
        if resindex != -1:
            Inus = Inus + model.Rel[resindex,nt] + model.Spill[resindex,nt]
        else:
            Inus = Inus + model.Qds[model.uslink1[nc],nt]
    if  model.uslink2[nc] != -1:
        resindex = model.resyn[model.uslink2[nc]]
        if resindex != -1:
            Inus = Inus + model.Rel[resindex,nt] + model.Spill[resindex,nt]
        else:
            Inus = Inus + model.Qds[model.uslink2[nc],nt]
    if  model.uslink3[nc] != -1:
        resindex = model.resyn[model.uslink3[nc]]
        if resindex != -1:
            Inus = Inus + model.Rel[resindex,nt] + model.Spill[resindex,nt]
        else:
            Inus = Inus + model.Qds[model.uslink3[nc],nt]
    if nc == 33: # Lower Chao Phraya gets only a portion of Qds from Upper Chao Phraya
        Inus = model.Rel[2,nt] + model.Spill[2,nt] + (1 - model.ThaChin) * model.Qds[24,nt]
    if nc == 28: # Tha Chin gets the rest of Qds from Upper Chao Phraya
        Inus  = model.ThaChin*model.Qds[24,nt]       
    return model.Qds[nc,nt] == Inus + model.RO[nc,nt] - model.Aag[nc,nt] - model.Aind[nc,nt]- model.Adom[nc,nt]
model.wb = Constraint(model.ncatch, model.ntimes, rule=wb_c)

# Reservoir mass balance.. Active for every time step and reservoir, thus two indices
# Storage at the end of the time step is initial stroage plus inflow minus release minus spill
def res_c(model, nr, nt):
    inflowindex = model.catch4res[nr]
    if nt == 1:
        return model.Send[nr,nt] == model.ResSini[nr] + model.Qds[inflowindex,nt] - model.Rel[nr,nt] - model.Spill[nr,nt]
    else:
        return model.Send[nr, nt] == model.Send[nr,nt-1] + model.Qds[inflowindex,nt] - model.Rel[nr,nt] - model.Spill[nr,nt]
model.res = Constraint(model.nres, model.ntimes, rule=res_c)

# Reservoir capacity
# storage in each reservoir should be less than capacity. Active for every time step and reservoir, thus two indices
def rescap_c(model, nr, nt):
    return model.Send[nr,nt] <= model.ResCap[nr]
model.rescap = Constraint(model.nres, model.ntimes, rule=rescap_c)

# Turbine capacity
# turbined power should be less than capacity at each time step, each reservoir. . Active for every time step and reservoir, thus two indices
def turcap_c(model, nr, nt):
    return model.Rel[nr,nt] <= model.ResTCapm3[nr]
model.turcap = Constraint(model.nres, model.ntimes, rule=turcap_c)

# End storage equal to initial storage. Active for each reservoir, thus one index
def endstor_c(model, nr):
    return model.Send[nr,model.endtime] == model.ResSini[nr]
model.endstor = Constraint(model.nres, rule=endstor_c)

# Solve the model

model.dual = Suffix(direction=Suffix.IMPORT) # formulate dual problem to provide shadow prices

# Create a solver
opt = SolverFactory('glpk')
#Solve
results = opt.solve(model)

####
# Output of objective function, optimal decisions and shadow prices
#----------------------------------------------------------------------------------------------
# You can of course adjust all file names as you may wish


# Objective value
print("Total Benefit in optimal solution: ", round(value(model.obj)/(len(model.ntimes)/12)/1000,2), " billion THB per year")
print("Aggregrated benefit", round(value(ag_ben)/(len(model.ntimes)/12)/1000,2), " billion THB per year" )
print("Domestic benefit", round(value(dom_ben)/(len(model.ntimes)/12)/1000,2), " billion THB per year" )
print("Industry benefit", round(value(ind_ben)/(len(model.ntimes)/12)/1000,2), " billion THB per year" )
print("Power benefit", round(value(pow_ben)/(len(model.ntimes)/12)/1000,2), " billion THB per year" )


#Save optimal decisions
# Agricultural allocations, saved to path outpath
outpath =  savepath + os.sep + r'Ag_optimal_allocations.xlsx'
defoutpath =  savepath + os.sep + r'Ag_deficits_optimal.xlsx'
optAAg = dict()
optDAg = dict()
for c in ncatch:
    moptA = dict()
    moptD = dict()
    for t in ntimes:
        moptA[t]=model.Aag[c,t].value
        moptD[t]=model.AgDem[c,t]-model.Aag[c,t].value
    optAAg[c]=moptA
    optDAg[c]=moptD

# Average optimal Allocation
AvoptAAg = dict()
for cindex in optAAg.keys():
    AvoptAAg[cindex] = np.mean(list(optAAg[cindex].values()))

optAAg = pd.DataFrame.from_dict(optAAg)
optDAg = pd.DataFrame.from_dict(optDAg)
optAAg.to_excel(outpath)
optDAg.to_excel(defoutpath)


# Industrial allocations, saved to path outpath
outpath =  savepath + os.sep + r'Ind_optimal_allocations.xlsx'
defoutpath =  savepath + os.sep + r'Ind_deficits_optimal.xlsx'
optAInd = dict()
optDInd = dict()
for c in ncatch:
    moptA = dict()
    moptD = dict()
    for t in ntimes:
        moptA[t]=model.Aind[c,t].value
        moptD[t]=model.IndDem[c]-model.Aind[c,t].value
    optAInd[c]=moptA
    optDInd[c]=moptD

# Average optimal Allocation
AvoptAInd = dict()
for cindex in optAInd.keys():
    AvoptAInd[cindex] = np.mean(list(optAInd[cindex].values()))


optAInd = pd.DataFrame.from_dict(optAInd)
optDInd = pd.DataFrame.from_dict(optDInd)
optAInd.to_excel(outpath)
optDInd.to_excel(defoutpath)



# Domestic allocations, saved to path outpath
outpath =  savepath + os.sep + r'Dom_optimal_allocations.xlsx'
defoutpath =  savepath + os.sep + r'Dom_deficits_optimal.xlsx'
optADom = dict()
optDDom = dict()
for c in ncatch:
    moptA = dict()
    moptD = dict()
    for t in ntimes:
        moptA[t]=model.Adom[c,t].value
        moptD[t]=model.DomDem[c]-model.Adom[c,t].value
    optADom[c]=moptA
    optDDom[c]=moptD

# Average optimal Allocation
AvoptADom = dict()
for cindex in optADom.keys():
    AvoptADom[cindex] = np.mean(list(optADom[cindex].values()))

optADom = pd.DataFrame.from_dict(optADom)
optDDom = pd.DataFrame.from_dict(optDDom)
optADom.to_excel(outpath)
optDDom.to_excel(defoutpath)  

# Catchment outflow, saved to path outpath
outpath =  savepath + os.sep + r'Catchment_outflow_opt.xlsx'
optOF = dict()
for c in ncatch:
    moptA = dict()
    for t in ntimes:
        moptA[t]=model.Qds[c,t].value
    optOF[c]=moptA
optOF = pd.DataFrame.from_dict(optOF)
optOF.to_excel(outpath)

# Reservoir release, saved to path outpath
outpath =  savepath + os.sep + r'Res_release_opt.xlsx'
optRelease = dict()
for r in model.nres:
    moptA = dict()
    for t in ntimes:
        moptA[t]=model.Rel[r,t].value
    optRelease[r]=moptA
optRelease = pd.DataFrame.from_dict(optRelease)
optRelease.to_excel(outpath)

# Reservoir spills, saved to path outpath
outpath =  savepath + os.sep + r'Res_spills_opt.xlsx'
optSpill = dict()
for r in model.nres:
    moptA = dict()
    for t in ntimes:
        moptA[t]=model.Spill[r,t].value
    optSpill[r]=moptA
optSpill = pd.DataFrame.from_dict(optSpill)
optSpill.to_excel(outpath)

# Reservoir end storage, saved to path outpath
outpath =  savepath + os.sep + r'Res_Send_opt.xlsx'
optStor = dict()
for r in model.nres:
    moptA = dict()
    for t in ntimes:
        moptA[t]=model.Send[r,t].value
    optStor[r]=moptA
optStor = pd.DataFrame.from_dict(optStor)
optStor.to_excel(outpath)  
#****************************************************************************
#Save Shadow prices for all constraints
#******************************************************************************
# Ag demand constraints shadow prices saved to path outpath
outpath =  savepath + os.sep + r'Ag_dem_SP.xlsx'
SPAgDem = dict()
for c in ncatch:
    moptA = dict()
    for t in ntimes:
        moptA[t]=model.dual[model.wd_ag[c,t]]
    SPAgDem[c]=moptA
SPAgDem = pd.DataFrame.from_dict(SPAgDem)
SPAgDem.to_excel(outpath)

    
# Ind demand constraints shadow prices saved to path outpath
outpath =  savepath + os.sep + r'Ind_dem_SP.xlsx'
SPIndDem = dict()
for c in ncatch:
    moptA = dict()
    for t in ntimes:
        moptA[t]=model.dual[model.wd_ind[c,t]]
    SPIndDem[c]=moptA
SPIndDem = pd.DataFrame.from_dict(SPIndDem)
SPIndDem.to_excel(outpath)
    
# Domestic demand constraints shadow prices saved to path outpath
outpath =  savepath + os.sep + r'Dom_dem_SP.xlsx'
SPDomDem = dict()
for c in ncatch:
    moptA = dict()
    for t in ntimes:
        moptA[t]=model.dual[model.wd_dom[c,t]]
    SPDomDem[c]=moptA
SPDomDem = pd.DataFrame.from_dict(SPDomDem)
SPDomDem.to_excel(outpath)
    
# Catchment water balances shadow prices saved to outpath
outpath =  savepath + os.sep + r'Catch_WB_SP.xlsx'
SPCWB = dict()
for c in ncatch:
    moptA = dict()
    for t in ntimes:
        moptA[t]=model.dual[model.wb[c,t]]
    SPCWB[c]=moptA
SPCWB = pd.DataFrame.from_dict(SPCWB)
SPCWB.to_excel(outpath)
    
# Reservoir mass balance constraint shadow prices saved to outpath
outpath =  savepath + os.sep + r'Res_WB_SP.xlsx'
SPResMB = dict()
for r in model.nres:
    moptA = dict()
    for t in ntimes:
        moptA[t]=model.dual[model.res[r,t]]
    SPResMB[r]=moptA
SPResMB = pd.DataFrame.from_dict(SPResMB)
SPResMB.to_excel(outpath)

# Reservoir capacity constraint shadow prices saved to outpath
outpath =  savepath + os.sep + r'Res_Cap_SP.xlsx'
SPResCap = dict()
for r in model.nres:
    moptA = dict()
    for t in ntimes:
        moptA[t]=model.dual[model.rescap[r,t]]
    SPResCap[r]=moptA
SPResCap = pd.DataFrame.from_dict(SPResCap)
SPResCap.to_excel(outpath)

# Turbine capacity constraint shadow prices saved to outpath
outpath =  savepath + os.sep + r'Res_TurCap_SP.xlsx'
SPTCap = dict()
for r in model.nres:
    moptA = dict()
    for t in ntimes:
        moptA[t]=model.dual[model.turcap[r,t]]
    SPTCap[r]=moptA
SPTCap = pd.DataFrame.from_dict(SPTCap)
SPTCap.to_excel(outpath)

#----------------------------------------------------------------------------------------------
# Output of objective function, optimal decisions and shadow prices
#----------------------------------------------------------------------------------------------
# You can of course adjust all

# Plotting
#***********************************************************
# KC over time

plt.rcParams.update({'font.size': 16})


plt.figure(figsize=[20,10])
Kct=[]
for ts in ntimestamps:
    Kct.append(Kc(ts))
plt.bar(ntimes[0:24],Kct[0:24])
plt.xlabel('time step')
plt.ylabel('Crop Coefficient, dimensionless')
# Runoff time series from any catchment


plt.figure(figsize=[20,10])

catchselect = 24
plt.bar(ROpl[catchselect].keys(),ROpl[catchselect].values())
plt.xlabel('time step')
plt.ylabel('runoff in million m^3 per month')
plt.title('Catchment: ' + str(catchselect))

# Ag demand time series for any catchment
plt.figure(figsize=[20,10])
catchselect = 24
plt.bar(AgDempl[catchselect].keys(),AgDempl[catchselect].values())
plt.xlabel('time step')
plt.ylabel('Irrigation water demand in million m^3 per month')
plt.title('Catchment: ' + str(catchselect))
# Average runoff per catchment
plt.figure(figsize=[20,10])
AvROpl = dict()
for cindex in ROpl.keys():
    AvROpl[cindex] = np.mean(list(ROpl[cindex].values()))

plt.bar(np.arange(len(AvROpl.keys())),AvROpl.values())
plt.xticks(np.arange(len(AvROpl.keys())), AvROpl.keys())
plt.xlabel('Catchment ID')
plt.ylabel('Average runoff in million m^3 per month')
# Produce table that you can join to the subcatchment attribute table in QGIS
AvROpc_pd = pd.DataFrame.from_dict(AvROpl, orient = 'index',columns=['Average RO in m3/month'])
AvROpc_pd.to_excel(savepath + os.sep + 'AvROpl.xlsx',index_label='ID')


# Average runoff for each catchment without their area. Unit m/moth
AvROindividual = dict()
for cindex in ROindividual.keys():
    AvROindividual[cindex] = np.mean(list(ROindividual[cindex].values()))
AvROindividualpc_pd = pd.DataFrame.from_dict(AvROindividual, orient = 'index',columns=['Average RO in m/month'])
AvROindividualpc_pd.to_excel(savepath + os.sep + 'AvROindividual.xlsx',index_label='ID')

# Average irrigation demand per catchment
plt.figure(figsize=[20,10])
AvAgDempl = dict()
for cindex in AgDempl.keys():
    AvAgDempl[cindex] = np.mean(list(AgDempl[cindex].values()))
plt.bar(np.arange(len(AvAgDempl.keys())),AvAgDempl.values())
plt.xticks(np.arange(len(AvAgDempl.keys())), AvAgDempl.keys())
plt.xlabel('Catchment ID')
plt.ylabel('Average irrigation demand in million m^3 per month')

#######


# Produce table that you can join to the subcatchment attribute table in QGIS
AvAgDempl_pandas = pd.DataFrame.from_dict(AvAgDempl, orient = 'index',columns=['Average Agreggation demand in m3/month'])
AvAgDempl_pandas.to_excel(savepath + os.sep + 'AvAgDempl.xlsx',index_label='ID')


# PLOTTING THEM ALL TOGETHER
catchments = connectivity['CATCHID']
sum_dem = np.array([])
sum_allo = np.array([])

# define the number of rows and columns for the subplots
nrows = 4
ncols = 4

# create a new figure with the specified number of subplots and tight layout
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=[20,10], constrained_layout=True)

# loop over each catchment and plot it in a subplot
for i, catchselect in enumerate(catchments):
    # calculate the row and column index for the current subplot
    row_idx = i // ncols
    col_idx = i % ncols
    
    seltimes = np.arange(1, 24, 1)
    #sums yearly demand and allocation for each catchment
    sum_dem = np.append(sum_dem, sum(AgDempl[catchselect].values()))
    sum_allo = np.append(sum_allo, sum(optAAg[catchselect].values))

    # plot the catchment in the current subplot
    axs[row_idx, col_idx].bar(AgDempl[catchselect].keys(), AgDempl[catchselect].values())
    axs[row_idx, col_idx].bar(optAAg[catchselect].keys(), optAAg[catchselect].values)
    #axs[row_idx, col_idx].set_xlabel('time step')
    #axs[row_idx, col_idx].set_ylabel('Irrigation demand and allocation, million m3')
    axs[row_idx, col_idx].set_title('Catchment: ' + str(catchselect))
    #axs[row_idx, col_idx].legend(('Demand', 'Allocation'))

# add a common x label and a common legend
fig.supxlabel('time step')
fig.supylabel('Irrigation water, million m3')
fig.legend(('Demand', 'Allocation'), loc='lower center', ncol=2)

# remove the empty subplots
for i in range(len(catchments), nrows*ncols):
    row_idx = i // ncols
    col_idx = i % ncols
    axs[row_idx, col_idx].remove()

# show the plot
plt.show()


# percentage of demand met for each catchment
perc_dem = (sum_allo/sum_dem)*100
str_catchments = [str(x) for x in catchments]
plt.figure(figsize=[20,10])
plt.bar(str_catchments,perc_dem)
plt.xlabel('Catchment ID')
plt.ylabel('Percentage of demand met by allocation')
plt.title('Percentage of demand met by allocation')

# irrigation demand and allocation
plt.figure(figsize=[20,10])
catchselect = 1
seltimes = np.arange(1,24,1)
plt.bar(AgDempl[catchselect].keys(),AgDempl[catchselect].values())
plt.bar(optAAg[catchselect].keys(),optAAg[catchselect].values)
plt.xlabel('time step')
plt.ylabel('Irrigation water demand and irrigation water allocation, million m3')
plt.title('Catchment: ' + str(catchselect))
plt.legend(('Demand','Allocation'))

#spill time series for all reservoirs
fig, ax = plt.subplots(figsize=[20, 10])

total_spill = np.zeros(len(optSpill[list(optSpill.keys())[0]]))

for Aname in optSpill.keys():
    total_spill += optSpill[Aname]

ax.bar(optSpill[list(optSpill.keys())[0]].keys(), total_spill)
ax.set_xlabel('time step')
ax.set_ylabel('Total reservoir spill')
ax.set_title('Total spill from all reservoirs')
plt.show()

# End Storage time series
plt.figure(figsize=[20,10])
resselect = 'Bhumipol'
rselect = Aname2[resselect]
plt.bar(optStor[rselect].keys(),optStor[rselect].values)
plt.xlabel('time step')
plt.ylabel('End storage in million m^3')
plt.title('Reservoir: ' + str(resselect))

#End storage time series for all reservoirs
fig, ax = plt.subplots(figsize=[20, 10])

total_storage = np.zeros(len(optStor[list(optStor.keys())[0]]))

for Aname in optStor.keys():
    total_storage += optStor[Aname]

ax.bar(optStor[list(optStor.keys())[0]].keys(), total_storage)
ax.set_xlabel('time step')
ax.set_ylabel('Total reservoir end storage in million m^3')
ax.set_title('Total end storage from all reservoirs')
plt.show()

# Water Shadow price time series for any catchment
plt.figure(figsize=[20,10])
catchselect = 24
plt.bar(SPCWB[catchselect].keys(),SPCWB[catchselect].values)
plt.xlabel('time step')
plt.ylabel('Water Shadow price, THB per m3')
plt.title('Catchment: ' + str(catchselect))


# Average water shadow price per catchment
plt.figure(figsize=[20,10])
AvSPCWB = dict()
for cindex in SPCWB.keys():
    AvSPCWB[cindex] = np.mean(list(SPCWB[cindex].values))
plt.bar(np.arange(len(AvSPCWB.keys())),AvSPCWB.values())
plt.xticks(np.arange(len(AvSPCWB.keys())), AvSPCWB.keys())
plt.xlabel('Catchment ID')
plt.ylabel('Average Water Shadow price, THB per m3')
# Produce table that you can join to the subcatchment attribute table in QGIS
AvSPCWB_pandas = pd.DataFrame.from_dict(AvSPCWB, orient = 'index',columns=['Average Water Shadow price, THB per m3'])
AvSPCWB_pandas.to_excel(savepath + os.sep + 'AvSPCWB.xlsx',index_label='ID')
# Reservoir capacity Shadow price time series for any reservoir
plt.figure(figsize=[20,10])
resselect = 'Bhumipol'
rselect = Aname2[resselect]
plt.bar(SPResCap[rselect].keys(),SPResCap[rselect].values)
plt.xlabel('time step')
plt.ylabel('Reservoir capacity Shadow price, THB per m3')
plt.title('Reservoir: ' + str(resselect))
# Average capacity shadow price per reservoir
plt.figure(figsize=[20,10])
AvSPResCap = dict()
for rindex in SPResCap.keys():
    AvSPResCap[rindex] = np.mean(list(SPResCap[rindex].values))
plt.bar(AvSPResCap.keys(),AvSPResCap.values())
plt.xlabel('Reservoir ID')
plt.ylabel('Average reservoir capacity shadow price, THB per m3')
# Produce table that you can join to the subcatchment attribute table in QGIS
AvSPResCap_pandas = pd.DataFrame.from_dict(AvSPResCap, orient = 'index',columns=['Average Reservoir Capacity Shadow price, THB per m3'])
AvSPResCap_pandas.to_excel(savepath + os.sep + 'AvSPResCap.xlsx',index_label='ID')


# Deficit and demand plots for all catchments + sum of deficit
# PLOTTING THEM ALL TOGETHER

DeficitSum = dict()

# define the number of rows and columns for the subplots
nrows = 4
ncols = 4

# create a new figure with the specified number of subplots and tight layout
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=[20,10], constrained_layout=True)

# loop over each catchment and plot it in a subplot
for i, catchselect in enumerate(ncatch):
    # calculate the row and column index for the current subplot
    row_idx = i // ncols
    col_idx = i % ncols
    
    seltimes = np.arange(1, 24, 1)
    
    # plot the catchment in the current subplot
    axs[row_idx, col_idx].bar(AgDempl[catchselect].keys(), AgDempl[catchselect].values())
    axs[row_idx, col_idx].bar(optDAg[catchselect].keys(), optDAg[catchselect].values, color='green')
    axs[row_idx, col_idx].set_xlabel('time step')
    #axs[row_idx, col_idx].set_ylabel('Irrigation water, million m3')
    axs[row_idx, col_idx].set_title('Catchment: ' + str(catchselect))
    #axs[row_idx, col_idx].legend(('Ag. Demand', 'Ag. optimal deficit'))

    # Calculate sum of deficit for all catchment and add to a dictionary
    DeficitSum[catchselect] = np.sum(optDAg[catchselect])

# add a common x label and a common legend
#fig.supxlabel('time step')
fig.supylabel('Irrigation water, million m3')
fig.legend(('Ag. Demand', 'Ag. optimal deficit'), loc='lower center', ncol=2)

# remove the empty subplots
for i in range(len(ncatch), nrows*ncols):
    row_idx = i // ncols
    col_idx = i % ncols
    axs[row_idx, col_idx].remove()

# show the plot
plt.show()

# Reservoir release plots
plt.figure(figsize=[20,10])
plt.bar(optRelease[rselect].keys(),optRelease[rselect].values)
plt.xlabel('time step')
plt.ylabel('Reservoir release')
plt.title('Reservoir: ' + str(rselect))

# Reservoir release plot sum of all
fig, ax = plt.subplots(figsize=[20, 10])

total_release = np.zeros(len(optRelease[list(optRelease.keys())[0]]))

for Aname in optRelease.keys():
    total_release += optRelease[Aname]

ax.bar(optRelease[list(optRelease.keys())[0]].keys(), total_release)
ax.set_xlabel('time step')
ax.set_ylabel('Total reservoir release')
ax.set_title('Total release from all reservoirs')
plt.show()

# Sum of all three water demands

SumDem = dict()
for key in DomDem.keys():
    SumDem[key] = IndDem[key] + AvAgDempl[key] + DomDem[key]


SumA = dict()
for key in AvoptADom.keys():
    SumA[key] = AvoptADom[key] + AvoptAAg[key] + AvoptAInd[key]

# Add to DataFrames and export to csv
SumDem_pandas = pd.DataFrame.from_dict(SumDem, orient = 'index',columns=['Demand [m3]'])
SumDem_pandas.to_csv(savepath + os.sep + 'SumDemand.csv',index_label='ID')

SumA_pandas = pd.DataFrame.from_dict(SumA, orient = 'index',columns=['Allocation [m3]'])
SumA_pandas.to_csv(savepath + os.sep + 'SumAllocation.csv',index_label='ID')

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 10:40:05 2023

@author: magnu
"""

import os
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
assets_char = pd.read_excel(os.path.join(datafolder, 'Assets_CPY_input_zero.xlsx'))

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
WTPPow = 100*37 # THB /MWh
ThaChinDiv = 0.5 #ThaChin diversion, i.e. the fraction of the flow downstream of Upper Chao Phraya catchment that is diverted into Tha Chin. Fraction (dimensionless)


savepath = r'test_savepath' #adust this path to write results in specific folder on your system

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
    IndDem[c] = 0*scatch_people[c]*per_cap_ind_demand/12/1E6 # industrial demand million m3 per month = number of people times per capita demand
    DomDem[c] = 0*scatch_people[c]*per_cap_dom_demand/12/1E6 # domestic demand million m3 per month = number of people times per capita demand
    AgDempl[c]=dict() # create one empty sub-dictionary per catchment
    ROpl[c]=dict() # create one empty sub-dictionary per catchment
    ROindividual[c] = dict()
    for t in ntimes: # loop through all time steps
        tstamp = ntsdic[t] # look up the time stamp corresponding to the time step
        irrigation_rate = (Kc(tstamp)*pet_rate[c][tstamp]*365/12 - prec_rate[c][tstamp])*(1+saltLR)/basineff #Calculate the irrigation rate
        if irrigation_rate < 0: # if rainfall exceeds crop water demand
            irrigation_rate = 0 # set irrigation rate to zero
        AgDem[(c,t)] = 0*irrigation_rate/1000*scatch_airr[c]/100 # ag. demand dependent on time in million m3 per month
        RO[(c,t)] = runoff_rate[c][tstamp]*365/12/1000*scatch_areas[c] # Runoff generated in each catchment in million m3 per month
        AgDempl[c][t] = 0*irrigation_rate/1000*scatch_airr[c]/100 # same values stored in nested dictionary
        ROpl[c][t] = runoff_rate[c][tstamp]*365/12/1000*scatch_areas[c] # same values stored in nested dictionary
        ROindividual[c][t] = runoff_rate[c][tstamp]*365/12/1000 # m/month

# Reservoir data
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

# Environmental flow requirements
# Mean annual runoff
#MAR = {c: np.mean(list(ROpl[c].values())) for c in ROpl}

# Desired ecosystem status
# Change: poor = 0, fair = 10, good = 25, natural = 50   
#eco_stat = 0.50

# # Low flow requirement (LFR), high flow requirement (HFR), and environmental flow requirement (EFR)
# RO_sort = {c: np.sort(list(ROpl[c].values()))[::-1] for c in ncatch}
# exceed = {c: np.arange(1, len(ROpl_sort[c])+1)/len(ROpl_sort[c]) for c in ncatch}
# LFR = {c: np.percentile(list(ROpl[c].values()),eco_stat) for c in ncatch}
# HFR = {}
# EFR = {}

# for c in ncatch:
#      HFR_90 = np.percentile(list(ROpl[c].values()),90)
#      if HFR_90 <= 0.1*MAR[c]:
#          HFR[c] = 0.2*MAR[c]
#      elif HFR_90 <= 0.2*MAR[c]:
#          HFR[c] = 0.15*MAR[c]
#      elif HFR_90 <= 0.3*MAR[c]:
#          HFR[c] = 0.07*MAR[c]
#      else:
#          HFR[c] = 0
#      EFR[c] = LFR[c]+HFR[c]
# print(EFR)


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
#model.AEFR = Var(model.ncatch, model.ntimes, within=NonNegativeReals) # EFR allocation

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
#model.EFRDem  = Param(model.ncatch,within=NonNegativeReals,initialize = EFR)
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

# Environmental flow requirement constraint
# def wd_EFR_c(model, nc, nt):
#     return model.AEFR[nc,nt] >= model.EFRDem[nc]
# model.wd_EFR = Constraint(model.ncatch, model.ntimes, rule=wd_EFR_c)

# def wd_EFR_c(model, nc, nt):
#     return model.Qds[nc,nt] >= model.EFRDem[nc]
# model.wd_EFR = Constraint(model.ncatch, model.ntimes, rule=wd_EFR_c)


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
print("Agricultural benefit", round(value(ag_ben)/(len(model.ntimes)/12)/1000,2), " billion THB per year" )
print("Domestic benefit", round(value(dom_ben)/(len(model.ntimes)/12)/1000,2), " billion THB per year" )
print("Industry benefit", round(value(ind_ben)/(len(model.ntimes)/12)/1000,2), " billion THB per year" )
print("Power benefit", round(value(pow_ben)/(len(model.ntimes)/12)/1000,2), " billion THB per year" )

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


###############MAR for our model

# Initialize a new dictionary to store average runoff for all years for each catchment
MAR_optOF = {}

num_years = 29

# Calculate the sum of each year and the average for all years for each catchment
for catchment in optOF.columns:
    total_sum = 0
    for year in range(1, num_years + 1):
        yearly_sum = 0
        for month in range(1, 13):
            index = (year - 1) * 12 + month - 1  # subtract 1 to match DataFrame index
            if index < len(optOF) and index in optOF.index:
                yearly_sum += optOF.loc[index, catchment] # Access the catchment value in the row
        total_sum += yearly_sum
        
    average = total_sum / num_years
    MAR_optOF[catchment] = average

print(MAR_optOF)

total_MAR_optOF = sum(MAR_optOF.values()) / len(MAR_optOF)
print(total_MAR_optOF)



###################Flow duration curve
# Combine all catchment data into a single list
flow_data = []
for catchment in optOF.columns:
    flow_data.extend(optOF[catchment].values)

# Sort the data in descending order
flow_data = np.array(flow_data)
flow_data_sorted = np.sort(flow_data)[::-1]

# Calculate the exceedance probability for each flow value
n = len(flow_data_sorted)
exceedance_probabilities = np.arange(1, n + 1) / (n + 1)

# Create the plot
plt.figure(figsize=(10,5))
plt.plot(exceedance_probabilities * 100, flow_data_sorted)
plt.xlim(0, 100)
plt.xlabel("Exceedance Probability (%)", fontsize = 10)
plt.ylabel("Flow [MCM]", fontsize = 10)
plt.title("Flow Duration Curve", fontsize = 10)

# Display the plot
plt.show()

########LFR and HFR

# Calculate percentiles for LFR
LFR_natural = np.percentile(flow_data_sorted, 50)*12
LFR_good = np.percentile(flow_data_sorted, 25)*12
LFR_fair = np.percentile(flow_data_sorted, 10)*12

# Calculate HFR
HFR = 0.2 * total_MAR_optOF

# Calculate EWR
EWR_natural = LFR_natural + HFR
EWR_good = LFR_good + HFR
EWR_fair = LFR_fair + HFR

# Create a pandas DataFrame with the desired values
table = pd.DataFrame({
    'LFR': [LFR_natural, LFR_good, LFR_fair],
    'HFR': [HFR, HFR, HFR],
    'EWR': [EWR_natural, EWR_good, EWR_fair]
}, index=['Natural', 'Good', 'Fair'])

print(table)

print('MAR:',total_MAR_optOF)



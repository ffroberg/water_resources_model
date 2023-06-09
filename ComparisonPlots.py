

# FIRST RUN ONE OF THE SC1-3 FILES TO BE ABLE TO USE THE "VALUE"-FUNCTION

import matplotlib.pyplot as plt
from matplotlib import ticker
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# Import total_release from scenarios
from SC1_WRS_model import total_spill as total_Spill1, ntimes, ntimestamps
from SC2_WRS_model import total_spill as total_Spill2
from SC3_WRS_model import total_spill as total_Spill3, monsoon

# Import benefit
from SC1_WRS_model import ag_ben as ag_ben1, ind_ben as ind_ben1, dom_ben as dom_ben1, pow_ben as pow_ben1
from SC2_WRS_model import ag_ben as ag_ben2, ind_ben as ind_ben2, dom_ben as dom_ben2, pow_ben as pow_ben2
from SC3_WRS_model import ag_ben as ag_ben3, ind_ben as ind_ben3, dom_ben as dom_ben3, pow_ben as pow_ben3

# Import Power generation
from SC1_WRS_model import AVpow_gen as AVpow_gen1, SUMpow_gen as SUMpow_gen1
from SC2_WRS_model import AVpow_gen as AVpow_gen2, SUMpow_gen as SUMpow_gen2
from SC3_WRS_model import AVpow_gen as AVpow_gen3, SUMpow_gen as SUMpow_gen3

# Shadow price water + capacity
from SC1_WRS_model import SPResCap as SPResCap1, SPCWB as SPCWB1
from SC2_WRS_model import SPResCap as SPResCap2, SPCWB as SPCWB2
from SC3_WRS_model import SPResCap as SPResCap3, SPCWB as SPCWB3

datafolder = os.path.relpath(r'Data')

# Download precipitation data
prec = pd.read_excel(os.path.join(datafolder,'CRU_PRE_CPY_1991_2020.xlsx'))# EXCEL File with rainfall time series per subcatchment in mm/month
pet = pd.read_excel(os.path.join(datafolder,'CRU_PET_CPY_1991_2020.xlsx'), sheet_name = 1, skiprows= 1)
grun = pd.read_excel(os.path.join(datafolder, 'G-RUN_CPY_1990_2019.xlsx'), sheet_name=1, skiprows=1)

prec_filtered = prec[prec['Month'].isin(ntimestamps)]
prec_average = prec_filtered.mean(axis = 1)


pet_filtered = pet[pet['Month'].isin(ntimestamps)]
pet_average = pet_filtered.mean(axis = 1)


grun_filtered = grun[grun['Month'].isin(ntimestamps)]
grun_average = grun_filtered.mean(axis = 1)


#make plots bigger
plt.rc('font', size=15)

### Remaking ntimes to a list of all the month names
month_names = [date.strftime("%b") for date in ntimestamps]
years = 4

###############################
### Plot precipitation (average over all reservoirs), and total reservoir spill for 3 scenarios
fig, ax1 = plt.subplots(figsize =(15,5))
# plot the three graphs on the first y-axis
ax1.plot(ntimes, total_Spill1, label = 'Baseline')
ax1.plot(ntimes, total_Spill2, label = 'with KST')
ax1.plot(ntimes, total_Spill3, label = 'with KST and FS')
# customize the first y-axis
ax1.set_ylabel('Total resevoir spill [MCM]')
ax1.tick_params(axis='y')
# create a secondary y-axis
ax2 = ax1.twinx()
ax2.plot(ntimes, prec_average, color='darkblue', linestyle = '--', label = 'Average precipitation [mm/month]')
# customize the secondary y-axis
ax2.set_ylabel('Average Precipitation [mm/month]')
ax2.tick_params(axis='y')
# add a blue shaded region from July to October
for i in range(years):
    plt.axvspan(7+i*12, 10+i*12, facecolor = 'blue', alpha = 0.2)
ax1.legend(loc='upper left', bbox_to_anchor=(0.0, 0.9))
ax2.legend(loc='upper left', bbox_to_anchor=(0.0, 1.0))
ax1.set_xticks(ntimes)
ax1.set_xticklabels(month_names, rotation=45, fontsize  = 10)
plt.xlim(0,years*12)
plt.subplots_adjust(bottom=0.15)
plt.show()

###############################
# Benefit for all
scenario1 = np.array([value(ag_ben1), value(ind_ben1), value(dom_ben1), value(pow_ben1)])
scenario2 = np.array([value(ag_ben2), value(ind_ben2), value(dom_ben2), value(pow_ben2)])
scenario3 = np.array([value(ag_ben3), value(ind_ben3), value(dom_ben3), value(pow_ben3)])

plt.figure(figsize=(15,5))
barWidth = 0.15
r1 = np.arange(len(scenario1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
plt.bar(r1, scenario1, width=barWidth, edgecolor='white', label='Baseline')
plt.bar(r2, scenario2, width=barWidth, edgecolor='white', label='with KST')
plt.bar(r3, scenario3, width=barWidth, edgecolor='white', label='with KST +FS')

plt.xlabel('Benefit')
plt.xticks([r + barWidth for r in range(len(scenario1))], ['Agriculture', 'Industry', 'Domestic', 'Power'])
# add ylabel and title
plt.ylabel('billion THB per year')
#plt.title('Values for different variables in three different scenarios')
plt.legend()
plt.show()


###############################
# Power generation

plt.figure(figsize=(15,5))
plt.plot(ntimes, SUMpow_gen1, label = 'Baseline')
plt.plot(ntimes, SUMpow_gen2, label = 'with KST')
plt.plot(ntimes, SUMpow_gen3, label = 'with KST and FS')
plt.xticks(ntimes, month_names, rotation=45, fontsize  = 10)
for i in range(years):
    plt.axvspan(7+i*12, 10+i*12, facecolor = 'blue', alpha = 0.2)
plt.legend(loc='upper right')
plt.ylabel('Total power generation [kWh]')
plt.xlim(0,years*12)
plt.show()

plt.figure(figsize=(15,5))
plt.plot(ntimes, AVpow_gen1, label = 'Baseline')
plt.plot(ntimes, AVpow_gen2, label = 'with KST')
plt.plot(ntimes, AVpow_gen3, label = 'with KST and FS')
plt.xticks(ntimes, month_names, rotation=45, fontsize  = 10)
for i in range(years):
    plt.axvspan(7+i*12, 10+i*12, facecolor = 'blue', alpha = 0.2)
plt.legend(loc='upper right')
plt.ylabel('Mean power generation [kWh]')
plt.xlim(0,years*12)
plt.show()


###############################
# Shadow price water + capacity

avSPResCap1 = SPResCap1.mean(axis = 1)
avSPResCap2 = SPResCap2.mean(axis = 1)
avSPResCap3 = SPResCap3.mean(axis = 1)

avSPCWB1 = SPCWB1.mean(axis = 1)
avSPCWB2 = SPCWB2.mean(axis = 1)
avSPCWB3 = SPCWB3.mean(axis = 1)

plt.figure(figsize=(15,5))
plt.plot(ntimes, avSPResCap1, label ='Baseline')
plt.plot(ntimes, avSPResCap2, label ='KST')
plt.plot(ntimes, avSPResCap3, label ='KST + FS')
plt.xticks(ntimes, month_names, rotation=45, fontsize  = 10)
for i in range(years):
    plt.axvspan(7+i*12, 10+i*12, facecolor = 'blue', alpha = 0.2)
plt.legend(loc='upper right')
plt.ylabel('Reservoir capacity shadow price THB per m3')
plt.xlim(0,years*12)
plt.show()

plt.figure(figsize=(15,5))
plt.plot(ntimes, avSPCWB1, label ='Baseline')
plt.plot(ntimes, avSPCWB2, label ='KST')
plt.plot(ntimes, avSPCWB3, label ='KST + FS')
plt.xticks(ntimes, month_names, rotation=45, fontsize  = 10)
for i in range(years):
    plt.axvspan(7+i*12, 10+i*12, facecolor = 'blue', alpha = 0.2)
plt.legend(loc='upper right')
plt.ylabel('Water shadow price THB per m3')
plt.xlim(0,years*12)
plt.show()




###############################
# Deficit for all 
# from SC1_WRS_model import optDInd as optDInd1, optDAg as optDAg1, optDDom as optDDom1, optOF as optOF1
# from SC2_WRS_model import optDInd as optDInd2, optDAg as optDAg2, optDDom as optDDom2, optOF as optOF2
# from SC3_WRS_model import optDInd as optDInd3, optDAg as optDAg3, optDDom as optDDom3, optOF as optOF3

# # Gather all deficit for agriculture, domestic and industry and find mean across all catchments
# deficit_sum1 = (optDInd1.mean(axis=1) + optDAg1.mean(axis = 1) + optDDom1.mean(axis = 1))/3
# deficit_sum2 = (optDInd2.mean(axis=1) + optDAg2.mean(axis = 1) + optDDom2.mean(axis = 1))/3
# deficit_sum3 = (optDInd3.mean(axis=1) + optDAg3.mean(axis = 1) + optDDom3.mean(axis = 1))/3

# plt.figure(figsize=(15,5))
# plt.plot(ntimes, deficit_sum1, label = 'Baseline')
# plt.plot(ntimes, deficit_sum2, label = 'with KST')
# plt.plot(ntimes, deficit_sum3, label = 'with KST and FS')
# plt.xticks(ntimes, month_names, rotation=45, fontsize  = 10)
# for i in range(years):
#     plt.axvspan(7+i*12, 10+i*12, facecolor = 'blue', alpha = 0.2)
# plt.legend()
# plt.ylabel('Mean deficit [MCM]')
# plt.xlim(0,12*years)
# plt.show()

# plt.figure(figsize=(15,5))
# plt.plot(ntimes, optOF1.mean(axis = 1), label = 'Baseline')
# plt.plot(ntimes, optOF2.mean(axis = 1), label = 'with KST')
# plt.plot(ntimes, optOF3.mean(axis = 1), label = 'with KST and FS')
# plt.xticks(ntimes, month_names, rotation=45, fontsize  = 10)
# for i in range(years):
#     plt.axvspan(7+i*12, 10+i*12, facecolor = 'blue', alpha = 0.2)
# plt.legend()
# plt.ylabel('Outflow [MCM]')
# plt.xlim(0,years*12)
# plt.show()


# ###############################
# # Precipitation for all catchments with monsoon
# fig, axes = plt.subplots(nrows=13, ncols=1, figsize = (10,25), constrained_layout=True)
# for i, key in enumerate(prec_filtered.columns):
#     if key == 'Month':
#         continue
#     ax = axes[i-1]
#     ax.plot(prec_filtered[key], label = key)
#     ax.legend(loc = 'upper right', fontsize = 10)
#     for j in prec_filtered.index:
#         if j in monsoon and j-1 not in monsoon:
#             ax.axvspan(j,j+4, facecolor = 'blue', alpha = 0.2)

    
# fig.text(0.04, 0.5, '[mm/month]', ha='center', va='center', rotation='vertical', fontsize = 10)
# plt.xlabel('Time [month]')
# plt.show()





###############################
# Average precipitation
fig, ax1 = plt.subplots(figsize=(15,5))
ax1.plot(ntimes, prec_average, label = 'Precipitation')
ax1.plot(ntimes, pet_average, label = 'PET')
ax1.set_ylabel('[mm/month]')

ax2 = ax1.twinx()
ax2.plot(ntimes, grun_average, label = 'G-RUN', color = 'green')
ax2.set_ylabel('[$m^3$/s]')
for j in prec_filtered.index:
    if j in monsoon and j-1 not in monsoon:
        plt.axvspan(j,j+4, facecolor = 'blue', alpha = 0.2)
ax1.legend(loc = 'upper right')
ax2.legend(loc = 'upper right', bbox_to_anchor = (1.0, 0.80))

plt.xlabel('time [Month]')
plt.show()



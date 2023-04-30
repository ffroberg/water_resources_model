
import matplotlib.pyplot as plt
from matplotlib import ticker
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# Import total_release from scenarios
from SC1_WRS_model import total_release as total_release1, total_spill as total_Spill1
from SC2_WRS_model import total_release as total_release2, total_spill as total_Spill2
from SC3_WRS_model import total_release as total_release3, total_spill as total_Spill3

from SC1_WRS_model import ntimes, ntimestamps

from SC3_WRS_model import monsoon

datafolder = os.path.relpath(r'Data')

# Download precipitation data
prec = pd.read_excel(os.path.join(datafolder,'CRU_PRE_CPY_1991_2020.xlsx'))# EXCEL File with rainfall time series per subcatchment in mm/month
prec_filtered = prec[prec['Month'].isin(ntimestamps)]
prec_average = prec_filtered.mean(axis = 1)

### Remaking ntimes to a list of all the month names
month_names = [date.strftime("%b") for date in ntimestamps]

# Plot data and set x-tick labels to January of each year
plt.plot(ntimes, total_release1, label='1')
plt.plot(ntimes, total_release2, label='2')
plt.plot(ntimes, total_release3, label='3')
plt.xticks(ntimes, month_names, rotation=45)
plt.axvspan(7, 10, facecolor='blue', alpha=0.2)
plt.legend()
plt.xlim(0, 12)
plt.savefig('release_plot.png', bbox_inches='tight',pad_inches = 0.1,dpi=300)
plt.show()

### Plot precipitation (average over all reservoirs), and total reservoir spill for 3 scenarios
fig, ax1 = plt.subplots()
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
plt.axvspan(7, 10, facecolor='blue', alpha=0.2)
plt.axvspan(7+12, 10+12, facecolor='blue', alpha=0.2)
plt.axvspan(7+12*2, 10+12*2, facecolor='blue', alpha=0.2)
plt.axvspan(7+12*3, 10+12*3, facecolor='blue', alpha=0.2)
ax1.legend(loc='upper left', bbox_to_anchor=(0.0, 0.92))
ax2.legend(loc='upper left', bbox_to_anchor=(0.0, 1.0))
ax1.set_xticks(ntimes)
ax1.set_xticklabels(month_names, rotation=45, fontsize  = 8)
plt.xlim(0,36)
plt.savefig('spill_plot.png', bbox_inches='tight',pad_inches = 0.1, dpi=300)
plt.subplots_adjust(bottom=0.15)
plt.show()

# Deficit for all 
from SC1_WRS_model import optDInd as optDInd1, optDAg as optDAg1, optDAg, optDDom as optDDom1, optOF as optOF1
from SC2_WRS_model import optDInd as optDInd2, optDAg as optDAg2, optDAg, optDDom as optDDom2, optOF as optOF2
from SC3_WRS_model import optDInd as optDInd3, optDAg as optDAg3, optDAg, optDDom as optDDom3, optOF as optOF3

# Gather all deficit for agriculture, domestic and industry and find mean across all catchments
deficit_sum1 = (optDInd1.mean(axis=1) + optDAg1.mean(axis = 1) + optDDom1.mean(axis = 1))/3
deficit_sum2 = (optDInd2.mean(axis=1) + optDAg2.mean(axis = 1) + optDDom2.mean(axis = 1))/3
deficit_sum3 = (optDInd3.mean(axis=1) + optDAg3.mean(axis = 1) + optDDom3.mean(axis = 1))/3

plt.plot(ntimes, deficit_sum1, label = 'Baseline')
plt.plot(ntimes, deficit_sum2, label = 'with KST')
plt.plot(ntimes, deficit_sum3, label = 'with KST and FS')
plt.xticks(ntimes, month_names, rotation=45, fontsize  = 8)
plt.axvspan(7, 10, facecolor='blue', alpha=0.2)
plt.axvspan(7+12, 10+12, facecolor='blue', alpha=0.2)
plt.axvspan(7+12+12, 10+12+12, facecolor='blue', alpha=0.2)
plt.legend()
plt.xlim(0,36)
plt.savefig('deficit_plot.png', bbox_inches='tight',pad_inches = 0.1,dpi=300)
plt.show()

plt.plot(ntimes, optOF1.mean(axis = 1), label = 'Baseline')
plt.plot(ntimes, optOF2.mean(axis = 1), label = 'with KST')
plt.plot(ntimes, optOF3.mean(axis = 1), label = 'with KST and FS')
plt.xticks(ntimes, month_names, rotation=45, fontsize  = 8)
plt.axvspan(7, 10, facecolor='blue', alpha=0.2)
#plt.axvspan(7+12, 10+12, facecolor='blue', alpha=0.2)
#plt.axvspan(7+12+12, 10+12+12, facecolor='blue', alpha=0.2)
plt.legend()
plt.xlim(0,12)
plt.savefig('runoff_plot.png', bbox_inches='tight',pad_inches = 0.1,dpi=300)
plt.show()


# Benefit for all
from SC1_WRS_model import ag_ben as ag_ben1, ind_ben as ind_ben1, dom_ben as dom_ben1, pow_ben as pow_ben1
from SC2_WRS_model import ag_ben as ag_ben2, ind_ben as ind_ben2, dom_ben as dom_ben2, pow_ben as pow_ben2
from SC3_WRS_model import ag_ben as ag_ben3, ind_ben as ind_ben3, dom_ben as dom_ben3, pow_ben as pow_ben3

scenario1 = np.array([value(ag_ben1), value(ind_ben1), value(dom_ben1), value(pow_ben1)])
scenario2 = np.array([value(ag_ben2), value(ind_ben2), value(dom_ben2), value(pow_ben2)])
scenario3 = np.array([value(ag_ben3), value(ind_ben3), value(dom_ben3), value(pow_ben3)])

# set the width of the bars
barWidth = 0.25

# set the positions of the bars on the x-axis
r1 = np.arange(len(scenario1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

# create the bar plot
plt.bar(r1, scenario1, width=barWidth, edgecolor='white', label='Baseline')
plt.bar(r2, scenario2, width=barWidth, edgecolor='white', label='with KST')
plt.bar(r3, scenario3, width=barWidth, edgecolor='white', label='with KST +FS')

# add xticks on the middle of the group bars
plt.xlabel('Benefit')
plt.xticks([r + barWidth for r in range(len(scenario1))], ['Agriculture', 'Industry', 'Domestric', 'Power'])
# add ylabel and title
plt.ylabel('billion THB per year')
#plt.title('Values for different variables in three different scenarios')

plt.legend()
plt.savefig('benefit_barplot.png', bbox_inches='tight',pad_inches = 0.1,dpi=300)
plt.show()


# Agriculture allocation
from SC1_WRS_model import optAAg as optAAg1
from SC2_WRS_model import optAAg as optAAg2
from SC3_WRS_model import optAAg as optAAg3

# Agriculture demand
from SC1_WRS_model import optDAg

mon_optAAg1 = np.mean(optAAg1.loc[monsoon])
mon_optAAg2 = np.mean(optAAg2.loc[monsoon])
mon_optAAg3 = np.mean(optAAg3.loc[monsoon])

dry_optAAg1 = np.mean(optAAg1.loc[~np.in1d(ntimes, monsoon)])
dry_optAAg2 = np.mean(optAAg2.loc[~np.in1d(ntimes, monsoon)])
dry_optAAg3 = np.mean(optAAg3.loc[~np.in1d(ntimes, monsoon)])

mon_optDAg = np.mean(optDAg.loc[monsoon])
dry_optDAg = np.mean(optDAg.loc[~np.in1d(ntimes, monsoon)])


mon_perc_Ag1 = (mon_optAAg1/mon_optDAg)*100

# optAAg1 = optAAg1.mean(axis = 1)
# optAAg2 = optAAg2.mean(axis = 1)
# optAAg3 = optAAg3.mean(axis = 1)


plt.plot(ntimes, optAAg1, label = 'Baseline')
plt.plot(ntimes, optAAg2, label = 'with KST')
plt.plot(ntimes, optAAg3, label = 'with KST and FS')
plt.xticks(ntimes, month_names, rotation=45, fontsize  = 8)
plt.axvspan(7, 10, facecolor='blue', alpha=0.2)
plt.axvspan(7+12, 10+12, facecolor='blue', alpha=0.2)
plt.axvspan(7+12+12, 10+12+12, facecolor='blue', alpha=0.2)
plt.legend(loc = 'upper right')
plt.xlim(0,36)
plt.savefig('Agriculture_Allocation.png', bbox_inches='tight',pad_inches = 0.1,dpi=300)
plt.show()

# Power generation
from SC1_WRS_model import AVpow_gen as AVpow_gen1, SUMpow_gen as SUMpow_gen1
from SC2_WRS_model import AVpow_gen as AVpow_gen2, SUMpow_gen as SUMpow_gen2
from SC3_WRS_model import AVpow_gen as AVpow_gen3, SUMpow_gen as SUMpow_gen3

plt.plot(ntimes, SUMpow_gen1, label = 'Baseline')
plt.plot(ntimes, SUMpow_gen2, label = 'with KST')
plt.plot(ntimes, SUMpow_gen3, label = 'with KST and FS')
plt.xticks(ntimes, month_names, rotation=45, fontsize  = 8)
plt.axvspan(7, 10, facecolor='blue', alpha=0.2)
plt.axvspan(7+12, 10+12, facecolor='blue', alpha=0.2)
plt.axvspan(7+12+12, 10+12+12, facecolor='blue', alpha=0.2)
plt.legend(loc='upper right')
plt.xlim(0,36)
plt.savefig('Power_Generation_sum.png', bbox_inches='tight',pad_inches = 0.1,dpi=300)
plt.show()

plt.plot(ntimes, AVpow_gen1, label = 'Baseline')
plt.plot(ntimes, AVpow_gen2, label = 'with KST')
plt.plot(ntimes, AVpow_gen3, label = 'with KST and FS')
plt.xticks(ntimes, month_names, rotation=45, fontsize  = 8)
plt.axvspan(7, 10, facecolor='blue', alpha=0.2)
plt.axvspan(7+12, 10+12, facecolor='blue', alpha=0.2)
plt.axvspan(7+12+12, 10+12+12, facecolor='blue', alpha=0.2)
plt.legend(loc='upper right')
plt.xlim(0,36)
plt.savefig('Power_Generation_AV.png', bbox_inches='tight',pad_inches = 0.1,dpi=300)
plt.show()




fig, axes = plt.subplots(nrows=len(prec_filtered.columns), ncols=1, figsize = (8,20), constrained_layout=True)
for i, key in enumerate(prec_filtered.columns):
    ax = axes[i]
    ax.plot(prec_filtered[key], label = key)
    ax.legend(loc = 'upper right', fontsize = 10)

    
fig.text(0.04, 0.5, '[MCM]', ha='center', va='center', rotation='vertical', fontsize = 10)
plt.xlabel('Time [month]')
plt.show()





# prec_filtered.set_index('Month', inplace=True)
# month_names = [date.strftime("%b") for date in grun.index]
# year_names = [date.strftime("%y") for date in grun.index]
# monsoon_names = ['Jul', 'Aug', 'Sep', 'Oct']

# # Create a figure with one column and 13 rows
# fig, axes = plt.subplots(nrows=13, ncols=1, figsize=(8, 20), constrained_layout = True)
# # Iterate over the number columns and plot each one as a subplot
# for i, col in enumerate(prec_filtered.columns):
#     ax = prec_filtered[col].plot(ax=axes[i], kind='bar', fontsize = 8)
#     ax.set_ylabel('MCM', fontsize = 8)
#     ax.set_xlabel('')

#     #xticks = ax.get_xticks()[::12]
#     #xticklabels = year_names[::12]
#     #ax.set_xticks(xticks)
#     ax.set_xticklabels(xticklabels, rotation=45, ha='right', fontsize = 8)
#     ax.legend(loc = 'upper right', fontsize = 10)
    

# # Add a common x-axis label and title
# fig.text(0.5, 0.08, 'Month', ha='center', fontsize = 10)
# fig.suptitle('Values by Month', fontsize=10)
# # Show the plot
# plt.show()




###### FROM SC1
# plt.plot(ntimes, optOF[7], label = 'Outflow')
# plt.plot(ntimes, optAAg[7], label = 'Ag. allocation')
# plt.plot(ntimes, optAInd[7], label = 'Ind. allocation')
# plt.plot(ntimes, optADom[7], label = 'Dom. allocation')
# plt.plot(ntimes, optSpill[1], label = 'spill')
# plt.plot(ntimes, ROpl[7].values(), label = 'Runoff')
# plt.axvspan(7, 10, facecolor='blue', alpha=0.2)
# plt.axvspan(7+12, 10+12, facecolor='blue', alpha=0.2)
# plt.axvspan(7+12+12, 10+12+12, facecolor='blue', alpha=0.2)
# plt.legend()
# plt.xlim(0,36)
# #plt.savefig('Agriculture_Allocation.png', bbox_inches='tight',pad_inches = 0.1,dpi=300)
# plt.show()
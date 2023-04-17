# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 11:45:30 2023

@author: magnu
"""

import matplotlib.pyplot as plt
from matplotlib import ticker
import pandas as pd
import numpy as np
import os

# TEST
a = 3


# Import data
datafolder=os.path.relpath(r'Data')
MCA = pd.read_excel(os.path.join(datafolder, 'Multivariate_table_short.xlsx'))
MCA

cols = ['Flood safety [MCM]', 'Demand meet [%]', 'Total benefit [million THB/year]', 'Environmental impact']
x = [i for i, _ in enumerate(cols)]

# Create (X-1) sublots along x axis
fig, axes = plt.subplots(1, len(x)-1, sharey=False, figsize=(15,5))

# Get min, max and range for each column
# Normalize the data for each column
min_max_range = {}
MCAnorm = MCA.copy()
for col in cols:
    min_max_range[col] = [MCAnorm[col].min(), MCAnorm[col].max(), np.ptp(MCAnorm[col])]
    MCAnorm[col] = np.true_divide(MCAnorm[col] - MCAnorm[col].min(), np.ptp(MCAnorm[col]))

    # Plot each row
for i, ax in enumerate(axes):
    for idx in MCA.index:
        ax.plot(x, MCAnorm.loc[idx, cols],label=MCAnorm['Unnamed: 0'][idx])
        handles, labels = ax.get_legend_handles_labels()
    ax.set_xlim([x[i], x[i+1]])
    
    
# Set the tick positions and labels on y axis for each plot
# Tick positions based on normalised data
# Tick labels are based on original data
def set_ticks_for_axis(dim, ax, ticks):
    min_val, max_val, val_range = min_max_range[cols[dim]]
    step = val_range / float(ticks-1)
    tick_labels = [round(min_val + step * i, 2) for i in range(ticks)]
    norm_min = MCAnorm[cols[dim]].min()
    norm_range = np.ptp(MCAnorm[cols[dim]])
    norm_step = norm_range / float(ticks-1)
    ticks = [round(norm_min + norm_step * i, 2) for i in range(ticks)]
    ax.yaxis.set_ticks(ticks)
    ax.set_yticklabels(tick_labels)

for dim, ax in enumerate(axes):
    ax.xaxis.set_major_locator(ticker.FixedLocator([dim]))
    set_ticks_for_axis(dim, ax,ticks=6)
    ax.set_xticklabels([cols[dim]])

# Move the final axis' ticks to the right-hand side
ax = plt.twinx(axes[-1])
dim = len(axes)
ax.xaxis.set_major_locator(ticker.FixedLocator([x[-2], x[-1]]))
set_ticks_for_axis(dim, ax, ticks=6)
ax.set_xticklabels([cols[-2], cols[-1]])
ax.legend(handles, labels, loc='center right')

# Remove space between subplots
plt.subplots_adjust(wspace=0)

# Add legend to plot

plt.show()
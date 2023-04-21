
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

from SC1_WRS_model import ntimes

### Remaking ntimes to a list of all the month names
# define start and end dates
start_date = datetime(1990, 1, 1)
end_date = start_date + relativedelta(months=+348)

# generate a sequence of dates at the start of each month
datetime_list = []
current_date = start_date
while current_date < end_date:
    datetime_list.append(current_date)
    current_date += relativedelta(months=+1)

month_names = [date.strftime("%b") for date in datetime_list]

# Plot data and set x-tick labels to January of each year
plt.plot(ntimes, total_release1, label='1')
plt.plot(ntimes, total_release2, label='2')
plt.plot(ntimes, total_release3, label='3')
plt.xticks(ntimes, month_names, rotation=45)
plt.axvspan(7, 10, facecolor='blue', alpha=0.2)
plt.legend()
plt.xlim(0, 12)
plt.show()

plt.plot(ntimes, total_Spill1, label = '1')
plt.plot(ntimes, total_Spill2, label = '2')
plt.plot(ntimes, total_Spill3, label = '3')
plt.xticks(ntimes, month_names, rotation=45)
plt.axvspan(7, 10, facecolor='blue', alpha=0.2)
plt.legend()
plt.xlim(0,12)
plt.show()



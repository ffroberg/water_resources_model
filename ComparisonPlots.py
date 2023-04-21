
import matplotlib.pyplot as plt
from matplotlib import ticker
import pandas as pd
import numpy as np
import os

# Import total_release from scenarios
from SC1_WRS_model import total_release as total_release1
from SC2_WRS_model import total_release as total_release2
from SC3_WRS_model import total_release as total_release3

from SC1_WRS_model import ntimes

plt.plot(ntimes, total_release1)
plt.plot(ntimes, total_release2)
plt.plot(ntimes, total_release3)
plt.xlim(0,24)
plt.show()
import os
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Set Working directory
datafolder=os.path.relpath(r'Data')

scatch_char = pd.read_excel(os.path.join(datafolder,'Subcatchments_CPY_input.xlsx')) # EXCEL File with subcatchment characteristics
prec = pd.read_excel(os.path.join(datafolder,'CRU_PRE_CPY_1991_2020.xlsx'))# EXCEL File with rainfall time series per subcatchment in mm/month
pet = pd.read_excel(os.path.join(datafolder,'CRU_PRE_CPY_1991_2020.xlsx'))
grun = pd.read_excel(os.path.join(datafolder, 'Subcatchments_CPY_input.xlsx'))
connectivity = pd.read_excel(os.path.join(datafolder, 'G-RUN_CPY_1990_2019.xlsx'))
assests_char = pd.read_excel(os.path.join(datafolder, 'Assets_CPY_input.xlsx'))





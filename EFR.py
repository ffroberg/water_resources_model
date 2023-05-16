# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 09:38:43 2023

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
from SC0 import ncatch
from SC0 import ntimes
from SC0 import optOF


datafolder=os.path.relpath(r'Data')

scatch_char = pd.read_excel(os.path.join(datafolder,'Subcatchments_CPY_input.xlsx')) # EXCEL File with subcatchment characteristics
prec = pd.read_excel(os.path.join(datafolder,'CRU_PRE_CPY_1991_2020.xlsx'))# EXCEL File with rainfall time series per subcatchment in mm/month
pet = pd.read_excel(os.path.join(datafolder,'CRU_PET_CPY_1991_2020.xlsx'))
grun = pd.read_excel(os.path.join(datafolder, 'G-RUN_CPY_1990_2019.xlsx'))
connectivity = pd.read_excel(os.path.join(datafolder, 'CPY_catch_connectivity.xlsx'))
assets_char = pd.read_excel(os.path.join(datafolder, 'Assets_CPY_input_zero.xlsx'))

MAR = {c: optOF[c].mean() for c in optOF}

#Desired ecosystem status
#Change: poor = 0, fair = 10, good = 25, natural = 50   
eco_stat = 0.5

# Low flow requirement (LFR), high flow requirement (HFR), and environmental flow requirement (EFR)
optOF_sort = {c: np.sort(optOF[c])[::-1] for c in ncatch}
exceed = {c: np.arange(1, len(optOF_sort[c])+1)/len(optOF_sort[c]) for c in ncatch}
LFR = {c: optOF[c].quantile(eco_stat) for c in optOF.columns}
HFR = {}
EFR = {}

for c in ncatch:
      HFR_90 = np.percentile(optOF[c], 90)
      if HFR_90 <= 0.1*MAR[c]:
          HFR[c] = 0.2*MAR[c]
      elif HFR_90 <= 0.2*MAR[c]:
          HFR[c] = 0.15*MAR[c]
      elif HFR_90 <= 0.3*MAR[c]:
          HFR[c] = 0.07*MAR[c]
      else:
          HFR[c] = 0
      EFR[c] = LFR[c]+HFR[c]
print(EFR)

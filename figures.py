# -*- coding: utf-8 -*-
"""
Created on Mon May 30 09:44:18 2022

@author: giamm
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

#%%
# file setup
basepath = Path(__file__).parent
output = "Output"
filename = "katokoba_100.csv"
# plot setup
step_norm = 1e4 #step normalisation in plot
ymin = 0.3 #minimum y value in plot
ymax = 1 #maximum y values in plot
fgs = (12, 10) #figure size (inch)
fts = 30 #fontsize
#%%
# input data
data = pd.read_csv(basepath/output/filename,
                   sep=';')
rwds = data.values[:,1:]
#%%
# data processing
rmax = np.max(rwds)
rwds = rwds / rmax
rwds_min = np.min(rwds, axis=0)
rwds_25 = np.percentile(rwds, 25, axis=0)
rwds_mean = np.mean(rwds, axis=0)
rwds_75 = np.percentile(rwds, 75, axis=0)
rwds_max = np.max(rwds, axis=0)
#%%
# plotting
steps = np.arange(rwds_mean.size) / step_norm
fig, ax = plt.subplots(figsize=fgs)
ax.fill_between(x=steps, y1=rwds_min, y2=rwds_max,
                color='lightsteelblue', alpha=1)
ax.fill_between(x=steps, y1=rwds_25, y2=rwds_75,
                color='yellow', alpha=0.5)
ax.plot(steps, rwds_mean, color='tab:blue')
ax.set_ylim([ymin, ymax])
ax.set_xlim(steps[[0, -1]])
ax.set_xlabel("Training step (x 10^{:.0f})".format(np.log10(step_norm)),
              fontsize=fts)
ax.set_ylabel("Normalised reward", fontsize=fts)
ax.tick_params(labelsize=fts)
ax.grid()


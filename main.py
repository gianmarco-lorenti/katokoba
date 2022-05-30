# -*- coding: utf-8 -*-
"""
Created on Mon May 30 10:24:38 2022

@author: giamm
"""

"""
Main for Kato & Kobaiashi procedure (Immune System's model through 
Reinforcement Learning).
"""
# libraries and modules
import numpy as np
import matplotlib.pyplot as plt
from katokoba import KatoKoba

# %%
# main one-shot
# build and run model
model = KatoKoba()
model.run()
# get output values
r_hist = model.r_hist[:] #reward history
n = model.agent.n #th-clones size distribution at stationary

# iterrs = 1
# r_hists = np.zeros((iterrs, tmax))
# if __name__ == '__main__':
#     for iterr in range(iterrs):
#         model = KatoKoba()
#         model.run()
#         r_hist = model.r_hist[:]
#         n = model.agent.n
#         r_hists[iterr, :] = r_hist
#         print(iterr)
        
# # data = pd.DataFrame(r_hists)
# # data.to_csv("r_hists_{}.csv".format(iterrs), sep=';')
# rwds = np.mean(r_hists, axis=0)
# plt.plot(rwds)

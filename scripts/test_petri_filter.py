# find high frequency noise

import matplotlib.pyplot as plt
import h5py
import numpy as np
#f = h5py.File("/data/kaggle-petri/Salmonella_EColi_10strains_3replicates/Segmented cubes/E Coli/E_Coli_9_PCA_24_D5_rep3-1_segmented.mat")
#d = f["CubeR_seg"]
f = h5py.File("/data/kaggle-petri/Salmonella_EColi_10strains_3replicates/Raw cubes/E Coli/E_Coli_9_PCA_24_D5_rep3-1.mat")
d = f["CubeR"]

i = 616
plt.plot(d[:,286,i])
plt.plot(d[:,286,i]-np.mean(d[:,286,i-1:i+1],axis=-1))

plt.show()
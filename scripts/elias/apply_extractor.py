import pickle as pkl
import numpy as np
from extract_nodules import extract_nodules_best_kmeans

folder = 'storage/metadata/dsb3/model-predictions/ikorshun/luna_scan_v3_dice-20170202-154805/'

patient_id = '1.3.6.1.4.1.14519.5.2.1.6279.6001.121391737347333465796214915391' 
# there is only one cancerous nodule in this patient
# original coos
# [(49.31293635, -63.21521025, -118.7995619, 22.13322034)]
# vox coos
# [array([ 312.98274683,  169.22729589,  154.76035048])]
# diameter
# [22.13322034]

# load in predicted segmentation
pred = pkl.load(open(folder+'pred_'+patient_id+'.pkl', 'rb' ))
print pred.shape

extract_nodules_best_kmeans(pred[0,0])
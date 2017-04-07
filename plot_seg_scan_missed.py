import glob
import os
import sys
import numpy as np
import blobs_detection
import pathfinder
import utils
import utils_lung
import utils_plots
from configuration import set_configuration
import data_transforms
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    sys.exit("Usage: evaluate_luna_scan.py <configuration_name>")

config_name = sys.argv[1]
set_configuration('configs_seg_scan', config_name)

predictions_dir = utils.get_dir_path('model-predictions', pathfinder.METADATA_PATH)
outputs_path = predictions_dir + '/%s' % config_name

blob_files = sorted(glob.glob(outputs_path + '/*.pkl'))
# print blob_files

data_path=pathfinder.LUNA_DATA_PATH

pid2annotations = utils_lung.read_luna_annotations(pathfinder.LUNA_LABELS_PATH)

tp = 0
n_pos = 0
n_blobs = 0


analysis_dir=utils.get_dir_path('analysis', pathfinder.METADATA_PATH) 
outputs_path = analysis_dir + '/%s/'%config_name
utils.auto_make_dir(outputs_path)
        


for p in blob_files:
    pid = utils_lung.extract_pid_filename(p, '.pkl')
    blobs = utils.load_pkl(p)
    n_blobs += len(blobs)
    print pid
    print 'n_blobs', len(blobs)
    print 'tp / n pos ', int(np.sum(blobs[:, -1])), len(pid2annotations[pid])

    
    
    if int(np.sum(blobs[:, -1])) < len(pid2annotations[pid]):
        print '-------- HERE!!!!!! ------------'

        annotations=pid2annotations[pid]

        # print out annotations:
        print "annotations: {}".format(pid2annotations[pid])
        print "pid: {}".format(pid)
        print "p: {}".format(p)
        
        # load the patient in question
        patient_path=data_path + '/' + pid + '.mhd'
        img, origin, pixel_spacing = utils_lung.read_mhd(patient_path)
        
        for idx,annotation in enumerate(annotations):
            image_coords = (annotation[:3] - origin)/pixel_spacing
            image_coords=[int(coord) for coord in image_coords]
            print "annotation: {}".format(annotation)
            print "image coords: {}".format(image_coords)
            fig = plt.figure()
            #plt.imshow(img[0,:,:])
            
            img_to_write=img[image_coords[0],:,:]

            circ_mask=np.zeros(img_to_write.shape)
            
            print "saving to  {}".format(outputs_path)
            plt.imsave(outputs_path+'%s_notfound_%s.png'%(pid,idx),img[image_coords[0],:,:])

            
            
        print "img shape: {}".format(img.shape)
        print "img origin: {}".format(origin)
        print "img pixelspacing: {}".format(pixel_spacing)
  
        
        #utils_plots.plot_slice_3d_3axis(input=img[0],
         #                           pid='missed_{}'.format(pid),
          #                          img_dir=outputs_path,
           #                         idx=np.array(x_chunk_train[0, 0].shape) / 2)

        

        



        
    tp += np.sum(blobs[:, -1])

    
    print '====================================='

print 'n patients', len(blob_files)
print 'TP', tp
print 'n blobs', n_blobs
print n_pos

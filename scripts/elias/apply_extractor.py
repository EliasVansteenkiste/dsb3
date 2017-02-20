import pickle as pkl
import numpy as np
import subprocess
from skimage.measure import label, regionprops
from extract_nodules import extract_nodules_blob_detection, extract_nodules_conv_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



def check_nodules_found_v3(patient_id, folder, no_rois=5, plot=False):
	# load in predicted segmentation
	pred = pkl.load(open(folder+'pred_'+patient_id+'.pkl', 'rb' ))
	# load in target
	target = pkl.load(open(folder+'tgt_'+patient_id+'.pkl', 'rb' ))
	# load in original 3D volume
	original = pkl.load(open(folder+'in_'+patient_id+'.pkl', 'rb' ))



	if plot:
		#only for masks with one nodule
		listofcoos = np.where(target[0,0]==1)
		center = [np.round(np.average(listofcoos[0])).astype(int), np.round(np.average(listofcoos[1])).astype(int), np.round(np.average(listofcoos[2])).astype(int)]

		fig = plt.figure()

		ax1 = fig.add_subplot(3,1,1)
		ax1.imshow(target[0,0,center[0],:,:].transpose())
		circ1 = plt.Circle((center[1],center[2]), 24, color='y', fill=False)
		ax1.add_patch(circ1)

		ax2 = fig.add_subplot(3,1,2)
		ax2.imshow(target[0,0,:,center[1],:])
		circ2 = plt.Circle((center[0],center[2]), 24, color='y', fill=False)
		ax2.add_patch(circ2)

		ax3 = fig.add_subplot(3,1,3)
		ax3.imshow(target[0,0,:,:,center[2]].transpose())
		circ3 = plt.Circle((center[0],center[1]), 24, color='y', fill=False)
		ax3.add_patch(circ3)
		fig.savefig('original_mask.jpg')

	
	extract_nodules_conv_filter(pred[0,0], original[0,0], no_rois=no_rois, dim=32, plot=False, dbg_target=target[0,0])

	labeled_target = label(target[0,0])
	regions = regionprops(labeled_target)
	print 'number of regions found in target', len(regions)
	return len(regions) # temporary



MAX_HU = 400.
MIN_HU = -1000.
def normHU2HU(x):
    """
    Modifies input data
    :param x:
    :return:
    """
    x = x * (MAX_HU - MIN_HU) + MIN_HU
    return x

def check_nodules_found_v4(patient_id, folder_in, folder_out, no_rois=10, plot=False):
	# load in predicted segmentation
	pred = np.load(folder_in+'pred_'+patient_id+'.npy',)
	# load in target
	target = np.load(folder_in+'tgt_'+patient_id+'.npy')
	# load in original 3D volume
	original = np.load(folder_in+'in_'+patient_id+'.npy')


	if plot:
		#only for masks with one nodule
		listofcoos = np.where(target==1)
		center = [np.round(np.average(listofcoos[0])).astype(int), np.round(np.average(listofcoos[1])).astype(int), np.round(np.average(listofcoos[2])).astype(int)]

		fig = plt.figure()

		ax1 = fig.add_subplot(3,1,1, adjustable='box', aspect=1.0)
		ax1.imshow(target[0,0,center[0],:,:].transpose())
		circ1 = plt.Circle((center[1],center[2]), 24, color='y', fill=False)
		ax1.add_patch(circ1)

		ax2 = fig.add_subplot(3,1,2, adjustable='box', aspect=1.0)
		ax2.imshow(target[0,0,:,center[1],:])
		circ2 = plt.Circle((center[0],center[2]), 24, color='y', fill=False)
		ax2.add_patch(circ2)

		ax3 = fig.add_subplot(3,1,3, adjustable='box', aspect=1.0)
		ax3.imshow(target[0,0,:,:,center[2]].transpose())
		circ3 = plt.Circle((center[0],center[1]), 24, color='y', fill=False)
		ax3.add_patch(circ3)
		fig.savefig('original_mask.jpg')

	
	#extract_nodules_conv_filter(pred, original, no_rois=no_rois, dim=32, plot=False, dbg_target=None, nodules=target)
	rois, centers = extract_nodules_blob_detection(pred, normHU2HU(original), patient_id, dim=32, plot=False, dbg_target=None, nodules=target)
	
	with open(folder_out+'prednodules_'+patient_id+'.npy', 'w') as outfile:
		pred = np.save(outfile, centers)

	print 'number of regions in target', len(target)
	return len(target)


def check_ira_v3():
	folder = 'storage/metadata/dsb3/model-predictions/ikorshun/luna_patch1_v3_dice-20170130-235726-scan/'

	total_nodules = 0

	p_find = subprocess.Popen('find '+folder+' -name "pred_*.pkl"', shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
	for line in p_find.stdout.readlines():
		patient_id = line.rstrip().split('/')[-1].strip('pred_').rstrip('.pkl')
		print 'patient_id', patient_id
		n_nodules_ground_truth = check_nodules_found_v3(patient_id, folder)
		total_nodules+=n_nodules_ground_truth

	print 'total nodules in run', total_nodules

def check_ira_v4(folder_in, folder_out):
	
	total_nodules = 0

	p_find = subprocess.Popen('find '+folder_in+' -name "pred_*.npy"', shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
	lines = p_find.stdout.readlines()
	total_patients = len(lines)
#	for idx, line in enumerate(lines):
        for idx in range(63,119):
		line = lines[idx]
		patient_id = line.rstrip().split('/')[-1].strip('pred_').rstrip('.npy')
		print 'patient', idx, '/', total_patients, 'patient_id', patient_id
		n_nodules_ground_truth = check_nodules_found_v4(patient_id, folder_in, folder_out)
		total_nodules+=n_nodules_ground_truth

	print 'total nodules in run', total_nodules



if __name__ == '__main__':
	print 'check v4 segmentation'
	#check_ira_v4('storage/metadata/dsb3/model-predictions/ikorshun/s_luna_patch_v4_dice/', 'storage/metadata/dsb3/model-predictions/eavsteen/s_luna_patch_v4_dice/')
	check_ira_v4('storage/metadata/dsb3/model-predictions/ikorshun/s2_luna_patch_v4_dice/','storage/metadata/dsb3/model-predictions/eavsteen/s2_luna_patch_v4_dice/')






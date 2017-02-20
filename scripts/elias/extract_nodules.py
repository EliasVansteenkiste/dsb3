import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import BayesianGaussianMixture
from blob import blob_dog, blob_doh, blob_log


from scipy.spatial import distance
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


from scipy.ndimage.filters import convolve
from tf_convolution import conv3d
from segmentation_kernel import get_segmentation_mask


def sample_fits_inside_volume(s):
	for i in range(3):
		if s[i]<0 or s[i]>=512:
			return False
	return True

def fit_gmix(segmentation_volume, n_comp=2):
	"This function finds the best gaussian mix for a volume of probabilities"
	no_samples = 5000
	occurences =  np.round(no_samples*segmentation_volume/np.sum(segmentation_volume)).astype(int)
	total_occ = np.sum(occurences)
	samples = np.zeros((total_occ,3))
	counter = 0

	for x in xrange(segmentation_volume.shape[0]):
		for y in xrange(segmentation_volume.shape[1]):
			for z in xrange(segmentation_volume.shape[2]):
				for occ in range(occurences[x,y,z]):
					samples[counter]=[x,y,z]
					counter += 1

	gmix = BayesianGaussianMixture(n_components=n_comp, covariance_type='full')
	gmix.fit(samples)
	print 'means', gmix.means_
	print 'covariances', gmix.covariances_
	print 'weights', gmix.weights_

	return gmix

def extract_nodules_best_gmix(segmentation_volume, max_n_components=40, plot=False):
	"This function finds the best gaussian mix for a volume of probabilities"
	no_samples = 10000
	occurences =  np.round(no_samples*segmentation_volume/np.sum(segmentation_volume)).astype(int)
	total_occ = np.sum(occurences)
	samples = np.zeros((total_occ,3))
	counter = 0

	for x in xrange(segmentation_volume.shape[0]):
		for y in xrange(segmentation_volume.shape[1]):
			for z in xrange(segmentation_volume.shape[2]):
				for occ in range(occurences[x,y,z]):
					samples[counter]=[x,y,z]
					counter += 1

	best_score = -1
	best_gmix = None
	best_no_c = -1
	for no_c  in range(1,max_n_components):
		gmix = BayesianGaussianMixture(n_components=no_c, covariance_type='full')
		gmix.fit(samples)
		score = gmix.score(samples)
		print 'score', score
		print 'means'
		print gmix.means_
		print 'weights'
		print gmix.weights_
		if plot:
			for idx,mean in enumerate(gmix.means_):
				center = np.round(mean).astype(np.int)
				fig = plt.figure()
				fig.suptitle('weight='+str(gmix.weights_[idx]))
				ax1 = fig.add_subplot(3,1,1)
				ax1.imshow(segmentation_volume[center[0],:,:].transpose())
				circ1 = plt.Circle((center[1],center[2]), 10, color='g', fill=False)
				ax1.add_patch(circ1)

				ax2 = fig.add_subplot(3,1,2)
				ax2.imshow(segmentation_volume[:,center[1],:])
				circ2 = plt.Circle((center[0],center[2]), 10, color='g', fill=False)
				ax2.add_patch(circ2)

				ax3 = fig.add_subplot(3,1,3)
				ax3.imshow(segmentation_volume[:,:,center[2]].transpose())
				circ3 = plt.Circle((center[0],center[1]), 10, color='g', fill=False)
				ax3.add_patch(circ3)
				fig.savefig('no_c_'+str(no_c)+'_'+str(idx)+'.pdf')

		if score>best_score:
			best_score = score
			best_gmix = best_gmix
			best_no_c = no_c

	print "Best gaussian mix when using", best_no_c, 'gaussians'

	return best_gmix

def extract_nodules_conv_filter(segmentation_volume, ct_scan, no_rois=5, dim=8, plot=False, dbg_target=None, nodules=None):
	assert(segmentation_volume.shape==ct_scan.shape)

	#Construct convolutional filter
	cfilt = np.zeros((dim,dim,dim))
	for x in range(dim):
		for y in range(dim):
			for z in range(dim):
				dist = (x-dim/2)**2 + (y-dim/2)**2 + (z-dim/2)**2
				cfilt[x,y,z] = 3*(dim/2)**2-dist
	if plot:
		fig = plt.figure()
		plt.imshow(cfilt[0])
		fig.savefig('filter.pdf')

	#Convolution
	#result = convolve(segmentation_volume, cfilt)
	result = conv3d(segmentation_volume, cfilt)
	if plot:
		fig = plt.figure()
		plt.imshow(result[result.shape[0]/2])
		fig.savefig('result.pdf')
	print result.shape

	#Extract a given number of regions
	rois = []
	if nodules is not None:
		print nodules
		nodule_found = np.zeros((len(nodules)))
	for i in range(no_rois):
		indices = np.where(result == result.max())
		center = [indices[0][0],indices[1][0],indices[2][0]]
		print 'region', i, center, 'max', result.max()

		#cut out patch		
		selection = (slice(center[0]-dim/2,center[0]+dim/2), slice(center[1]-dim/2,center[1]+dim/2), slice(center[2]-dim/2,center[2]+dim/2))
		if dbg_target is not None:
			print 'center in target?', dbg_target[center[0],center[1],center[2]]
		if nodules is not None:
			nodule_in_patch = 0
			#is center in the neigborhoud of nodule?
			for idx, nodule in enumerate(nodules):
				if not nodule_found[idx]:
					if (abs(center[0]-nodule[0])<dim) and (abs(center[1]-nodule[1])<dim) and (abs(center[2]-nodule[2])<dim):
						nodule_in_patch += 1
						nodule_found[idx]=1
			print 'center in target?', nodule_in_patch

		roi = ct_scan[selection]	
		rois.append(roi)

		#set roi to zero in result mask
		zshape = result[selection].shape
		result[selection] = np.zeros(zshape)


		if plot:
			for i in range(len(indices[0])):
				center = [indices[0][i],indices[1][i],indices[2][i]]
				fig = plt.figure()

				ax1 = fig.add_subplot(3,1,1)
				ax1.imshow(segmentation_volume[center[0],:,:].transpose())
				circ1 = plt.Circle((center[1],center[2]), 24, color='y', fill=False)
				ax1.add_patch(circ1)

				ax2 = fig.add_subplot(3,1,2)
				ax2.imshow(segmentation_volume[:,center[1],:])
				circ2 = plt.Circle((center[0],center[2]), 24, color='y', fill=False)
				ax2.add_patch(circ2)

				ax3 = fig.add_subplot(3,1,3)
				ax3.imshow(segmentation_volume[:,:,center[2]].transpose())
				circ3 = plt.Circle((center[0],center[1]), 24, color='y', fill=False)
				ax3.add_patch(circ3)
				fig.savefig('coos_'+str(i)+'.jpg')

	return rois

def check_nodule_fits(center,nodule,dim):
	return (abs(center[0]-nodule[0])<dim) and (abs(center[1]-nodule[1])<dim) and (abs(center[2]-nodule[2])<dim)


def check_in_mask(segmentation_mask, spherical_mask, r_mask, l_x, h_x, l_y, h_y):
	max_x = segmentation_mask.shape[0]
	max_y = segmentation_mask.shape[1]

	if l_x < 0:
		spherical_mask = spherical_mask[-l_x:,:]
		l_x = 0
	if h_x > max_x:
		spherical_mask = spherical_mask[:max_x-h_x,:]
		h_x = max_x
	if l_y < 0:
		spherical_mask = spherical_mask[:,-l_y:]
		l_y = 0
	if h_y > max_y:
		spherical_mask = spherical_mask[:,:max_y-h_y]
		h_y = max_y

	print 'spherical_mask.shape', spherical_mask.shape

	return np.sum(spherical_mask*segmentation_mask[l_x:h_x,l_y:h_y]) > 0





def extract_nodules_blob_detection(segmentation_volume, ct_scan, patient_id=None, dim=8, plot=False, dbg_target=None, nodules=None):
	print segmentation_volume.shape
	print ct_scan.shape
	ct_scan = ct_scan[16:336,16:336,16:336]
	print np.histogram(ct_scan)

	#assert(segmentation_volume.shape==ct_scan.shape)

	results = blob_dog(segmentation_volume, min_sigma=1, max_sigma=15, threshold=0.1)

	#Extract a given number of regions
	rois = []
	centers_in_mask = []
	if nodules is not None:
		print nodules
		nodule_found_in_patch = np.zeros((len(nodules)))
		nodule_found_center = np.zeros((len(nodules)))
		nodule_found_in_patch_in_mask = np.zeros((len(nodules)))
		nodule_found_center_in_mask = np.zeros((len(nodules)))
	for i in range(len(results)):
		center = np.round(results[i]).astype(int)
		print 'region', i, center

		#check if the candidate is in the lungs
		d0_segmentation_mask = get_segmentation_mask(ct_scan[center[0],:,:],z=patient_id+'_'+str(i)+'_d0')
		d1_segmentation_mask = get_segmentation_mask(ct_scan[:,center[1],:],z=patient_id+'_'+str(i)+'_d1')
		d2_segmentation_mask = get_segmentation_mask(ct_scan[:,:,center[2]],z=patient_id+'_'+str(i)+'_d2')

		#Construct 2D spherical mask
		r_mask=3
		dim_mask = 2*r_mask+1
		spherical_mask = np.zeros((dim_mask,dim_mask))
		for x in range(dim_mask):
			for y in range(dim_mask):
				dist = (x-r_mask)**2 + (y-r_mask)**2
				if dist <= (r_mask**2):
					spherical_mask[x,y] = 1


		in_mask_d0 = check_in_mask(d0_segmentation_mask, spherical_mask, r_mask, \
			center[1]-r_mask, center[1]+r_mask+1, center[2]-r_mask, center[2]+r_mask+1)

		in_mask_d1 = check_in_mask(d1_segmentation_mask, spherical_mask, r_mask, \
			center[0]-r_mask, center[0]+r_mask+1, center[2]-r_mask, center[2]+r_mask+1)

		in_mask_d2 = check_in_mask(d2_segmentation_mask, spherical_mask, r_mask, \
			center[0]-r_mask, center[0]+r_mask+1, center[1]-r_mask, center[1]+r_mask+1)

		"""
		majority implementation: A shitty method to bypass the fact that lung segmentation 
		doesn't work when the lung touches the border of the image
		"""
		# in_mask =  (in_mask_d0 and in_mask_d1) or \
		# 		   (in_mask_d1 and in_mask_d2) or \
		# 		   (in_mask_d0 and in_mask_d2) 

		in_mask =  (in_mask_d0 or in_mask_d1 or in_mask_d2)

		#cut out patch		
		selection = (slice(center[0]-dim/2,center[0]+dim/2), slice(center[1]-dim/2,center[1]+dim/2), slice(center[2]-dim/2,center[2]+dim/2))
		
		if in_mask:
			centers_in_mask.append(center)
			roi = ct_scan[selection]	
			rois.append(roi)
		#only for v3
		if dbg_target is not None:
			print 'center in target?', dbg_target[center[0],center[1],center[2]]
		
		#for v4 and onwards
		nodule_in_patch = 0
		center_in_nodule = 0
		if nodules is not None:
			nearest_nodule = None
			d2_nearest_nodule = 999999999.
			nodules_in_patch = []
			nearest_nodule_fits = False
			#is center in the neigborhoud of nodule?
			for idx, nodule in enumerate(nodules):

				if check_nodule_fits(center,nodule,dim):
					nodule_in_patch += 1
					nodule_found_in_patch[idx]=1
					nodules_in_patch.append(nodule)
					if in_mask:
						nodule_found_in_patch_in_mask[idx]=1

				d2_nodule = ((center[0]-nodule[0])**2 + (center[1]-nodule[1])**2 + (center[2]-nodule[2])**2)
				print 'center', center
				print 'nodule', nodule
				print 'd2_nodule', d2_nodule
				if d2_nodule < d2_nearest_nodule:
					d2_nearest_nodule = d2_nodule
					nearest_nodule = nodule
					nearest_nodule_fits = check_nodule_fits(center,nearest_nodule,dim)

				print 'd2_nodule', d2_nodule
				print 'nodule[3]**2/4', (nodule[3]**2)/4
				if d2_nodule < (nodule[3]**2)/4:
					center_in_nodule += 1
					nodule_found_center[idx]=1
					if in_mask:
						nodule_found_center_in_mask[idx]=1
					else:
						#plot at the center of the nearest nodule
						nn_visible_in_d0_slice = (abs(nearest_nodule[0]-center[0]) < nearest_nodule[3]) and nodule_in_patch
						nn_visible_in_d1_slice = (abs(nearest_nodule[1]-center[1]) < nearest_nodule[3]) and nodule_in_patch
						nn_visible_in_d2_slice = (abs(nearest_nodule[2]-center[2]) < nearest_nodule[3]) and nodule_in_patch

						fig = plt.figure()
						nn_rel_coos = (nearest_nodule[0:3]-center[0:3]+[dim,dim,dim]).astype(int)
						title_str = 'Center '+str(center) \
							+'\nNearest nodule at '+str(d2_nearest_nodule**(1./2))+' '+str(nearest_nodule) \
							+'\nnodule in cubic patch? '+str(nodule_in_patch) \
							+'\ncenter blob in nodule? '+str(center_in_nodule) \
							+'\nnn_visible_in_d0_slice '+str(nn_visible_in_d0_slice) \
							+'\nnn_visible_in_d1_slice '+str(nn_visible_in_d1_slice) \
							+'\nnn_visible_in_d2_slice '+str(nn_visible_in_d2_slice) 
						fig.suptitle(title_str, color="green")
						#plot at the center of the blob
						ax1 = fig.add_subplot(3,3,1, adjustable='box', aspect=1.0)
						ax1.set_title('dim0 slice')
						ax1.imshow(ct_scan[center[0],center[1]-dim:center[1]+dim,center[2]-dim:center[2]+dim].transpose(),interpolation='none',cmap=plt.cm.gray)
						if nn_visible_in_d0_slice:
							circ1 = plt.Circle((nn_rel_coos[1],nn_rel_coos[2]), nearest_nodule[3], color='y', fill=False)
							ax1.add_patch(circ1)
						ax2 = fig.add_subplot(3,3,2, adjustable='box', aspect=1.0)
						ax2.set_title('dim1 slice')
						ax2.imshow(ct_scan[center[0]-dim:center[0]+dim,center[1],center[2]-dim:center[2]+dim].transpose(),interpolation='none',cmap=plt.cm.gray)
						if nn_visible_in_d1_slice:	
							circ2 = plt.Circle((nn_rel_coos[0],nn_rel_coos[2]), nearest_nodule[3], color='y', fill=False)
							ax2.add_patch(circ2)
						ax3 = fig.add_subplot(3,3,3, adjustable='box', aspect=1.0)
						ax3.set_title('dim2 slice')
						ax3.imshow(ct_scan[center[0]-dim:center[0]+dim,center[1]-dim:center[1]+dim,center[2]].transpose(),interpolation='none',cmap=plt.cm.gray)
						if nn_visible_in_d2_slice:	
							circ3 = plt.Circle((nn_rel_coos[0],nn_rel_coos[1]), nearest_nodule[3], color='y', fill=False)
							ax3.add_patch(circ3)
						
						ax4 = fig.add_subplot(3,3,4, adjustable='box', aspect=1.0)
						ax4.set_title('dim0 slice')
						ax4.imshow(d0_segmentation_mask.transpose(),interpolation='none',cmap=plt.cm.gray)
						circ4 = plt.Circle((nearest_nodule[1],nearest_nodule[2]), nearest_nodule[3], color='y', fill=False)
						ax4.add_patch(circ4)
						ax5 = fig.add_subplot(3,3,5, adjustable='box', aspect=1.0)
						ax5.set_title('dim1 slice')
						ax5.imshow(d1_segmentation_mask.transpose(),interpolation='none',cmap=plt.cm.gray)
						circ5 = plt.Circle((nearest_nodule[0],nearest_nodule[2]), nearest_nodule[3], color='y', fill=False)
						ax5.add_patch(circ5)
						ax6 = fig.add_subplot(3,3,6, adjustable='box', aspect=1.0)
						ax6.set_title('dim2 slice')
						ax6.imshow(d2_segmentation_mask.transpose(),interpolation='none',cmap=plt.cm.gray)
						circ6 = plt.Circle((nearest_nodule[0],nearest_nodule[1]), nearest_nodule[3], color='y', fill=False)
						ax6.add_patch(circ6)

						ax7 = fig.add_subplot(3,3,7, adjustable='box', aspect=1.0)
						ax7.set_title('dim0 slice')
						ax7.imshow(ct_scan[center[0],:,:].transpose(),interpolation='none',cmap=plt.cm.gray)
						circ7 = plt.Circle((nearest_nodule[1],nearest_nodule[2]), nearest_nodule[3], color='y', fill=False)
						ax7.add_patch(circ7)
						ax8 = fig.add_subplot(3,3,8, adjustable='box', aspect=1.0)
						ax8.set_title('dim1 slice')
						ax8.imshow(ct_scan[:,center[1],:].transpose(),interpolation='none',cmap=plt.cm.gray)
						circ8 = plt.Circle((nearest_nodule[0],nearest_nodule[2]), nearest_nodule[3], color='y', fill=False)
						ax8.add_patch(circ8)
						ax9 = fig.add_subplot(3,3,9, adjustable='box', aspect=1.0)
						ax9.set_title('dim2 slice')
						ax9.imshow(ct_scan[:,:,center[2]].transpose(),interpolation='none',cmap=plt.cm.gray)
						circ9 = plt.Circle((nearest_nodule[0],nearest_nodule[1]), nearest_nodule[3], color='y', fill=False)
						ax9.add_patch(circ9)

						plt.tight_layout()

						fig.savefig('debug/'+patient_id+'_candidate_'+str(i)+'.jpg', dpi=300)

			print 'center in target?', nodule_in_patch
			print 'center in nodule?', center_in_nodule

			if plot:
				nn_visible_in_d0_slice = (abs(nearest_nodule[0]-center[0]) < nearest_nodule[3]) and nodule_in_patch
				nn_visible_in_d1_slice = (abs(nearest_nodule[1]-center[1]) < nearest_nodule[3]) and nodule_in_patch
				nn_visible_in_d2_slice = (abs(nearest_nodule[2]-center[2]) < nearest_nodule[3]) and nodule_in_patch

				fig = plt.figure()
				nn_rel_coos = (nearest_nodule[0:3]-center[0:3]+[dim,dim,dim]).astype(int)
				title_str = 'Center '+str(center) \
					+'\nNearest nodule at '+str(d2_nearest_nodule**(1./2))+' '+str(nearest_nodule) \
					+'\nnodule in cubic patch? '+str(nodule_in_patch) \
					+'\ncenter blob in nodule? '+str(center_in_nodule) \
					+'\nnn_visible_in_d0_slice '+str(nn_visible_in_d0_slice) \
					+'\nnn_visible_in_d1_slice '+str(nn_visible_in_d1_slice) \
					+'\nnn_visible_in_d2_slice '+str(nn_visible_in_d2_slice) 
				fig.suptitle(title_str)



				#plot at the center of the blob
				ax1 = fig.add_subplot(1,3,1, adjustable='box', aspect=1.0)
				ax1.set_title('dim0 slice')
				ax1.imshow(ct_scan[center[0],center[1]-dim:center[1]+dim,center[2]-dim:center[2]+dim].transpose(),interpolation='none',cmap=plt.cm.gray)
				if nn_visible_in_d0_slice:
					circ1 = plt.Circle((nn_rel_coos[1],nn_rel_coos[2]), nearest_nodule[3], color='y', fill=False)
					ax1.add_patch(circ1)
				ax2 = fig.add_subplot(1,3,2, adjustable='box', aspect=1.0)
				ax2.set_title('dim1 slice')
				ax2.imshow(ct_scan[center[0]-dim:center[0]+dim,center[1],center[2]-dim:center[2]+dim].transpose(),interpolation='none',cmap=plt.cm.gray)
				if nn_visible_in_d1_slice:	
					circ2 = plt.Circle((nn_rel_coos[0],nn_rel_coos[2]), nearest_nodule[3], color='y', fill=False)
					ax2.add_patch(circ2)
				ax3 = fig.add_subplot(1,3,3, adjustable='box', aspect=1.0)
				ax3.set_title('dim2 slice')
				ax3.imshow(ct_scan[center[0]-dim:center[0]+dim,center[1]-dim:center[1]+dim,center[2]].transpose(),interpolation='none',cmap=plt.cm.gray)
				if nn_visible_in_d2_slice:	
					circ3 = plt.Circle((nn_rel_coos[0],nn_rel_coos[1]), nearest_nodule[3], color='y', fill=False)
					ax3.add_patch(circ3)

				plt.tight_layout()

				fig.savefig('plots/'+patient_id+'_candidate_'+str(i)+'.jpg')


				fig = plt.figure()
				nn_rel_coos = (nearest_nodule[0:3]-center[0:3]+[dim,dim,dim]).astype(int)
				title_str = 'Center '+str(center) \
					+'\nNearest nodule at '+str(d2_nearest_nodule**(1./2))+' '+str(nearest_nodule) \
					+'\nnodule in cubic patch? '+str(nodule_in_patch) \
					+'\ncenter blob in nodule? '+str(center_in_nodule) \
					+'\nnn_visible_in_d0_slice '+str(nn_visible_in_d0_slice) \
					+'\nnn_visible_in_d1_slice '+str(nn_visible_in_d1_slice) \
					+'\nnn_visible_in_d2_slice '+str(nn_visible_in_d2_slice) 
				fig.suptitle(title_str)
				#plot the segementation volume at the center of the blob
				ax1 = fig.add_subplot(1,3,1, adjustable='box', aspect=1.0)
				ax1.set_title('dim0 slice')
				ax1.imshow(segmentation_volume[center[0],:,:].transpose(),interpolation='none',cmap=plt.cm.gray)
				if nn_visible_in_d0_slice:
					circ1 = plt.Circle((nearest_nodule[1],nearest_nodule[2]), nearest_nodule[3], color='y', fill=False)
					ax1.add_patch(circ1)
				ax2 = fig.add_subplot(1,3,2, adjustable='box', aspect=1.0)
				ax2.set_title('dim1 slice')
				ax2.imshow(segmentation_volume[:,center[1],:].transpose(),interpolation='none',cmap=plt.cm.gray)
				if nn_visible_in_d1_slice:	
					circ2 = plt.Circle((nearest_nodule[0],nearest_nodule[2]), nearest_nodule[3], color='y', fill=False)
					ax2.add_patch(circ2)
				ax3 = fig.add_subplot(1,3,3, adjustable='box', aspect=1.0)
				ax3.set_title('dim2 slice')
				ax3.imshow(segmentation_volume[:,:,center[2]].transpose(),interpolation='none',cmap=plt.cm.gray)
				if nn_visible_in_d2_slice:	
					circ3 = plt.Circle((nearest_nodule[0],nearest_nodule[1]), nearest_nodule[3], color='y', fill=False)
					ax3.add_patch(circ3)

				plt.tight_layout()

				fig.savefig('segmentation_volumes/'+patient_id+'_at_candidate_'+str(i)+'.jpg')

				#plot at the center of the nearest nodule
				fig = plt.figure()
				title_str = 'Nodule '+str(nearest_nodule) \
					+'\nCenter at '+str(d2_nearest_nodule**(1./2))+' '+str(center) \
					+'\nnodule in cubic patch? '+str(nodule_in_patch) \
					+'\ncenter blob in nodule? '+str(center_in_nodule)
				fig.suptitle(title_str)
				nearest_nodule = nearest_nodule.astype(int)

				ax1 = fig.add_subplot(1,3,1, adjustable='box', aspect=1.0)
				ax1.set_title('dim0 slice')
				ax1.imshow(ct_scan[nearest_nodule[0], nearest_nodule[1]-dim:nearest_nodule[1]+dim, nearest_nodule[2]-dim:nearest_nodule[2]+dim].transpose(),interpolation='none',cmap=plt.cm.gray)

				ax2 = fig.add_subplot(1,3,2, adjustable='box', aspect=1.0)
				ax2.set_title('dim1 slice')
				ax2.imshow(ct_scan[nearest_nodule[0]-dim:nearest_nodule[0]+dim,nearest_nodule[1],nearest_nodule[2]-dim:nearest_nodule[2]+dim].transpose(),interpolation='none',cmap=plt.cm.gray)

				ax3 = fig.add_subplot(1,3,3, adjustable='box', aspect=1.0)
				ax3.set_title('dim2 slice')
				ax3.imshow(ct_scan[nearest_nodule[0]-dim:nearest_nodule[0]+dim,nearest_nodule[1]-dim:nearest_nodule[1]+dim,nearest_nodule[2]].transpose(),interpolation='none',cmap=plt.cm.gray)

				plt.tight_layout()

				fig.savefig('plots/'+patient_id+'_candidate_'+str(i)+'_nn.jpg')

	if plot:
		for idx, nodule in enumerate(nodules):
			if not nodule_found[idx] or not nodule_found_2[idx]:
				#plot nodule in not_found folder
				fig = plt.figure()
				title_str = 'Nodule '+str(nodule) \
					+'\nnodule in cubic patch? '+str(nodule_found[idx]) \
					+'\ncenter blob in nodule? '+str(nodule_found_2[idx])
				fig.suptitle(title_str)
				nodule = nodule.astype(int)

				ax1 = fig.add_subplot(1,3,1, adjustable='box', aspect=1.0)
				ax1.set_title('dim0 slice')
				ax1.imshow(ct_scan[nodule[0], nodule[1]-dim:nodule[1]+dim, nodule[2]-dim:nodule[2]+dim].transpose(),interpolation='none',cmap=plt.cm.gray)

				ax2 = fig.add_subplot(1,3,2, adjustable='box', aspect=1.0)
				ax2.set_title('dim1 slice')
				ax2.imshow(ct_scan[nodule[0]-dim:nodule[0]+dim,nodule[1],nodule[2]-dim:nodule[2]+dim].transpose(),interpolation='none',cmap=plt.cm.gray)

				ax3 = fig.add_subplot(1,3,3, adjustable='box', aspect=1.0)
				ax3.set_title('dim2 slice')
				ax3.imshow(ct_scan[nodule[0]-dim:nodule[0]+dim,nodule[1]-dim:nodule[1]+dim,nearest_nodule[2]].transpose(),interpolation='none',cmap=plt.cm.gray)

				plt.tight_layout()

				fig.savefig('not_found/'+patient_id+'_candidate_'+str(idx)+'.jpg')

	print "stats-nodule_found_in_patch:", int(np.sum(nodule_found_in_patch)), "/", len(nodules)
	print "stats-nodule_found_center:", int(np.sum(nodule_found_center)), "/", len(nodules)
	print "stats-nodule_found_in_patch_in_mask:", int(np.sum(nodule_found_in_patch_in_mask)), "/", len(nodules)
	print "stats-nodule_found_center_in_mask:", int(np.sum(nodule_found_center_in_mask)), "/", len(nodules)
	print "stats-no-candidates:", len(results)  
	print "stats-no-candidates-mask:", len(rois) 

	return rois, results

def compute_bic(kmeans,X):
    """
    Computes the BIC metric for a given clusters

    Parameters:
    -----------------------------------------
    kmeans:  List of clustering object from scikit learn

    X     :  multidimension np array of data points

    Returns:
    -----------------------------------------
    BIC value
    """
    # assign centers and labels
    centers = [kmeans.cluster_centers_]
    labels  = kmeans.labels_
    #number of clusters
    m = kmeans.n_clusters
    # size of the clusters
    n = np.bincount(labels)
    #size of data set
    N, d = X.shape

    #compute variance for all clusters beforehand
    cl_var = (1.0 / (N - m) / d) * sum([sum(distance.cdist(X[np.where(labels == i)], [centers[0][i]], 
             'euclidean')**2) for i in range(m)])

    const_term = 0.5 * m * np.log(N) * (d+1)

    BIC = np.sum([n[i] * np.log(n[i]) -
               n[i] * np.log(N) -
             ((n[i] * d) / 2) * np.log(2*np.pi*cl_var) -
             ((n[i] - 1) * d/ 2) for i in range(m)]) - const_term

    return(BIC)

def extract_nodules_best_kmeans(segmentation_volume, max_n_components=10, plot=False):
	"This function finds the best kmeans clustering for a volume of probabilities"
	no_samples = 50000
	occurences =  np.round(no_samples*segmentation_volume/np.sum(segmentation_volume)).astype(int)
	total_occ = np.sum(occurences)
	samples = np.zeros((total_occ,3))
	counter = 0

	for x in xrange(segmentation_volume.shape[0]):
		for y in xrange(segmentation_volume.shape[1]):
			for z in xrange(segmentation_volume.shape[2]):
				for occ in range(occurences[x,y,z]):
					samples[counter]=[x,y,z]
					counter += 1

	best_score = -99999999999.0
	best_km = None
	best_no_c = 0
	for no_c  in range(1,max_n_components):
		km = KMeans(n_clusters=no_c)
		km.fit(samples)
		score = compute_bic(km,samples)/1.0e3
		print score
		print km.cluster_centers_
		if plot:
			for idx,center in enumerate(km.cluster_centers_):
				center = np.round(center).astype(np.int)
				fig = plt.figure()

				ax1 = fig.add_subplot(3,1,1)
				ax1.imshow(segmentation_volume[center[0],:,:].transpose())
				circ1 = plt.Circle((center[1],center[2]), 10, color='g', fill=False)
				ax1.add_patch(circ1)

				ax2 = fig.add_subplot(3,1,2)
				ax2.imshow(segmentation_volume[:,center[1],:])
				circ2 = plt.Circle((center[0],center[2]), 10, color='g', fill=False)
				ax2.add_patch(circ2)

				ax3 = fig.add_subplot(3,1,3)
				ax3.imshow(segmentation_volume[:,:,center[2]].transpose())
				circ3 = plt.Circle((center[0],center[1]), 10, color='g', fill=False)
				ax3.add_patch(circ3)
				fig.savefig('no_c_'+str(no_c)+'_'+str(idx)+'.pdf')

		if score>best_score:
			best_score = score
			best_km = km
			best_no_c = no_c

	print "Best kmeans fit when using", best_no_c, 'clusters'

	return best_km

def extract_rois(segmentation_volume, size_roi=40, max_n_rois=10):
	"""
	Extract the region of interest by fitting a Gaussian mixture model
	

    Parameters
    ----------
    segmentation_volume : 3D numpy array with floating numbers. 
    	For each voxel, the floating point represents the probability that 
    	the voxel is inside a malignant nodule.
    size_roi : integer
    	The size for the region of interests that will be returned. 
    	In other words, the dimension of the 3D cube that is cut out around 
    	the centerof the nodules that have been found.
    max_n_rois : integer
        The maximum number of regions of interest that will be searched for and cut out.
        This is also equals the number of clusters in the kmeans model that is 
        fit to the probabilties of the segmentation_volume

    Returns
    -------
    numpy ndarray
        A 4D numpy array with the regions of interest

	"""

	# fit the gaussian mix
	km = extract_nodules_best_kmeans(segmentation_volume, max_n_rois)
	
	# extract region of interests
	rois = np.zeros((n_rois,size_roi,size_roi,size_roi))
	for idx, center in enumerate(km.cluster_centers_):
		center=np.round(center).astype(int)
		print center
		half_size = size_roi/2
		rois[idx] = segmentation_volume[center[0]-half_size:center[0]+half_size,center[1]-half_size:center[1]+half_size,center[2]-half_size:center[2]+half_size]

	return rois

if __name__=="__main__":
	#testing out the functions
	mean1 = [200, 100, 50]
	cov1 = [[3, 0, 0], [0, 15, 0], [0, 0, 20]] 
	samples1 = np.random.multivariate_normal(mean1, cov1, 5000)

	mean2 = [30, 40, 10]
	cov2 = [[8, 0, 0], [0, 15, 0], [0, 0, 10]] 
	samples2 = np.random.multivariate_normal(mean2, cov2, 5000)

	samples = np.vstack((samples1,samples2))
	print samples.shape

	occurences = np.zeros((512,512,512))
	no_samples = 0
	print 'sum', np.sum(occurences)
	for s in samples:
		ss = np.round(s).astype(int)
		if sample_fits_inside_volume(ss):
			occurences[tuple(ss)] += 1
			no_samples += 1
		else:
			print 'warning sample omitted because it does not fit in volume'

	test_probs = 1.0*occurences/np.sum(occurences)

	extract_nodules_best_kmeans(test_probs)





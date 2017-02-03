import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import BayesianGaussianMixture
from scipy.spatial import distance
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.ndimage.filters import convolve


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

def extract_nodules_conv_filter(segmentation_volume, ct_scan, no_rois=5, dim=8, plot=False, dbg_target=None):
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
	result = convolve(segmentation_volume, cfilt)
	if plot:
		fig = plt.figure()
		plt.imshow(result[result.shape[0]/2])
		fig.savefig('result.pdf')
	print result.shape

	#Extract a given number of regions
	rois = []
	for i in range(no_rois):
		indices = np.where(result == result.max())
		center = [indices[0][0],indices[1][0],indices[2][0]]
		print 'region', i, center, 'max', result.max()

		#cut out patch		
		selection = (slice(center[0]-dim/2,center[0]+dim/2), slice(center[1]-dim/2,center[1]+dim/2), slice(center[2]-dim/2,center[2]+dim/2))
		if dbg_target is not None:
			print 'center in target?', dbg_target[center[0],center[1],center[2]]

		roi = ct_scan[selection]
		rois.append(roi)

		#set roi to zero in result mask
		print selection
		result[selection] = np.zeros((dim,dim,dim))


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





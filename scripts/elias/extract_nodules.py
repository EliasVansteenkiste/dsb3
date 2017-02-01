import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from sklearn.cluster import KMeans
from scipy.spatial import distance


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

def extract_nodules_best_gmix(segmentation_volume, max_n_components=10):
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
	for no_c  in range(1,max_n_components):
		gmix = BayesianGaussianMixture(n_components=no_c, covariance_type='full')
		gmix.fit(samples)
		score = gmix.score(samples)
		print 'score', score
		print 'means'
		print gmix.means_
		print 'weights'
		print gmix.weights_
		if score>best_score:
			best_score = score
			best_gmix = gmix

	return best_gmix

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

def extract_nodules_best_kmeans(segmentation_volume, max_n_components=10):
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





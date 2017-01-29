import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture



"""
The most important function in this file for use in our stack is extract_rois
"""

def get_rv(mu_x, mu_y, mu_z, variance_x, variance_y, variance_z):
	return multivariate_normal([mu_x, mu_y, mu_z], [[variance_x, 0, 0], [0, variance_y, 0], [0, 0, variance_z]])

def fit_gmix(segmentation_volume, n_comp=2):
	"This function finds the best gaussian mix for a volume of probabilities"
	no_samples = 4000
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

	gmix = GaussianMixture(n_components=n_comp, covariance_type='full')
	gmix.fit(samples)
	print 'means', gmix.means_
	print 'covariances', gmix.covariances_
	print 'weights', gmix.weights_

	return gmix

def extract_rois(segmentation_volume, size_roi=10, n_rois=2):
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
    n_rois : integer
        The number of regions of interest that will be searched for and cut out.
        This is also equals the number of Gaussian processes in the model that is 
        fit to the probabilties of the segmentation_volume

    Returns
    -------
    numpy ndarray
        The input array in the ``floatX`` dtype configured for Theano.
        If `arr` is an ndarray of correct dtype, it is returned as is.

	"""

	# fit the gaussian mix
	gmix = fit_gmix(segmentation_volume,n_rois)
	
	# extract region of interests
	rois = np.zeros((n_rois,size_roi,size_roi,size_roi))
	for idx, center in enumerate(gmix.means_):
		center=np.round(center).astype(int)
		print center
		half_size = size_roi/2
		rois[idx] = segmentation_volume[center[0]-half_size:center[0]+half_size,center[1]-half_size:center[1]+half_size,center[2]-half_size:center[2]+half_size]




def extract_nodules_best_gmix(segmentation_volume, max_n_components=10):
	"!!!!!!!!!!!!does not work well yet"
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
		gmix = GaussianMixture(n_components=no_c, covariance_type='full')
		gmix.fit(samples)
		score = gmix.score(samples)
		print 'score', score
		print 'means', gmix.means_
		# print 'covariances', gmix.covariances_
		print 'weights', gmix.weights_
		if score>best_score:
			best_score = score
			best_gmix = gmix

	return best_gmix


if __name__=="__main__":
	#Create grid and multivariate normal
	x = np.linspace(0,512,512)
	y = np.linspace(0,512,512)
	z = np.linspace(0,256,512)

	X, Y, Z = np.meshgrid(x,y,z)

	print X.shape
	print Y.shape

	pos = np.empty(X.shape + (3,))
	pos[:, :, :, 0] = X
	pos[:, :, :, 1] = Y
	pos[:, :, :, 2] = Z

	print pos.shape

	rv1 = get_rv(200,100,50,3,15,20)
	rv2 = get_rv(30,40,10,8,15,10)


	test = rv1.pdf(pos)+rv2.pdf(pos)
	print 'rv1.pdf(pos)+rv2.pdf(pos).shape', test.shape

	extract_rois(test, n_rois=2)







import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture


def get_rv(mu_x, mu_y, mu_z, variance_x, variance_y, variance_z):
	return multivariate_normal([mu_x, mu_y, mu_z], [[variance_x, 0, 0], [0, variance_y, 0], [0, 0, variance_z]])

def extract_nodules_gmix(segmentation_volume, no_components=2):
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

	gmix = GaussianMixture(n_components=no_c, covariance_type='full')
	gmix.fit(samples)
	score = gmix.score(samples)
	print 'score', score
	print 'means', gmix.means_
	print 'covariances', gmix.covariances_
	print 'weights', gmix.weights_

	return bgmix

def extract_nodules_best_gmix(segmentation_volume, max_n_components=10):
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

	best_score = -1
	best_gmix = None
	for no_c  in range(1,max_n_components):
		gmix = GaussianMixture(n_components=no_c, covariance_type='full')
		gmix.fit(samples)
		score = gmix.score(samples)
		print 'score', score
		print 'means', gmix.means_
		print 'covariances', gmix.covariances_
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

	extract_nodules_gmix(test)







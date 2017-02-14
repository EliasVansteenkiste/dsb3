from skimage import data
from blob import blob_dog

print blob_dog(data.coins(), threshold=.5, max_sigma=40)

# FID
# Frechet Inception Distance Implementation



# For our dense layers implementation

# example of calculating the frechet inception distance in Keras
import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import shuffle
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.datasets.mnist import load_data
from skimage.transform import resize
from keras.datasets import cifar10

# scale an array of images to a new size
def scale_images(images, new_shape):
	images_list = list()
	for image in images:
		# resize with nearest neighbor interpolation
		new_image = resize(image, new_shape, 0)
		# store
		images_list.append(new_image)
	return asarray(images_list)

# calculate frechet inception distance
def calculate_fid(model, images1, images2):
	# calculate activations
	act1 = model.predict(images1)
	act2 = model.predict(images2)
	# calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = numpy.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid

# prepare the inception v3 model
model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))

  
from PIL import Image
import matplotlib.pyplot as plt
from IPython.display import display
!pip install ipyplot
import ipyplot

dataSetNew = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data',
               transform=torchvision.transforms.ToTensor(),
               download=True),
        batch_size=10,
        shuffle=True)

dataIter = iter(dataSetNew)
imagesOriginal, labels = dataIter.next()
imagesOriginal = imagesOriginal.to(device)
imagesTransformed = vae(imagesOriginal)

imagesOriginal = imagesOriginal.view(10,28,28).cpu().detach().numpy()
imagesTransformed = imagesTransformed.view(10,28,28).cpu().detach().numpy()

print('\nimagesOriginal')
ipyplot.plot_images(imagesOriginal)

print('\nimagesTransformed')
ipyplot.plot_images(imagesTransformed)

# convert integer to floating point values
imagesOriginal = imagesOriginal.astype('float32')
images2 = imagesTransformed.astype('float32')
# resize images
imagesOriginal = scale_images(imagesOriginal, (299,299,3))
imagesTransformed = scale_images(imagesTransformed, (299,299,3))
print('Scaled', imagesTransformed.shape, imagesTransformed.shape)
# pre-process images
imagesOriginal = preprocess_input(imagesOriginal)
imagesTransformed = preprocess_input(imagesTransformed)
# calculate fid, if fid=0 -> images are identical, the lower fid the better
fid = calculate_fid(model, imagesOriginal, imagesTransformed)
print('FID: %.3f' % fid)

# ********************************************************************************************************
# For our conv layers implementation


# example of calculating the frechet inception distance in Keras
import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import shuffle
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.datasets.mnist import load_data
from skimage.transform import resize
from keras.datasets import cifar10

# scale an array of images to a new size
def scale_images(images, new_shape):
	images_list = list()
	for image in images:
		# resize with nearest neighbor interpolation
		new_image = resize(image, new_shape, 0)
		# store
		images_list.append(new_image)
	return asarray(images_list)

# calculate frechet inception distance
def calculate_fid(model, images1, images2):
	# calculate activations
	act1 = model.predict(images1)
	act2 = model.predict(images2)
	# calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = numpy.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid

# prepare the inception v3 model
model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))

from PIL import Image
import matplotlib.pyplot as plt
from IPython.display import display
!pip install ipyplot
import ipyplot

# plot_images(first_images.view(10,28,28).cpu().detach().numpy())
# plot_images(np_recon)
imagesOriginal = first_images.view(10,28,28).cpu().detach().numpy()
imagesTransformed = np_recon

print('\nimagesOriginal')
ipyplot.plot_images(imagesOriginal)

print('\nimagesTransformed')
ipyplot.plot_images(imagesTransformed)

# convert integer to floating point values
imagesOriginal = imagesOriginal.astype('float32')
imagesTransformed = imagesTransformed.astype('float32')
# resize images
imagesOriginal = scale_images(imagesOriginal, (299,299,3))
imagesTransformed = scale_images(imagesTransformed, (299,299,3))
# pre-process images
imagesOriginal = preprocess_input(imagesOriginal)
imagesTransformed = preprocess_input(imagesTransformed)
# calculate fid, if fid=0 -> images are identical, the lower fid the better
fid = calculate_fid(model, imagesOriginal, imagesTransformed)
print('FID: %.3f' % fid)

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib as mpl 
import h5py


filename = 'patient081_frame01.h5'


with h5py.File(filename,'r') as f:
	image = f['image'][:]
	label = f['label'][:]
	scribble = f['scribble'][:]



	print(np.shape(image))
	print(np.amax(image))
	print(np.shape(label))
	print(np.amax(label))
	print(np.shape(scribble))
	print(np.amax(scribble))





	image = scribble[5,:,:]


	plt.imshow(image, cmap='gray', vmin=0, vmax=4)
	plt.xticks([])
	plt.show()
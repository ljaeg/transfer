import numpy as np 
import pandas as pd 
from PIL import Image 
import os 
from skimage.measure import block_reduce
import h5py
import matplotlib.pyplot as plt 

Save_dir = "/Users/loganjaeger/Desktop/"
Dir = "/Users/loganjaeger/Downloads"
path_to_csv = os.path.join(Dir, "train_info.csv")
path_to_ims = os.path.join(Dir, "train_1")

style = "Post-Impressionism"

im_names = [f for f in os.listdir(path_to_ims)]

DF = pd.read_csv(path_to_csv)

b = DF["style"] == style
only_style = DF[b]
filenames = only_style["filename"]

# x = DF.groupby(by = "style").count().sort_values(by = "filename", ascending = False)
# print(x.head(20))

#print(DF["style"].unique())

def change_im(im):
	if len(im.shape) == 2:
		return None
	to_return = block_reduce(im, (4, 4, 1), func = np.mean)
	s = to_return.shape
	if s[0] < 300 or s[1] < 300:
		return None
	else:
		return to_return[:300, :300, :]

def retrieve_files():
	all_ims = []
	for i in range(len(filenames)):
		name = filenames.iloc[i]
		if os.path.isfile(os.path.join(path_to_ims, name)):
			im = np.array(Image.open(os.path.join(path_to_ims, name)))
			im = change_im(im)
			if im is not None:
				all_ims.append(im)
	print("number of ims: ", len(all_ims))
	all_ims = np.array(all_ims)
	f = h5py.File(os.path.join(Save_dir, "self_python_files", "transfer", "{}.hdf5".format(style)), "w")
	f.create_dataset("images", data = all_ims)
	f.close()

#retrieve_files()

f = h5py.File(os.path.join(Save_dir, "self_python_files", "transfer", "{}.hdf5".format(style)), "r")
ims = np.array(f["images"]).astype("uint8")
plt.imshow(ims[80])
plt.show()
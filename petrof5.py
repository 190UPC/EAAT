import os
import glob
import h5py
import scipy.misc as misc
import imageio
import numpy as np
import cv2

dataset_dir = "train"
dataset_type = "train"

f = h5py.File("DeepRock2D_new_default.h5".format(dataset_type), "w")
dt = h5py.special_dtype(vlen=np.dtype('uint8'))

for subdir in ["HR", "X4"]:
    if subdir in ["HR"]:
        im_paths = glob.glob(os.path.join(dataset_dir,
                                          "{}_HR".format(dataset_type),
                                          "*.png"))

    else:
        im_paths = glob.glob(os.path.join(dataset_dir,
                                          "{}_LR".format(dataset_type),
                                          subdir, "*.png"))
    im_paths.sort()
    grp = f.create_group(subdir)

    for i, path in enumerate(im_paths):
        im = imageio.imread(path)
        print(path)
        grp.create_dataset(str(i), data=im)

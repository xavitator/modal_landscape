import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from skimage import data

from PIL.Image import *

from imageio import imread
import glob
from tqdm import tnrange
from tqdm.notebook import tqdm_notebook
import tqdm
import torch
from torchvision import datasets,transforms
import torch.nn.functional as F
import torch.nn as nn
from torchvision.utils import save_image

import multiprocessing




size = 60

def image_to_numpy(image) :
    n = len(image)
    image_bis = []
    for i in range (n) :
        image_bis.append([])
        for j in range(n) :
            image_bis[i].append(image[i][j][0])
    return np.array(image_bis)
        
def load_img_from_path(img_path):
    id_image, label_image = img_path
    res = []
    deep = 3
    for i in range(deep):
        image_path = id_image + "c" + str(i) + ".png"
        image = imread(image_path)
        image = image_to_numpy(image)
        res.append(image)
    return (label_image, res)




#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/python

# Note to Kagglers: This script will not run directly in Kaggle kernels. You
# need to download it and run it on your local machine.

# Downloads images from the Google Landmarks dataset using multiple threads.
# Images that already exist will not be downloaded again, so the script can
# resume a partially completed download. All images will be saved in the JPG
# format with 90% compression quality.

import sys, os, multiprocessing, csv
from PIL import Image
from io import BytesIO
from urllib.request import urlopen
import tqdm
from tqdm import tnrange
from tqdm.contrib.concurrent import process_map

from matplotlib import pyplot as plt
from imageio import imread

debut_lab = -1
fin_lab = 100000
size = 120


def ParseData(data_file):
  csvfile = open(data_file, 'r')
  csvreader = csv.reader(csvfile)
  dataset = [line for line in csvreader]
  return dataset[1:]  # Chop off header


def DownloadImage(data):
  (key, url, label) = data
  try:
    img = imread(url)
    return img
  except:
    return


# In[2]:


def Create_labels (data_file):
    dataset_url = ParseData(data_file)
    dataset = []
    for data in tqdm.tqdm(dataset_url[:len(dataset_url)], total=len(dataset_url)) :
        (key, url, label) = data 
        if label != "None" :
            dataset.append(int(label))
    return dataset


# In[3]:


list_labels = Create_labels("train.csv")


# In[4]:


len(list_labels)


# In[5]:


max(list_labels)


# In[6]:


num_labels = plt.hist(list_labels, bins=range(max(list_labels)+2))


# In[7]:


num_labels = num_labels[0]


# In[8]:


max(num_labels)


# In[9]:


i= 0
nouv_lab = []
for j in tnrange(len(list_labels)) :
    if num_labels[list_labels[j]]>debut_lab and num_labels[list_labels[j]]<fin_lab :
        nouv_lab.append(list_labels[j])
list_labels = nouv_lab


# In[10]:


num_labels = plt.hist(list_labels, bins=range(14950+2))


# In[11]:


num_labels = num_labels[0]


# In[12]:


max(num_labels)


# In[13]:


len(list_labels)


# In[14]:


compteur = 0
dataset_url = ParseData("train.csv")
image_per_label = [0]*(15000)
for data in tqdm.tqdm(dataset_url[:len(dataset_url)], total=len(dataset_url)):
    (key, url, label) = data
    if (label != "None") and (num_labels[int(label)]>debut_lab and num_labels[int(label)]<fin_lab) and image_per_label[int(label)]<fin_lab :
        image_per_label[int(label)]+=1
        compteur+=1
compteur


# In[15]:


dataset_url = ParseData("train.csv")


# In[16]:


import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from skimage import data

import PIL.Image as IMG

from imageio import imread
import glob

import torch
from torchvision.utils import save_image

matplotlib.rcParams['font.size'] = 18


# In[17]:


from preprocessing import *

print("preprocessing")
# In[18]:


def rescale_reshape(img, size):
    #img_t = to_float32(img)
    img_t = rescale(img,size, size)
    #print(img_t)
    return img_t

def ajout_transfo(img, high1=0.5, low1=0.1, high2=0.2, low2=0.05) :
    r,g,b = rgb(img) 
    return [r,g,b]


# In[19]:


"""
image_per_label = multiprocessing.Array('i', [0]*(15000))


def create_and_register(t):
    i,data = t
    (key, url, label) = data
    if (label != "None") and (num_labels[int(label)]>150 and num_labels[int(label)]<300) and image_per_label[int(label)]<150 :
        pil_image = DownloadImage(data)
        if pil_image!= None :
            pil_image = np.array(pil_image)
            image_per_label[int(label)]+=1
            pil_image = rescale_reshape(pil_image, size)
            pil_image_li = ajout_transfo(pil_image)
            couche = 0
            for img in pil_image_li :
                save_image(torch.from_numpy(img),"/home/xavierdurand/transsup150/" + str(i) + 'l' + str(label) + 'c' + str(couche) +".png")
                couche += 1
            

def CreateDataset(data_file, num_labels):
    arg = [(i,dataset_url[i]) for i in range(len(dataset_url))]
    with multiprocessing.Pool() as p :
        list(tqdm.tqdm(p.imap(create_and_register, arg), total=len(dataset_url)))
"""


# In[23]:


image_per_label = multiprocessing.Array('i', [0]*(15000))

import torch, torchvision

img_path_folder = "/home/StanXav/modal_landscape/data"

def create_and_register(t):
    i,data = t
    (key, url, label) = data
    if (label != "None") and (num_labels[int(label)]>debut_lab and num_labels[int(label)]<fin_lab) :
        pil_image = DownloadImage(data)
        if type(pil_image)!= type(None) :
            pil_image = np.array(pil_image)
            pil_image = to_float32(pil_image)
            if len(pil_image.shape) < 3 :
                return
            image_per_label[int(label)]+=1
            pil_image = rescale_reshape(pil_image, size)
            pil_image_li_120 = ajout_transfo(pil_image)
            pil_image = rescale_reshape(pil_image, size//2)
            pil_image_li_60 = ajout_transfo(pil_image)
            couche = 0
            for img_check in pil_image_li_120 :
                if img_check.size == 0:
                    return
            for img_check in pil_image_li_60 :
                if img_check.size == 0:
                    return
            for i in range(3) :
                img = pil_image_li_120[i]
                img = img.reshape((1, size, size))
                img = torch.from_numpy(img)
                torchvision.utils.save_image(img, img_path_folder + "/" + str(i) + 'l' + str(label) + 'c' + str(couche) +".png")
                
                img = pil_image_li_60[i]
                img = img.reshape((1, size//2, size//2))
                img = torch.from_numpy(img)
                torchvision.utils.save_image(img, img_path_folder + "_60/" + str(i) + 'l' + str(label) + 'c' + str(couche) +".png")
                
                couche += 1
            

def CreateDataset(data_file, num_labels):
    arg = [(i,dataset_url[i]) for i in range(len(dataset_url))]
    with multiprocessing.Pool() as p :
        list(tqdm.tqdm(p.imap(create_and_register, arg), total=len(dataset_url)))


# In[24]:


CreateDataset("train.csv", num_labels)


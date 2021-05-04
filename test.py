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




def ParseData(data_file):
  csvfile = open(data_file, 'r')
  csvreader = csv.reader(csvfile)
  dataset = [line for line in csvreader]
  return dataset[1:]  # Chop off header


def DownloadImage(data):
  (key, url, label) = data

  try:
    response = urlopen(url)
    image_data = response.read()
  except:
    print('Warning: Could not download image %s from %s' % (key, url))
    return

  try:
    pil_image = Image.open(BytesIO(image_data))
    return pil_image
  except:
    print('Warning: Failed to parse image %s' % key)
    return


def CreateDataset(data_file, nb_data=100) :
    dataset_url = ParseData(data_file)
    dataset = []
    for data in tqdm.tqdm(dataset_url[:nb_data], total=nb_data) :
        (key, url, label) = data 
        if (int(label)<100) :
            pil_image = DownloadImage(data)
            dataset.append([pil_image,label])
    return dataset


def test(n):
    for i in range(n):
        print(i)

dataset = CreateDataset("train.csv", 1000000)

compteur = 0
for i in tqdm.tqdm(range(len(dataset)), total = len(dataset)) :
    if dataset[i][0]!= None :
        dataset[i][0].save("images\ " + dataset[i][1]+ "image" +str(compteur)+".png")
        compteur+=1

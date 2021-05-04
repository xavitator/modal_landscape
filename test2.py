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

def CreateDataset_bis(data_file, nb_data=100) :
    dataset_url = ParseData(data_file)
    dataset = []
    for i in range(nb_data) :
        (key, url, label) = dataset_url[i]
        if (int(label)<100) :
            pil_image = DownloadImage(dataset_url[i])
            dataset.append([pil_image,label])
            
    return dataset

def Compteur_label(data_file,label_limit, nb_data=100) :
    dataset_url = ParseData(data_file)
    compteur = 0
    for i in range(nb_data) :
        (key, url, label) = dataset_url[i]
        if (int(label)<label_limit) :
            compteur +=1
            
    return compteur


nb_data = Compteur_label("train.csv",200, 1000000)
print(nb_data)


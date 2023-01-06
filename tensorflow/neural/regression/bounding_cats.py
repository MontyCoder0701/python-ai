# CNN (Colored Image classification- Bounding box cats)
from keras import callbacks
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, mean_squared_error
import tensorflow as tf
from tensorflow import keras
from keras import layers
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import PIL
import glob
import xml.etree.ElementTree as ET
import os


# !gdown https://drive.google.com/uc?id=1-RBvPOYycsSpS7rVP0Pqwcbh18lZYDeb
# !unzip /content/BBRegression.zip

# Data 준비
# 함수 가져오기


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):

        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            bbx = member.find('bndbox')
            xmin = int(bbx.find('xmin').text)
            ymin = int(bbx.find('ymin').text)
            xmax = int(bbx.find('xmax').text)
            ymax = int(bbx.find('ymax').text)
            label = member.find('name').text

            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     label,
                     xmin,
                     ymin,
                     xmax,
                     ymax
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height',
                   'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


image_path = "/content/BBRegression"
file_name = "label_cats.csv"
csv_path = os.path.join(image_path, "train")
xml_df = xml_to_csv(csv_path)
xml_df.to_csv(file_name)

images = xml_df.iloc[:, 0].values
points = xml_df.iloc[:, 4:].values

# 시각화

dataset_images = []
dataset_bbs = []

for file, points in zip(images, points):
    f = os.path.join(image_path, "train", file)
    image = PIL.Image.open(f)
    arr = np.array(image)
    dataset_images.append(arr)
    dataset_bbs.append(points)

dataset_images = np.array(dataset_images)
dataset_bbs = np.array(dataset_bbs)

print(dataset_images.shape)
print(dataset_bbs.shape)

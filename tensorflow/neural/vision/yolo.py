# YOLO5 custom dataset
import yaml
from keras.applications import MobileNetV2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, mean_squared_error
import tensorflow as tf
from tensorflow import keras
from keras import layers
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
from keras.utils import to_categorical

# Data 준비
# !curl - L "https://public.roboflow.com/ds/euHstKVJx8?key=NTHyeODDoX" > roboflow.zip
# unzip roboflow.zip
# rm roboflow.zip

# Dataset 파일로 이동, labels 폴더 안에는 bounding box 좌표
# %cd /content
# 안에 Yolov5 모델 다운로드
# !git clone https://github.com/ultralytics/yolov5.git

# Requirements 설치
# %cd /content/yolov5
# !pip install -r requirements.txt

# 학습 데이터 지정 (data.yaml 파일)
# %cd /
from glob import glob
img_list = glob("/content/dataset/export/images/*.jpg")

train_img, val_img = train_test_split(img_list, test_size=0.2)
print(len(train_img), len(val_img))

with open("/content/dataset/train.txt", "w") as f:
    f.write("\n".join(train_img) + "\n")

with open("/content/dataset/val.txt", "w") as f:
    f.write("\n".join(val_img) + "\n")

# yaml loading
with open("/content/dataset/data.yaml", "r") as f:
    data = yaml.load(f, Loader=yaml.FullLoader)

# key에 지정
data["train"] = "/content/dataset/train.txt"
data["val"] = "/content/dataset/val.txt"

# yaml에 dump
with open("/content/dataset/data.yaml", "w") as f:
    yaml.dump(data, f)

# 학습
# %cd / content/yolov5
# !python train.py - -img 416 - -batch 16 - -epochs 50 - -data / content/dataset/data.yaml - -cfg / content/yolov5/models/yolov5s.yaml - -weight yolov5s.pt - -name racoon_yolov5s

# Testing (detect.py)
# val_img_path = val_img[2]
# !python detect.py - -weights / content/yolov5/runs/train/racoon_yolov5s/weights/best.pt - -save-txt - -img 416 - -conf 0.5 - -source "{val_img_path}"

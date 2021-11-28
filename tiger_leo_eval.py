import torch
import torchvision

from glob import glob

from PIL import Image

import pandas as pd

from torchvision import transforms

import numpy as np

from tqdm import tqdm_notebook as tqdm

import matplotlib.pyplot as plt

import pickle

import os

import sys

dataset_path = sys.argv[1]

imgs = glob(os.path.join(dataset_path, '*.jpg'))

df = pd.DataFrame({'path': imgs})

df['img_name'] = df.path.str.split(os.path.sep).str[-1].str.split('.').str[0]

os.system('rm -rf yolo_txt_results')

os.system('python yolov5/detect.py --source "%s" --weights yolov5s_animals.pt --img 640 --iou-thres 0.8 --conf-thres 0.7 --project yolo_txt_results --save-txt' % dataset_path)

labels = glob('yolo_txt_results/exp/labels/*.txt')

df['label'] = 3

for label in labels:
    with open(label, 'r') as f:
        clazz = f.read()[0]
        
    img_name = label.split(os.path.sep)[-1].split('.')[0]
    
    df.loc[df.img_name == img_name, 'label'] = 1 if clazz == '1'else 2
    
result_df = pd.DataFrame({'id': df.img_name.apply(lambda x: x + '.jpg'), 'class': df.label})

result_df.to_csv('labels.csv', index = False)

print('Result saved to labels.csv')
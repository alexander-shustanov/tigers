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


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu:0')

princess_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
princess_model = princess_model.eval()
princess_model.load_state_dict(torch.load('3.pt'))

_ = princess_model.to(device)

with open('princess_features_0.pkl', 'rb') as f:
    princess_features = pickle.load(f)

princess_features = torch.tensor(princess_features)
    
    
test_preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def sim_matrix(a, b, eps=1e-8):
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


dataset_path = sys.argv[1]

imgs = glob(os.path.join(dataset_path, '*.jpg'))

df = pd.DataFrame({'path': imgs})

df['img_name'] = df.path.str.split(os.path.sep).str[-1].str.split('.').str[0]

os.system('rm -rf yolo_detect_results')

os.system('python yolov5/detect.py --source "%s" --weights yolov5s_animals.pt --img 640 --iou-thres 0.8 --conf-thres 0.7 --project yolo_detect_results --save-crop' % dataset_path)

crops = glob('yolo_detect_results/exp/crops/Tiger/*.jpg')

df['label'] = 0

for crop in crops:
    img_name = crop.split(os.path.sep)[-1].split('.')[0]
    
    
    tensor = test_preprocess(Image.open(crop).convert('RGB')).unsqueeze(dim=0).to(device)
    
    with torch.no_grad():
        feature = princess_model(tensor).cpu()
        
    df.loc[df.img_name == img_name, 'label'] = int((sim_matrix(princess_features, feature).mean().item() > 0.67))
    
result_df = pd.DataFrame({'id': df.img_name.apply(lambda x: x + '.jpg'), 'class': df.label})

result_df.to_csv('labels.csv', index = False)

print('Result saved to labels.csv')
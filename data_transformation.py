from pathlib import Path
import os
import gc
import sys
from PIL import Image
import cv2
import math, random
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

from collections import OrderedDict
import boto3
import zipfile
from io import BytesIO
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW

import timm
from transformers import get_cosine_schedule_with_warmup

import albumentations as A
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import requests
from torchvision.datasets import ImageFolder
from torchvision import datasets
import dill as pickle

def transform_data():
    def read_csv_from_s3(csv_filename):
        s3 = boto3.client('s3')
        bucket_name = 'deeplearning-mlops-demo'
        file_key = 'rsna-2024-lumbar-spine-degenerative-classification.zip'
        with BytesIO() as zip_buffer:
            s3.download_fileobj(bucket_name, file_key, zip_buffer)
            zip_buffer.seek(0)
            with zipfile.ZipFile(zip_buffer, 'r') as zip_ref:    
                with zip_ref.open(csv_filename) as csv_file:
                    csv_data = pd.read_csv(csv_file)
                    print("CSV file read successfully.")
                    print(csv_data.head())
        return csv_data

    csv_filename = 'train.csv'
    df = read_csv_from_s3(csv_filename)
    df = df.fillna(-100)
    label2id = {'Normal/Mild': 0, 'Moderate': 1, 'Severe': 2}
    df = df.replace(label2id)
    
    AUG_PROB = 0.75
    IMG_SIZE = [512, 512]
    IN_CHANS = 30
    
    data_transform = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=AUG_PROB),
        A.OneOf([
            A.MotionBlur(blur_limit=5),
            A.MedianBlur(blur_limit=5),
            A.GaussianBlur(blur_limit=5),
            A.GaussNoise(var_limit=(5.0, 30.0)),
        ], p=AUG_PROB),
        A.OneOf([
            A.OpticalDistortion(distort_limit=1.0),
            A.GridDistortion(num_steps=5, distort_limit=1.),
            A.ElasticTransform(alpha=3),
        ], p=AUG_PROB),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=AUG_PROB),
        A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
        A.CoarseDropout(max_holes=16, max_height=64, max_width=64, min_holes=1, min_height=8, min_width=8, p=AUG_PROB),    
        A.Normalize(mean=0.5, std=0.5)
    ])

    class RSNA24Dataset(Dataset):
        def __init__(self, df, phase='train', transform=None):
            self.df = df
            self.transform = transform
            self.phase = phase
            self.set_classes()
            print("===============df : ", df)
        
        def set_classes(self):
            self.classes = {col: idx for idx, col in enumerate(self.df.columns[1:], start=1)}
        
        def __len__(self):
            return len(self.df)
        
        def __getitem__(self, idx):
            x = np.zeros((512, 512, IN_CHANS), dtype=np.uint8)
            t = self.df.iloc[idx]
            st_id = int(t['study_id'])
            label = t.iloc[1:].values.astype(np.int64)
            print("st_id----------", st_id)
            print("label----------", label)
            # Sagittal T1
            for i in range(0, 10, 1):
                try:
                    p = f'./cvt_jpg/{st_id}/Sagittal T1/{i:03d}.jpg'
                    img = Image.open(p).convert('L')
                    img = np.array(img)
                    x[..., i] = img.astype(np.uint8)
                except:
                    print(f'failed to load on {st_id}, Sagittal T1')
                    pass
                
            # Sagittal T2/STIR
            for i in range(0, 10, 1):
                try:
                    p = f'./cvt_jpg/{st_id}/Sagittal T2_STIR/{i:03d}.jpg'
                    img = Image.open(p).convert('L')
                    img = np.array(img)
                    x[..., i+10] = img.astype(np.uint8)
                except:
                    print(f'failed to load on {st_id}, Sagittal T2/STIR')
                    pass
                
            # Axial T2
            axt2 = glob(f'./cvt_jpg/{st_id}/Axial T2/*.jpg')
            axt2 = sorted(axt2)
        
            step = len(axt2) / 10.0
            st = len(axt2)/2.0 - 4.0*step
            end = len(axt2)+0.0001
                    
            for i, j in enumerate(np.arange(st, end, step)):
                try:
                    p = axt2[max(0, int((j-0.5001).round()))]
                    img = Image.open(p).convert('L')
                    img = np.array(img)
                    x[..., i+20] = img.astype(np.uint8)
                except:
                    print(f'failed to load on {st_id}, Sagittal T2/STIR')
                    pass  
                
            assert np.sum(x) > 0
                
            if self.transform is not None:
                x = self.transform(image=x)['image']
      
            x = x.transpose(2, 0, 1)
                    
            return x, label

    model_dataset = RSNA24Dataset(df, phase='train', transform=data_transform)
    img, ann = model_dataset[1]
    print("iiiiiiii", img)
    print("aaaaaaaaa:", ann)
    with open('spine-degenerative-mriscan-classification.pkl', 'wb') as f:
        pickle.dump(model_dataset, f)
    return model_dataset
    
transform_data()

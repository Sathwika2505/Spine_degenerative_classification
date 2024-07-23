import pandas as pd
import boto3
import zipfile
from io import BytesIO
import os
import numpy as np
from PIL import Image
import glob
import torch
from torch.utils.data import Dataset
import albumentations as A
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
    IMG_SIZE = [256, 256]
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
        A.Normalize(mean=0.5, std=0.5,)
    ])

    class RSNA24Dataset(Dataset):
        def __init__(self, df, phase='train', transform=None):
            self.df = df
            self.transform = transform
            self.phase = phase
            self.path = f'{os.getcwd()}/cvt_jpg/'
            self.set_classes()
            #print("===============df : ", df.columns)
            print("Classes after set_classes:", self.classes)  # Debugging statement
        
        def set_classes(self):
            self.classes = {col: idx for idx, col in enumerate(self.df.columns[1:], start=1)}
            print("Classes set in set_classes method:", self.classes)  # Debugging statement
        
        def __len__(self):
            return len(self.df)
        
        def __getitem__(self, idx):
            x = np.zeros((512, 512, IN_CHANS), dtype=np.uint8)
            t = self.df.iloc[idx]
            st_id = int(t['study_id'])
            labels = t.iloc[1:].values.astype(np.int64)
            
            print("st_id----------", st_id)
            print("labels----------", labels)
            
            # Adjust paths based on new file structure
            for ds in ['Sagittal T1', 'Sagittal T2_STIR', 'Axial T2']:
                folder = ds.replace('/', '_')
                allimgs = glob.glob(f'{self.path}/{folder}/{st_id}_*.jpg')
                allimgs = sorted(allimgs)
                
                if len(allimgs) == 0:
                    print(f'{st_id} {ds} has no images')
                    continue
    
                for j, impath in enumerate(allimgs[:10]):
                    img = Image.open(impath).convert('L')
                    img = np.array(img)
                    x[..., j] = img.astype(np.uint8)
                
                if self.transform is not None:
                    x = self.transform(image=x)['image']
              
                x = x.transpose(2, 0, 1)
                
                # Prepare labels in the format "column name: label"
                label_dict = {col: labels[i] for i, col in enumerate(self.df.columns[1:])}
                label_formatted = [(col, label_dict[col]) for col in self.df.columns[1:]]
                #print("label_formatted ============",label_formatted)
                
                return x, label_formatted

    model_dataset = RSNA24Dataset(df, phase='train', transform=data_transform)
    #for i in range(len(model_dataset)):
    img, ann = model_dataset[1]
    print("iiiiiiii", img)
    print("aaaaaaaaa:", ann)
        
    with open('spine-degenerative-mriscan-classification.pkl', 'wb') as f:
        pickle.dump(model_dataset, f)
    return model_dataset

transform_data()

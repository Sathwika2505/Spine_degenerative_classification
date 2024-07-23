import pandas as pd
import boto3
import zipfile
from io import BytesIO
import pydicom
import glob
import os
import numpy as np
import cv2
from tqdm import tqdm
import re
import random
import shutil

# Define paths
rd = os.path.join(os.getcwd(), "extracted_files")
visualization_path = "./visualization"
output_root_path = "./cvt_jpg"

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

csv_filename1 = 'train_label_coordinates.csv' 
csv_filename2 = 'train_series_descriptions.csv' 

dfc = read_csv_from_s3(csv_filename1)
df = read_csv_from_s3(csv_filename2)

print(dfc.head())
print(df['series_description'].value_counts())

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]

def imread_and_imwrite(src_path, dst_path):
    dicom_data = pydicom.dcmread(src_path)
    image = dicom_data.pixel_array
    image = (image - image.min()) / (image.max() - image.min() + 1e-6) * 255
    img = cv2.resize(image, (512, 512), interpolation=cv2.INTER_CUBIC)
    assert img.shape == (512, 512)
    cv2.imwrite(dst_path, img)

desc = list(df['series_description'].unique())
st_ids = df['study_id'].unique()
print(st_ids[:3], len(st_ids))

for idx, si in enumerate(tqdm(st_ids, total=len(st_ids))):
    pdf = df[df['study_id'] == si]
    for ds in desc:
        ds_ = ds.replace('/', '_')
        pdf_ = pdf[pdf['series_description'] == ds]
        os.makedirs(f'{output_root_path}/{ds_}', exist_ok=True)
        allimgs = []
        for i, row in pdf_.iterrows():
            pimgs = glob.glob(f'{rd}/train_images/{row["study_id"]}/{row["series_id"]}/*.dcm')
            pimgs = sorted(pimgs, key=natural_keys)
            allimgs.extend(pimgs)

        if len(allimgs) == 0:
            print(si, ds, 'has no images')
            continue

        if ds == 'Axial T2':
            for j, impath in enumerate(allimgs):
                dst = f'{output_root_path}/{ds_}/{si}_{j:03d}.jpg'
                imread_and_imwrite(impath, dst)

        elif ds == 'Sagittal T2/STIR':
            step = len(allimgs) / 10.0
            st = len(allimgs) / 2.0 - 4.0 * step
            end = len(allimgs) + 0.0001
            for j, i in enumerate(np.arange(st, end, step)):
                dst = f'{output_root_path}/{ds_}/{si}_{j:03d}.jpg'
                ind2 = max(0, int((i - 0.5001).round()))  
                imread_and_imwrite(allimgs[ind2], dst)

            # Debug print
            jpg_files = glob.glob(f'{output_root_path}/{ds_}/*.jpg')
            print(f'Directory: {output_root_path}/{ds_}/')
            print(f'Number of .jpg files: {len(jpg_files)}')

        elif ds == 'Sagittal T1':
            step = len(allimgs) / 10.0
            st = len(allimgs) / 2.0 - 4.0 * step
            end = len(allimgs) + 0.0001
            for j, i in enumerate(np.arange(st, end, step)):
                dst = f'{output_root_path}/{ds_}/{si}_{j:03d}.jpg'
                ind2 = max(0, int((i - 0.5001).round()))
                imread_and_imwrite(allimgs[ind2], dst)

            # Debug print
            jpg_files = glob.glob(f'{output_root_path}/{ds_}/*.jpg')
            if len(jpg_files) != 10:
                print(f'Warning: Expected 10 .jpg files, but found {len(jpg_files)} in {ds_}')
                continue
                
            print(f'Directory: {output_root_path}/{ds_}/')
            print(f'Number of .jpg files: {len(jpg_files)}')

# Select random images from 3 different folders
random_labels = random.sample(desc, 3)

for label in random_labels:
    label_ = label.replace('/', '_')
    label_path = os.path.join(output_root_path, label_)
    if os.path.isdir(label_path):
        jpg_files = [f for f in os.listdir(label_path) if f.endswith('.jpg')]
        
        if jpg_files:  # Ensure the folder is not empty
            random_file = random.choice(jpg_files)  # Select one random file
            
            jpg_path = os.path.join(label_path, random_file)
            new_jpg_path = os.path.join(os.getcwd(), f"{label_}_{random_file}")
            
            # Copy the JPG file to the current working directory
            shutil.copy2(jpg_path, new_jpg_path)
            print(f"Copied {jpg_path} to {new_jpg_path}")

print("Script finished successfully.")

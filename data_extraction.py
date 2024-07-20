import boto3
import zipfile
from io import BytesIO
import pandas as pd

def extract_data_and_read_csv(folder_to_extract, csv_filename, extract_to_folder):
    s3 = boto3.client('s3')
    bucket_name = 'deeplearning-mlops-demo'
    file_key = 'rsna-2024-lumbar-spine-degenerative-classification.zip'  # Ensure this key is correct
    
    try:
        with BytesIO() as zip_buffer:
            s3.download_fileobj(bucket_name, file_key, zip_buffer)
            zip_buffer.seek(0)
            
            with zipfile.ZipFile(zip_buffer, 'r') as zip_ref:
                all_files = zip_ref.namelist()
                #print("all_files : ", all_files)
                files_in_folder = [f for f in all_files if f.startswith(folder_to_extract)]
              
                for file in files_in_folder:
                    zip_ref.extract(file, path=extract_to_folder)
                    
                with zip_ref.open(csv_filename) as csv_file:
                    csv_data = pd.read_csv(csv_file)
                    print("CSV file read successfully.")
                    print(csv_data.head())
                
        if csv_data is None:
            print(f"CSV file '{csv_filename}' not found in folder '{folder_to_extract}'.")
        else:
            print("Data access complete.")
            return extract_to_folder, csv_data
        
    except Exception as e:
        print(f"An error occurred: {e}")
        
    return None
    
folder_to_extract = 'train_images/'
csv_filename = 'train_label_coordinates.csv'
extract_to_folder = './extracted_files/'
extract_data_and_read_csv(folder_to_extract, csv_filename, extract_to_folder)


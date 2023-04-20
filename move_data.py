import os
import shutil

folder_path = '/home/hoangnt107/Downloads/dataset_barcode_landmark_v2.0'
data_folder = '/home/hoangnt107/Desktop/3code'
 
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    if filename.endswith('.jpg') or filename.endswith('.png'):
        json_filename = os.path.splitext(filename)[0] + '.json'
        json_path = os.path.join(folder_path, json_filename)
        if os.path.exists(json_path):
            shutil.move(file_path, os.path.join(data_folder, filename))
            shutil.move(json_path, os.path.join(data_folder, json_filename))
            print(f'move {filename} and {json_filename}')
    else:
        print(f'{filename} have not a pair file JSON, Image ')

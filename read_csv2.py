import pandas as pd
import os
import requests
import cv2
import numpy as np

# Đọc file csv và lọc dữ liệu dựa trên 3 filter log_time, open_type, station_id 
df = pd.read_csv('/home/hoangnt107/Downloads/ALPR-Result_2023_03_20.csv')
condition = (df['log_time'].str.startswith('2023-03-19')) & (df['open_type'] == 'AUTO') & (df['station_id'] == 17801)
filtered_df = df[condition]

# Tạo folder
folder_path = '/home/hoangnt107/Downloads/ABC'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

for index, row in filtered_df.iterrows():
    image_url_tmp = row['license_plate_images']
    image_url = image_url_tmp.replace("[","").replace("]","").replace("'", "")
    print(image_url)
    response = requests.get(image_url)
    img_array = np.array(bytearray(response.content), dtype=np.uint8)
    img = cv2.imdecode(img_array, -1)
    h, w = img.shape[:2]

    # Vẽ một khoảng màu đen phía dưới ảnh
    black_bar_height = 50
    black_bar = np.zeros((black_bar_height, w, 3), np.uint8)
    img_with_black_bar = np.concatenate((img, black_bar), axis=0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    thickness = 2
    color = (255, 255, 0)  
    text_size = cv2.getTextSize(str(row['license_plate']), font, fontScale, thickness)[0]

    text_width, text_height = text_size[0], text_size[1]
    text_x = int((w - text_width) / 2)  # đặt tọa độ x để văn bản nằm chính giữa
    text_y = int(h + black_bar_height - (black_bar_height - text_size[1]) / 2)  # đặt tọa độ y để văn bản nằm ở phía dưới ảnh

    # vẽ license_plate lên khoảng đen vừa vẽ và lưu ảnh vào folder
    cv2.putText(img_with_black_bar, str(row['license_plate']), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    file_path = os.path.join(folder_path, f'{row["license_plate"]}.jpg')
    cv2.imwrite(file_path, img_with_black_bar)

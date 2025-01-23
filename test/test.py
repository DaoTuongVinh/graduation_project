# import os
# import time
# import cv2
# import pandas as pd
#
# from merge import MERGE
#
# model = MERGE()
#
# dir_image = '/home/vinhdt/Desktop/data_test/2024'
#
# data_excel = []
#
# for i, file in enumerate(os.listdir(dir_image)):
#     file_image = os.path.join(dir_image, file)
#     image = cv2.imread(file_image)
#     start = time.time()
#     data = model.run(image)
#     end = time.time()
#     if data['code'] == 403:
#         data_person = data['data'][0]
#     #     data_excel.append((i, file, data_person['id'], data_person['hoten'], data_person['ngaysinh'], data_person['gioitinh'], data_person['quoctich'], data_person['quequan'], data_person['diachi'], data_person['giatriden']))
#     # else:
#     #     data_excel.append((i, file, '', '', '', '', '', '', '', ''))
#
#         data_excel.append((i, file, data_person['id'], data_person['hoten'], data_person['ngaysinh'], data_person['gioitinh'], data_person['quoctich']))
#     else:
#         data_excel.append((i, file, '', '', '', '', ''))
#
# # df = pd.DataFrame(data_excel, columns=['stt', 'link', 'id', 'hoten', 'ngaysinh', 'gioitinh', 'quoctich', 'quequan', 'diachi', 'giatriden'])
# df = pd.DataFrame(data_excel, columns=['stt', 'link', 'id', 'hoten', 'ngaysinh', 'gioitinh', 'quoctich'])
# df.to_excel('/home/vinhdt/Desktop/data_test/predict_2024.xlsx', index=False)
#
#

import cv2

image = cv2.imread('/home/dtvingg/Desktop/Doan_20241/data/original.jpg')

image_90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
image_180 = cv2.rotate(image_90, cv2.ROTATE_90_CLOCKWISE)
image_270 = cv2.rotate(image_180, cv2.ROTATE_90_CLOCKWISE)

cv2.imwrite('/home/dtvingg/Desktop/Doan_20241/data/rotate_90.jpg', image_90)
cv2.imwrite('/home/dtvingg/Desktop/Doan_20241/data/rotate_180.jpg', image_180)
cv2.imwrite('/home/dtvingg/Desktop/Doan_20241/data/rotate_270.jpg', image_270)
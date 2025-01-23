# import os
# import shutil
#
# dir_image = '/home/vinhdt/Documents/CVS/data/doan/phat_hien_the/images/2024'
# dir_xml = '/home/vinhdt/Desktop/data_train_cccd_b1/xml'
# dir_save = '/home/vinhdt/Documents/CVS/data/doan/phat_hien_the/xml/2024'
#
# for file in os.listdir(dir_image):
#     if 'rotate' not in file:
#         continue
#     file_xml = dir_xml + '/' + file.replace('.jpg', '.xml')
#     shutil.move(file_xml, dir_save)

import os
import shutil

dir_original = '/home/vinhdt/Documents/CVS/data/doan/nhan_dien_ky_tu/data'
dir_select = '/home/vinhdt/Documents/CVS/data/doan/nhan_dien_ky_tu/data_'

for folder in os.listdir(dir_original):
    dir_folder = os.path.join(dir_original, folder)
    dir_folder_select = os.path.join(dir_select, folder)
    if not os.path.exists(dir_folder_select):
        os.makedirs(dir_folder_select)
    for i, file in enumerate(os.listdir(dir_folder)):
        if i % 50 == 0:
            file_image = os.path.join(dir_folder, file)
            shutil.move(file_image, dir_folder_select)
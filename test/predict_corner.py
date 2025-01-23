import time

import cv2
from ultralytics import YOLO
from utils import *
import os

class CORNER_DETECTOR(object):
    def __init__(self):
        self.weight = '/home/dtvingg/Desktop/Doan_20241/code/20204705_DaoTuongVinh_20241/weight/weight_corner_b2_yolov8m_v2.pt'
        self.model = self.load_model()

    def load_model(self):
        model = YOLO(self.weight)
        return model

    def detect_corner(self, image, thres=0.5):
        image = add_padding(image)
        height, width, _ = image.shape

        results = self.model.predict(image, save=False, conf=thres)

        dict_corner = get_corner(results)

        dict_center = get_center_point(dict_corner, height, width)

        if len(dict_center) <= 2:
            return None

        dict_center_all = calculate_missed_coord_corner(dict_center)
        # perspective transform
        maxWidth, maxHeight, M = perspective_transform(dict_center_all)
        image = cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags=cv2.INTER_LINEAR)

        return image

model = CORNER_DETECTOR()

dir_image = '/home/dtvingg/Desktop/Doan_20241/data/test'

for  file in os.listdir(dir_image):
    file_image = os.path.join(dir_image, file)
    image = cv2.imread(file_image)
    start =time.time()
    image_tra = model.detect_corner(image)
    end = time.time()
    print(end-start)


# link_image = '/home/dtvingg/Desktop/Doan_20241/data'
#
# image = cv2.imread(link_image)
#
# image_tra = model.detect_corner(image)
#
# cv2.imwrite('/home/vinhdt/Desktop/image_doan/test_b2.jpg', image_tra)

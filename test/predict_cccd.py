import os
import time

import cv2
from ultralytics import YOLO
from utils import *

def crop_cccd(results):
    list_box_cccd  = []

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box dưới dạng [x_min, y_min, x_max, y_max]
        scores = result.boxes.conf.cpu().numpy()  # Điểm tin cậy cho từng bounding box

        for box, score in zip(boxes, scores):
            x_min = math.floor(box[0])
            y_min = math.floor(box[1])
            x_max = math.ceil(box[2])
            y_max = math.ceil(box[3])
            list_box_cccd.append([x_min, y_min, x_max, y_max, score])

    if len(list_box_cccd) == 0:
        return list_box_cccd

    # check box
    if len(list_box_cccd) == 1:
        list_box_cccd = list_box_cccd
    else:
        list_box_cccd = non_max_suppression(list_box_cccd, 0.3)
        list_box_cccd = sorted(list_box_cccd, key=lambda x:x[1])

    return list_box_cccd

def crop_cccd_(image, results):
    list_box_cccd  = []
    list_cccd_crop = []

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box dưới dạng [x_min, y_min, x_max, y_max]
        scores = result.boxes.conf.cpu().numpy()  # Điểm tin cậy cho từng bounding box

        for box, score in zip(boxes, scores):
            x_min = math.floor(box[0])
            y_min = math.floor(box[1])
            x_max = math.ceil(box[2])
            y_max = math.ceil(box[3])
            list_box_cccd.append([x_min, y_min, x_max, y_max, score])

    if len(list_box_cccd) == 0:
        return list_cccd_crop

    # check box
    if len(list_box_cccd) == 1:
        list_box_cccd = list_box_cccd
    else:
        list_box_cccd = non_max_suppression(list_box_cccd, 0.3)

    for (x1, y1, x2, y2, score) in list_box_cccd:
        cropped_image = image[y1:y2, x1:x2]
        list_cccd_crop.append(cropped_image)

    return list_cccd_crop

class CCCD_DETECTOR(object):
    def __init__(self):
        self.weight = '/home/dtvingg/Desktop/Doan_20241/code/20204705_DaoTuongVinh_20241/weight/weight_cccd_b1_yolov8m_v2.pt'
        self.model = self.load_model()

    def load_model(self):
        model = YOLO(self.weight)
        return model

    def detect_cccd(self, image, thres=0.5):
        results = self.model.predict(image, save=False, conf=thres)

        list_box_cccd = crop_cccd(results)

        list_cccd = crop_cccd_(image, results)

        return list_box_cccd, list_cccd

model = CCCD_DETECTOR()

image_path = '/home/dtvingg/Desktop/Doan_20241/data/original.jpg'
image = cv2.imread(image_path)
_, list_cccd = model.detect_cccd(image)
for cccd in list_cccd:
    cccd = add_padding(cccd)
    cv2.imwrite('/home/dtvingg/Desktop/Doan_20241/data/b1.jpg', cccd)

# dir_image = '/home/dtvingg/Desktop/Doan_20241/data/phat_hien_the/images/chip'
# dir_check = '/home/vinhdt/Desktop/data_train_cccd_b1/check'
# for i, file in enumerate(os.listdir(dir_image)[:100]):
#     print(f'-{i}---------------------------')
#     file_image = dir_image + '/' + file
#     file_check = dir_check + '/' + file
#     image = cv2.imread(file_image)
#     start = time.time()
#     data_predict = model.detect_cccd(image)
#     end = time.time()
#     print(end-start)
    # for item in data_predict:
    #     box = item[:-1]
    #     score = float(item[-1])
    #     score_ = round(score, 2)
    #     cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    #     cv2.putText(image, f'CCCD {score_}', (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # for i, cccd in enumerate(list_cccd):
    #     cv2.imwrite(f'/home/vinhdt/Desktop/image_doan/test_{i}.jpg', cccd)
    #
    #
    # cv2.imwrite(file_check, image)
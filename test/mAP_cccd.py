import os
from ultralytics import YOLO
import xml.etree.ElementTree as ET
from utils import *

def crop_cccd(image, results):
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

def sort_box(list_result):
    def group_and_sort_boxes_by_line(boxes, threshold=10):
        # Sort boxes by their y1 coordinate
        boxes.sort(key=lambda x: x[1])

        lines = []
        current_line = [boxes[0]]

        for box in boxes[1:]:
            # If the y1 coordinate of the current box is close to the last box in the current line
            if abs(box[1] - current_line[-1][1]) <= threshold:
                current_line.append(box)
            else:
                # Sort the current line by x1 coordinate before adding to lines
                current_line.sort(key=lambda b: b[0])
                lines.append(current_line)
                current_line = [box]

        # Append the last line and sort it by x1 coordinate
        if current_line:
            current_line.sort(key=lambda b: b[0])
            lines.append(current_line)

        return lines

    def sort_boxes(boxes, threshold=10):
        # Group and sort boxes by line
        lines = group_and_sort_boxes_by_line(boxes, threshold)

        # Flatten the list of lines into a single list
        sorted_boxes = [box for line in lines for box in line]

        return sorted_boxes

    list_result = sort_boxes(list_result)
    return list_result

def extract_bboxes_from_xml(xml_file):
    # Đọc và phân tích file XML
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Danh sách chứa các bbox
    bboxes = []

    # Duyệt qua tất cả các object
    for obj in root.findall("object"):
        bndbox = obj.find("bndbox")

        # Lấy tọa độ bbox
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)

        # Thêm bbox vào danh sách
        bboxes.append([xmin, ymin, xmax, ymax])

    # Sắp xếp bbox từ trên xuống dưới, từ trái sang phải
    bboxes = sorted(bboxes, key=lambda x: x[1])

    return bboxes

def iou(box1, box2):
    # Tính Intersection over Union (IoU)
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area

class CCCD_DETECTOR(object):
    def __init__(self):
        self.weight = '/home/dtvingg/Desktop/Doan_20241/code/20204705_DaoTuongVinh_20241/weight/weight_cccd_b1_yolov8m_v2.pt'
        self.model = self.load_model()

    def load_model(self):
        model = YOLO(self.weight)
        return model

    def detect_cccd(self, image, thres=0.5):
        results = self.model.predict(image, save=False, conf=thres)

        list_box_cccd = crop_cccd(image, results)

        return list_box_cccd

model = CCCD_DETECTOR()

dir_image = '/home/dtvingg/Desktop/Doan_20241/data/phat_hien_the/images/chip'
dir_xml = '/home/dtvingg/Desktop/Doan_20241/data/phat_hien_the/xml/chip'

cnt = 0
all = 0
for i, file in enumerate(os.listdir(dir_image)[:200]):
    print(f'-{i}---------------------------')
    file_image = dir_image + '/' + file
    file_xml = dir_xml + '/' + file.replace('.jpg', '.xml')
    image = cv2.imread(file_image)
    data_predict = model.detect_cccd(image)
    data_xml = extract_bboxes_from_xml(file_xml)
    all += len(data_xml)
    if len(data_predict) == len(data_xml) and len(data_predict) != 0:
        for id_box in range(len(data_predict)):
            box_predict = data_predict[id_box]
            box_xml = data_xml[id_box]
            score_iou = iou(box_xml, box_predict)
            if score_iou >= 0.9:
                cnt += 1
    print(cnt/all)

print(cnt, all, cnt/all)




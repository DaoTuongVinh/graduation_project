from ultralytics import YOLO
import xml.etree.ElementTree as ET
from utils import *
import os

def parse_xml_to_dict(xml_file):
    # Đọc và phân tích tệp XML
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Dictionary để lưu kết quả
    object_dict = {}

    # Duyệt qua từng object trong XML
    for obj in root.findall("object"):
        name = obj.find("name").text  # Lấy tên object
        bndbox = obj.find("bndbox")

        # Lấy tọa độ bbox
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)

        bbox = [xmin, ymin, xmax, ymax]  # Định dạng bbox

        # Thêm bbox vào dict theo tên object
        if name not in object_dict:
            object_dict[name] = bbox

    return object_dict

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

class CORNER_DETECTOR(object):
    def __init__(self):
        self.weight = '/home/vinhdt/Documents/CVS/code/20204705_DaoTuongVinh_20241/weight/weight_corner_b2_yolov8m_v2.pt'
        self.model = self.load_model()

    def load_model(self):
        model = YOLO(self.weight)
        return model

    def detect_corner(self, image, thres=0.5):
        height, width, _ = image.shape

        results = self.model.predict(image, save=False, conf=thres)

        dict_corner = get_corner(results)

        return dict_corner

model = CORNER_DETECTOR()

list_corner = ['top_left', 'top_right', 'bottom_left', 'bottom_right']

dir_image = '/home/vinhdt/Desktop/data_b2/images'
dir_xml = '/home/vinhdt/Desktop/data_b2/xml'

for field in list_corner:
    cnt = 0
    all = 0
    for i, file in enumerate(os.listdir(dir_image)[:100]):
        print(f'--{field}---{i}---------------------')
        file_image = dir_image + '/' + file
        file_xml = dir_xml + '/' + file.replace('.jpg', '.xml')
        image = cv2.imread(file_image)
        data_predict = model.detect_corner(image)
        data_xml = parse_xml_to_dict(file_xml)
        if field not in data_predict.keys():
            data_predict[field] = []
        if field not in data_xml.keys():
            data_xml[field] = []
        field_predict = data_predict[field]
        field_xml = data_xml[field]
        all += 1
        if len(field_predict) == len(field_xml) and len(field_predict) != 0:
            score_iou = iou(field_predict, field_xml)
            if score_iou >= 0.5:
                cnt += 1
        print(cnt/all)

    print('+++++++', field, cnt, all, cnt/all)
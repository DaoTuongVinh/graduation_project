import os
import cv2
import xml.etree.ElementTree as ET

import warnings
warnings.filterwarnings('ignore')

from infor_detector import INFOR_DETECTOR

model_infor_chip = INFOR_DETECTOR('infor_chip')
model_infor_2024 = INFOR_DETECTOR('infor_2024')

dir_image = '/home/vinhdt/Desktop/data_infor_2024/images'
dir_xml = '/home/vinhdt/Desktop/data_infor_2024/xml'

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
            object_dict[name] = []
        object_dict[name].append(bbox)

    for key in object_dict:
        if len(object_dict[key]) != 0:
            object_dict[key] = sort_box(object_dict[key])

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

list_field = ['id', 'hoten', 'ngaysinh', 'gioitinh', 'quoctich']
# list_field = ['id', 'hoten', 'ngaysinh', 'gioitinh', 'quoctich', 'diachi', 'quequan', 'giatriden']

for field in list_field:
    cnt = 0
    all = 0
    for i, file in enumerate(os.listdir(dir_image)):
        file_image = dir_image + '/' + file
        file_xml = dir_xml + '/' + file.replace('.jpg', '.xml')
        image = cv2.imread(file_image)
        data_predict = model_infor_2024.run(image)
        data_xml = parse_xml_to_dict(file_xml)
        if field not in data_predict.keys():
            data_predict[field] = []
        if field not in data_xml.keys():
            data_xml[field] = []
        field_predict = data_predict[field]
        field_xml = data_xml[field]
        all += max(len(field_predict), len(field_xml))
        if len(field_predict) == len(field_xml) and len(field_predict) != 0:
            for id_box in range(len(field_predict)):
                box_predict = field_predict[id_box]
                box_xml = field_xml[id_box]
                score_iou = iou(box_xml, box_predict)
                if score_iou >= 0.5:
                    cnt += 1

    print(field, cnt, all, cnt/all)




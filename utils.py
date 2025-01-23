import cv2
import math
import numpy as np
import torch

def add_padding(image):
    top = 50
    bottom = 50
    left = 50
    right = 50
    border_color = (255, 255, 255)
    padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=border_color)

    return padded_image

#check box trung nhau, long nhau
def calculate_iou(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1[:4]
    x3, y3, x4, y4 = bbox2[:4]

    # Compute intersection area
    xi1, yi1 = max(x1, x3), max(y1, y3)
    xi2, yi2 = min(x2, x4), min(y2, y4)
    intersection_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    # Compute union area
    bbox1_area = (x2 - x1) * (y2 - y1)
    bbox2_area = (x4 - x3) * (y4 - y3)
    union_area = bbox1_area + bbox2_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area if union_area > 0 else 0.0
    return iou

def merge_bboxes(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1[:4]
    x3, y3, x4, y4 = bbox2[:4]

    # Tính toán các tọa độ của hộp bao quanh hợp nhất
    merged_x1 = min(x1, x3)
    merged_y1 = min(y1, y3)
    merged_x2 = max(x2, x4)
    merged_y2 = max(y2, y4)

    # Tạo hộp bao quanh hợp nhất với điểm số độ tin cậy cao nhất
    merged_confidence = max(bbox1[4], bbox2[4])
    return [merged_x1, merged_y1, merged_x2, merged_y2, merged_confidence]

def non_max_suppression(bboxes, iou_threshold):
    bboxes = sorted(bboxes, key=lambda x: x[4], reverse=True)  # Sắp xếp theo confidence giảm dần
    selected_bboxes = []

    while bboxes:
        current_bbox = bboxes.pop(0)
        temp_bboxes = []

        for bbox in bboxes:
            if calculate_iou(current_bbox, bbox) >= iou_threshold:
                # Gộp hai hộp bao quanh nếu chúng giao nhau nhiều hơn ngưỡng
                current_bbox = merge_bboxes(current_bbox, bbox)
            else:
                temp_bboxes.append(bbox)

        selected_bboxes.append(current_bbox)
        bboxes = temp_bboxes

    return selected_bboxes

def crop_cccd(image, results):
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

def check_duplicate_in_corner(dict_corner, iou_threshold=0.3):
    labels = list(dict_corner.keys())
    to_remove = set()

    # So sánh từng cặp bounding box trong dict_corner
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            label1, label2 = labels[i], labels[j]
            bbox1, score1 = dict_corner[label1]
            bbox2, score2 = dict_corner[label2]

            # Tính IoU giữa hai bounding box
            iou = calculate_iou(bbox1, bbox2)

            # Nếu IoU lớn hơn ngưỡng và box trùng/lồng nhau
            if iou >= iou_threshold:
                # Giữ lại box có score lớn hơn, xóa box có score nhỏ hơn
                if score1 > score2:
                    to_remove.add(label2)
                else:
                    to_remove.add(label1)

    # Tạo dict mới không chứa các box bị loại bỏ
    filtered_dict = {label: dict_corner[label][0] for label in dict_corner if label not in to_remove}

    return filtered_dict


def get_corner(results):
    dict_corner = dict()

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box dưới dạng [x_min, y_min, x_max, y_max]
        class_ids = result.boxes.cls.cpu().numpy()  # Nhãn (id) của từng bounding box
        scores = result.boxes.conf.cpu().numpy()  # Độ tin cậy (confidence score) của từng bounding box

        for box, class_id, score in zip(boxes, class_ids, scores):
            label_name = result.names[int(class_id)]  # Chuyển ID thành tên nhãn

            # Lấy tọa độ và chuyển về int
            x_min, y_min, x_max, y_max = [int(coord) for coord in box]

            # Kiểm tra nếu label chưa có trong dict hoặc score hiện tại cao hơn score đã lưu
            if label_name not in dict_corner or score > dict_corner[label_name][1]:
                dict_corner[label_name] = ([x_min, y_min, x_max, y_max], score)

    dict_corner_select = check_duplicate_in_corner(dict_corner)

    return dict_corner_select

def get_center_point(dict_corner_select, height, width):
    dict_center = dict()
    height2 = height/2
    width2 = width/2
    for key in dict_corner_select.keys():
        [xmin, ymin, xmax, ymax] = dict_corner_select[key]
        x_center = (xmin + xmax) / 2
        y_center = (ymin + ymax) / 2
        # if xmin < width2 and ymin < height2:
        #     x_center = x_center - 0.2 * (xmax - xmin)
        #     y_center = y_center - 0.2 * (ymax - ymin)
        # elif xmin > width2 and ymin < height2:
        #     x_center = x_center + 0.2 * (xmax - xmin)
        #     y_center = y_center - 0.2 * (ymax - ymin)
        # elif xmin > width2 and ymin > height2:
        #     x_center = x_center + 0.2 * (xmax - xmin)
        #     y_center = y_center + 0.2 * (ymax - ymin)
        # elif xmin < width2 and ymin > height2:
        #     x_center = x_center - 0.2 * (xmax - xmin)
        #     y_center = y_center + 0.2 * (ymax - ymin)
        # else:
        #     x_center = x_center
        #     y_center = y_center

        dict_center[key] = (x_center, y_center)

    return dict_center

def find_miss_corner(dict_center):
    position_name = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
    position_index = np.array([0, 0, 0, 0])

    for name in dict_center.keys():
        if name in position_name:
            position_index[position_name.index(name)] = 1

    index = np.argmin(position_index)

    return index

def calculate_missed_coord_corner(dict_center):
    thresh = 0

    index = find_miss_corner(dict_center)

    # calculate missed corner coordinate
    # case 1: missed corner is "top_left"
    if index == 0:
        midpoint = np.add(dict_center['top_right'], dict_center['bottom_left']) / 2
        y = 2 * midpoint[1] - dict_center['bottom_right'][1] - thresh
        x = 2 * midpoint[0] - dict_center['bottom_right'][0] - thresh
        dict_center['top_left'] = (x, y)
    elif index == 1:  # "top_right"
        midpoint = np.add(dict_center['top_left'], dict_center['bottom_right']) / 2
        y = 2 * midpoint[1] - dict_center['bottom_left'][1] - thresh
        x = 2 * midpoint[0] - dict_center['bottom_left'][0] - thresh
        dict_center['top_right'] = (x, y)
    elif index == 2:  # "bottom_left"
        midpoint = np.add(dict_center['top_left'], dict_center['bottom_right']) / 2
        y = 2 * midpoint[1] - dict_center['top_right'][1] - thresh
        x = 2 * midpoint[0] - dict_center['top_right'][0] - thresh
        dict_center['bottom_left'] = (x, y)
    elif index == 3:  # "bottom_right"
        midpoint = np.add(dict_center['bottom_left'], dict_center['top_right']) / 2
        y = 2 * midpoint[1] - dict_center['top_left'][1] - thresh
        x = 2 * midpoint[0] - dict_center['top_left'][0] - thresh
        dict_center['bottom_right'] = (x, y)

    return dict_center

def perspective_transform(dict_center_corner):

    pt_A = list(dict_center_corner['top_left'])
    pt_D = list(dict_center_corner['top_right'])
    pt_B = list(dict_center_corner['bottom_left'])
    pt_C = list(dict_center_corner['bottom_right'])

    # Here, I have used L2 norm. You can use L1 also.
    width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[0] - pt_D[0]) ** 2))
    width_BC = np.sqrt(((pt_B[1] - pt_C[1]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
    maxWidth = max(int(width_AD), int(width_BC))

    height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[0] - pt_B[0]) ** 2))
    height_CD = np.sqrt(((pt_C[1] - pt_D[1]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
    maxHeight = max(int(height_AB), int(height_CD))

    input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])

    output_pts = np.float32([[1, 1],
                             [1, maxHeight - 1],
                             [maxWidth - 1, maxHeight - 1],
                             [maxWidth - 1, 1]])

    M = cv2.getPerspectiveTransform(input_pts, output_pts)

    return maxWidth, maxHeight, M

def get_box(results, thresh):
    list_result = []
    for result in results:
        x1, y1, x2, y2, confidence = result
        if confidence > thresh:
            list_result.append((int(x1), int(y1), int(x2), int(y2), confidence))

    return list_result

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

def crop_box(image, list_box):
    list_crop = []
    for box in list_box:
        x1, y1, x2, y2, confidence = box
        cropped_image = image[int(y1):int(y2), int(x1):int(x2)]
        list_crop.append(cropped_image)

    return list_crop

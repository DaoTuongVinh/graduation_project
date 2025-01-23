import xml.etree.ElementTree as ET
from utils import *

def parse_xml_to_dict(xml_file):
    # Parse file XML
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Dictionary để lưu kết quả
    result = {}

    # Duyệt qua từng object trong XML
    for obj in root.findall(".//object"):
        # Lấy tên object
        name = obj.find("name").text

        # Lấy tọa độ xmin, ymin từ bounding box
        bbox = obj.find("bndbox")
        xmin = int(float(bbox.find("xmin").text))
        ymin = int(float(bbox.find("ymin").text))

        # Gán vào dictionary
        result[name] = (xmin, ymin)

    return result


file_image = '/home/vinhdt/Desktop/image_doan/test_0.jpg'
file_xml = '/home/vinhdt/Desktop/image_doan/test_0.xml'

dict_center = parse_xml_to_dict(file_xml)
image = cv2.imread(file_image)
# perspective transform
maxWidth, maxHeight, M = perspective_transform(dict_center)
image_crop = cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags=cv2.INTER_LINEAR)
cv2.imwrite('/home/vinhdt/Desktop/image_doan/test_b2.jpg', image_crop)


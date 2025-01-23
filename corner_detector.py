from ultralytics import YOLO

from utils import *

class CORNER_DETECTOR(object):
    def __init__(self):
        self.weight = 'weight/weight_corner_b2_yolov8m_v2.pt'
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

if __name__ == "__main__":
    file_image = 'output_0.jpg'
    image = cv2.imread(file_image)
    image  = CORNER_DETECTOR().detect_corner(image)
    cv2.imwrite('output_b2.jpg', image)

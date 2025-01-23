from ultralytics import YOLO

from utils import *

class CCCD_DETECTOR():
    def __init__(self):
        self.weight = 'weight/weight_cccd_b1_yolov8m_v2.pt'
        self.model = self.load_model()

    def load_model(self):
        model = YOLO(self.weight)
        return model

    def detect_cccd(self, image, thres=0.5):
        image = add_padding(image)

        results = self.model.predict(image, save=False, conf=thres)

        list_cccd_crop = crop_cccd(image, results)

        return list_cccd_crop

if __name__ == "__main__":
    file_image = '/home/dtvingg/Desktop/test/450135959_1025699059013882_8333558522755076836_n.jpg'
    image = cv2.imread(file_image)
    list_cccd_crop  = CCCD_DETECTOR().detect_cccd(image)
    if len(list_cccd_crop) == 0:
        print('ẢNh không có căn cước')
    else:
        for i, cccd in enumerate(list_cccd_crop):
            cv2.imwrite('output_{}.jpg'.format(i), cccd)

import glob
import joblib
import cv2
import warnings
warnings.filterwarnings('ignore')

class CCCD_CLASSIFICATION(object):
    def __init__(self):
        self.weight = '/home/vinhdt/Documents/CVS/code/20204705_DaoTuongVinh_20241/weight/svm_model.joblib'
        self.model = self.load_model()

    def load_model(self):
        model = joblib.load(self.weight)
        return model

    def classification_cccd(self, image):
        image = cv2.resize(image, (64, 64)).flatten()
        prediction = self.model.predict([image])
        if prediction[0] == 0:
            return 'cccd_2024'
        else:
            return 'cccd_chip'

if __name__ == "__main__":
    # file_image = 'output_b2.jpg'
    # image = cv2.imread(file_image)
    # label  = CCCD_CLASSIFICATION().classification_cccd(image)
    # print(label)

    dir_image_chip = '/home/vinhdt/Desktop/data_infor_chip/images'
    dir_image_2024 = '/home/vinhdt/Desktop/data_infor_2024/images'

    list_link_chip = glob.glob(dir_image_chip + '/*.jpg', recursive=False)
    list_link_2024 = glob.glob(dir_image_2024 + '/*.jpg', recursive=False)
    list_link = list_link_chip + list_link_2024
    cnt = 0
    all = len(list_link)
    for link in list_link:
        if 'data_infor_chip' in link:
            label = 'cccd_chip'
        else:
            label = 'cccd_2024'
        image = cv2.imread(link)
        label_predict = CCCD_CLASSIFICATION().classification_cccd(image)
        if label_predict != label:
            print(link)
            cnt += 1

    print((all-cnt)/all)
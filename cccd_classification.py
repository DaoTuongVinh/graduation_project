import joblib
import cv2

class CCCD_CLASSIFICATION(object):
    def __init__(self):
        self.weight = 'weight/svm_model.joblib'
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
    file_image = 'output_b2.jpg'
    image = cv2.imread(file_image)
    label  = CCCD_CLASSIFICATION().classification_cccd(image)
    print(label)
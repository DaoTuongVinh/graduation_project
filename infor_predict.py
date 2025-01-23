import torch
import cv2

from PIL import Image
from torch.autograd import Variable

from config import get_config
from CRNN.models.crnn import CRNN
from CRNN.utils import strLabelConverter
from CRNN import dataset

class INFOR_PREDICT(object):
    def __init__(self, name_config):
        self.config = get_config(name_config)
        self.model = self.load_model_crnn()
        self.converter = strLabelConverter(self.config['alphabet'])
        self.transformer = dataset.resizeNormalize((100, 32))


    def load_model_crnn(self):
        nclass = len(self.config['alphabet']) + 1
        model = CRNN(self.config['imgH'], self.config['nc'], nclass, self.config['nh'])
        model = model.to(self.config['device'])
        model.load_state_dict(torch.load(self.config['weight'], map_location=torch.device('cpu')))

        return model

    def predict_infor(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = Image.fromarray(image)
        image = self.transformer(image)

        if self.config['device'] == 'cuda':
            image = image.cuda()

        image = image.view(1, *image.size())
        image = Variable(image)

        preds = self.model(image)
        scores = torch.nn.functional.softmax(preds, dim=2)
        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        preds_size = Variable(torch.LongTensor([preds.size(0)]))

        # Decode prediction
        raw_pred = self.converter.decode(preds.data, preds_size.data, raw=True)
        sim_pred = self.converter.decode(preds.data, preds_size.data, raw=False)

        # Lấy điểm số của ký tự dự đoán cuối cùng
        final_score = scores[-1]  # Lấy điểm số của ký tự cuối cùng (hoặc bạn có thể chọn ký tự khác)
        final_score = final_score.max().item()  # Lấy giá trị tối đa trong scores, nằm trong khoảng [0, 1]

        return sim_pred



import time
import unidecode
import os
import glob
import pandas as pd

from utils import *
from correct_address.utils import correct_all

from cccd_detector import CCCD_DETECTOR
from corner_detector import CORNER_DETECTOR
from infor_detector import INFOR_DETECTOR
from cccd_classification import CCCD_CLASSIFICATION
from infor_predict import INFOR_PREDICT

class MERGE(object):
    def __init__(self,):
        self.model_cccd = CCCD_DETECTOR()
        self.model_corner = CORNER_DETECTOR()
        self.model_classification = CCCD_CLASSIFICATION()
        self.model_infor_chip = INFOR_DETECTOR('infor_chip')
        self.model_infor_2024 = INFOR_DETECTOR('infor_2024')
        self.model_predict_text = INFOR_PREDICT('crnn_chu')
        self.model_predict_number = INFOR_PREDICT('crnn_so')

    def run(self, image):
        data = {
            'data': [],
            'code': 400
        }
        list_cccd = self.model_cccd.detect_cccd(image)
        if len(list_cccd) == 0:
            data['code'] = 401
            return data

        for idx_cccd, cccd in enumerate(list_cccd):
            cccd_align = self.model_corner.detect_corner(cccd)
            if cccd_align is None:
                continue

            data_cccd = dict()
            data['code'] = 403
            label_classification = self.model_classification.classification_cccd(cccd_align)
            if label_classification == 'cccd_chip':
                dict_box_all = self.model_infor_chip.run(cccd_align)
            else:
                dict_box_all = self.model_infor_2024.run(cccd_align)

            for field, list_box_field in dict_box_all.items():
                data_cccd[field] = ''
                if len(list_box_field) != 0:
                    list_crop = crop_box(cccd_align, list_box_field)
                    list_crop = [crop for crop in list_crop if crop.size != 0]
                    if field == 'id':
                        for crop in list_crop:
                            text_crop = self.model_predict_number.predict_infor(crop)
                            text_crop = text_crop.replace('#', '/')
                            data_cccd[field] += text_crop
                        data_cccd[field] = ''.join(c for c in data_cccd[field] if c.isdigit())

                    elif field == 'ngaysinh' or field == 'giatriden':
                        for crop in list_crop:
                            text_crop = self.model_predict_number.predict_infor(crop)
                            text_crop = text_crop.replace('#', '/')
                            data_cccd[field] += ' ' + text_crop
                        list_text_field = data_cccd[field].split('/')
                        if len(list_text_field) > 4:
                            data_cccd[field] = 'Không giới hạn'

                    elif field == 'gioitinh':
                        for crop in list_crop:
                            text_crop = self.model_predict_text.predict_infor(crop)
                            text_crop = text_crop.replace('#', '/')
                            data_cccd[field] += ' ' + text_crop
                        # correct gioitinh
                        if (unidecode.unidecode(data_cccd[field].lower()) == 'nam') or ('a' in unidecode.unidecode(data_cccd[field].lower()) and len(data_cccd[field]) >= 3):
                            data_cccd[field] = 'Nam'
                        elif (unidecode.unidecode(data_cccd[field].lower()) == 'nu') or ('u' in unidecode.unidecode(data_cccd[field].lower()) and len(data_cccd[field]) <= 3):
                            data_cccd[field] = 'Nữ'

                    elif field == 'quoctich':
                        for crop in list_crop:
                            text_crop = self.model_predict_text.predict_infor(crop)
                            text_crop = text_crop.replace('#', '/')
                            data_cccd[field] += ' ' + text_crop
                        if len(data_cccd[field]) > 0:
                            data_cccd[field] = 'Việt Nam'
                            
                    elif field == 'quequan' or field == 'diachi':
                        for crop in list_crop:
                            text_crop = self.model_predict_text.predict_infor(crop)
                            text_crop = text_crop.replace('#', '/')
                            data_cccd[field] += ' ' + text_crop
                        # correct address
                        data_cccd[field] = correct_all(data_cccd[field])

                    else:
                        for crop in list_crop:
                            text_crop = self.model_predict_text.predict_infor(crop)
                            text_crop = text_crop.replace('#', '/')
                            data_cccd[field] += ' ' + text_crop

                data_cccd[field] = data_cccd[field].strip().title()

            data['data'].append(data_cccd)

        if len(data['data']) == 0:
            data['code'] = 402

        return data

# if __name__ == "__main__":
    # file_image = '/home/dtvingg/Desktop/test/2anh.jpeg'
    # model = MERGE()
    # image = cv2.imread(file_image)
    # start = time.time()
    # data = model.run(image)
    # end = time.time()
    #
    #
    #
    # print(data)
    # print(end-start)

    # dir_image = '/home/vinhdt/Desktop/data_test/2024'
    # model = MERGE()
    # list_link = glob.glob(os.path.join(dir_image, "*"))
    # data_excel = []
    # for i, file in enumerate(list_link):
    #     print('Index: ', i, file)
    #     image = cv2.imread(file)
    #     start = time.time()
    #     data, label_pl = model.run(image)
    #     end = time.time()
    #     print('*'*20, end-start, '*'*20)
    #     print(data)
    #
    #     if data['code'] == 403:
    #         for data_cccd in data['data']:
    #             if len(data_cccd.keys()) == 5:
    #                 data_excel.append([i + 1, file, label_pl, data_cccd['id'], data_cccd['hoten'], data_cccd['ngaysinh'], data_cccd['gioitinh'], data_cccd['quoctich'], '', '', ''])
    #             else:
    #                 data_excel.append([i + 1, file, label_pl, data_cccd['id'], data_cccd['hoten'], data_cccd['ngaysinh'], data_cccd['gioitinh'],data_cccd['quoctich'], data_cccd['quequan'], data_cccd['diachi'], data_cccd['giatriden']])
    #     else:
    #         data_excel.append([i + 1, file, '', '', '', '', '', '', '', '', ''])
    #
    # df = pd.DataFrame(data_excel, columns=['stt', 'link', 'phanloai', 'id', 'hoten', 'ngaysinh', 'gioitinh', 'quoctich', 'quequan', 'diachi', 'giatriden'])
    #
    # df.to_excel('output_v1.xlsx', index=False)
























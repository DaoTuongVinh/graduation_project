import cv2

from infor_detector import INFOR_DETECTOR

model = INFOR_DETECTOR('infor_chip')

image = cv2.imread('/home/dtvingg/Desktop/Doan_20241/data/test_b2.jpg')

dict_box_all  = model.run(image)

colors = {
    'id': (255,0,0),
    'hoten': (0,255,0),
    'ngaysinh': (0,0,255),
    'gioitinh': (255, 255, 0),
    'quoctich': (255, 0, 255),
    'quequan': (0, 255, 255),
    'diachi': (0,0,0),
    'giatriden': (255, 55, 25)
}

for field, list_box in dict_box_all.items():
    for box in list_box:
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), colors[field], 2)
        # cv2.putText(image, f'{field}', (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

cv2.imwrite('/home/dtvingg/Desktop/Doan_20241/data/b3.jpg', image)
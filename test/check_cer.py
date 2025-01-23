from difflib import SequenceMatcher
import pandas as pd

def calculate_cer(ground_truth, predicted_text):
    # Tìm số phép thay thế, thêm và xóa
    matcher = SequenceMatcher(None, ground_truth, predicted_text)
    substitutions, insertions, deletions = 0, 0, 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            substitutions += max(i2 - i1, j2 - j1)  # Ký tự thay thế
        elif tag == 'insert':
            insertions += (j2 - j1)  # Ký tự thêm
        elif tag == 'delete':
            deletions += (i2 - i1)  # Ký tự xóa

    # Tổng số ký tự trong văn bản tham chiếu
    total_chars = len(ground_truth)

    # Tính CER
    cer = (substitutions + insertions + deletions) / total_chars
    return cer

dir_rs = '/home/vinhdt/Desktop/data_test/rs_chip.xlsx'
dir_predict = '/home/vinhdt/Desktop/data_test/predict_chip.xlsx'

df_rs = pd.read_excel(dir_rs)
df_predict = pd.read_excel(dir_predict)

# Lấy dữ liệu từ cột a và b
# list_tuples_rs = list(zip(df_rs['id'], df_rs['hoten'], df_rs['gioitinh'], df_rs['quoctich'], df_rs['ngaysinh']))
# list_tuples_predict = list(zip(df_predict['id'], df_predict['hoten'], df_predict['gioitinh'], df_predict['quoctich'], df_predict['ngaysinh']))
list_tuples_rs = list(zip(df_rs['id'], df_rs['hoten'], df_rs['gioitinh'], df_rs['quoctich'], df_rs['ngaysinh'], df_rs['diachi'], df_rs['quequan'], df_rs['giatriden']))
list_tuples_predict = list(zip(df_predict['id'], df_predict['hoten'], df_predict['gioitinh'], df_predict['quoctich'], df_predict['ngaysinh'], df_predict['diachi'], df_predict['quequan'], df_predict['giatriden']))

print(list_tuples_rs)
print(list_tuples_predict)

cer_all_tb = 0

for i, item in enumerate(list_tuples_rs):
    cer_all = 0
    for j, field in enumerate(item):
        cer = calculate_cer(str(list_tuples_rs[i][j]), str(list_tuples_predict[i][j]))
        cer_all += cer

    cer_all_tb += cer_all

print(cer_all_tb/len(list_tuples_rs))

import json
import numpy as np
import unidecode
import editdistance

with open('correct_address/data.json', 'r') as file_json:
    data = json.load(file_json)
file_json.close()
list_name_tinh = list(data.keys())

def correct_tinh(list_text_reverse):
    l_text = min(5, len(list_text_reverse))
    list_result = np.zeros((l_text, len(list_name_tinh)), dtype=int)
    for i in range(0, l_text):
        index = i
        list_text_cp = []
        while 1:
            list_text_cp.append(list_text_reverse[index])
            index = index - 1
            if index == -1:
                break
        text_cp = " ".join(list_text_cp)
        for k in range(0, len(list_name_tinh)):
            text_input = unidecode.unidecode(text_cp)
            text_compare = unidecode.unidecode(list_name_tinh[k])
            list_result[i][k] = editdistance.eval(text_input, text_compare)

    list_min = []
    for i in range(0, l_text):
        index = np.argmin(list_result[i])
        text_arr = list_text_reverse[0:i + 1]
        text_arr.reverse()
        list_min.append([list_name_tinh[index], " ".join(text_arr), i, list_result[i][index]])

    list_min = sorted(list_min, key=lambda x: (x[3], -len(x[1])))
    best = list_min[0]
    if best[3] < 2:
        return best

    return None


def correct_huyen(name_tinh, list_text_reverse):
    if len(list_text_reverse) == 0:
        return None
    list_name_huyen = list(data[name_tinh]["quan"].keys())
    l_text = min(5, len(list_text_reverse))
    list_result = np.zeros((l_text, len(list_name_huyen)), dtype=int)
    for i in range(0, l_text):
        index = i
        list_text_cp = []
        while 1:
            list_text_cp.append(list_text_reverse[index])
            index = index - 1
            if index == -1:
                break
        text_cp = " ".join(list_text_cp)
        for k in range(0, len(list_name_huyen)):
            text_input = unidecode.unidecode(text_cp)
            text_compare = unidecode.unidecode(list_name_huyen[k])
            list_result[i][k] = editdistance.eval(text_input, text_compare)

    list_min = []
    for i in range(0, l_text):
        index = np.argmin(list_result[i])
        text_arr = list_text_reverse[0:i + 1]
        text_arr.reverse()
        list_min.append([list_name_huyen[index], " ".join(text_arr), i, list_result[i][index]])

    list_min = sorted(list_min, key=lambda x: (x[3], -len(x[1])))
    best = list_min[0]
    if best[3] < 2:
        return best

    return None


def correct_xa(name_tinh, name_huyen, list_text_reverse):
    if len(list_text_reverse) == 0:
        return None
    list_name_xa = list(data[name_tinh]["quan"][name_huyen]["xa"].keys())
    l_text = min(5, len(list_text_reverse))
    list_result = np.zeros((l_text, len(list_name_xa)), dtype=int)
    for i in range(0, l_text):
        index = i
        list_text_cp = []
        while 1:
            list_text_cp.append(list_text_reverse[index])
            index = index - 1
            if index == -1:
                break
        text_cp = " ".join(list_text_cp)
        for k in range(0, len(list_name_xa)):
            text_input = unidecode.unidecode(text_cp)
            text_compare = unidecode.unidecode(list_name_xa[k])
            list_result[i][k] = editdistance.eval(text_input, text_compare)

    list_min = []
    for i in range(0, l_text):
        index = np.argmin(list_result[i])
        text_arr = list_text_reverse[0:i + 1]
        text_arr.reverse()
        list_min.append([list_name_xa[index], " ".join(text_arr), i, list_result[i][index]])

    list_min = sorted(list_min, key=lambda x: (x[3], -len(x[1])))
    best = list_min[0]
    if best[3] < 2:
        return best

    return None

def format_address(list_text, xa=None, huyen=None, tinh=None):
    list_text.reverse()
    components = []

    if list_text:
        components.append(' '.join(list_text))
    if xa:
        components.append(xa)
    if huyen:
        components.append(huyen)
    if tinh:
        components.append(tinh)

    return ', '.join(components).title()

def correct_all(text):
    text_process = text.lower().strip().replace(",", "")
    list_text = text_process.split(' ')
    list_text.reverse()

    best_tinh = correct_tinh(list_text)
    if not best_tinh:
        return text.title()

    name_tinh = best_tinh[0]
    list_text = list_text[best_tinh[2] + 1:]

    best_huyen = correct_huyen(name_tinh, list_text)
    if not best_huyen:
        return format_address(list_text, tinh=name_tinh)

    name_huyen = best_huyen[0]
    list_text = list_text[best_huyen[2] + 1:]

    best_xa = correct_xa(name_tinh, name_huyen, list_text)
    if not best_xa:
        return format_address(list_text, huyen=name_huyen, tinh=name_tinh)

    name_xa = best_xa[0]
    list_text = list_text[best_xa[2] + 1:]

    return format_address(list_text, xa=name_xa, huyen=name_huyen, tinh=name_tinh)






# def correct_all(text):
#     text_process = text.lower().strip().replace(",", "")
#     list_text = text_process.split(' ')
#     list_text.reverse()
#     best_tinh = correct_tinh(list_text)
#     if best_tinh is not None:
#         name_tinh = best_tinh[0]
#         list_text_1 = list_text.copy()
#         list_text_1 = list_text_1[best_tinh[2] + 1:]
#         best_huyen = correct_huyen(name_tinh, list_text_1)
#         if best_huyen is not None:
#             name_huyen = best_huyen[0]
#             list_text_2 = list_text_1.copy()
#             list_text_2 = list_text_2[best_huyen[2] + 1:]
#             best_xa = correct_xa(name_tinh, name_huyen, list_text_2)
#             if best_xa is not None:
#                 name_xa = best_xa[0]
#                 list_text_3 = list_text_2.copy()
#                 list_text_3 = list_text_3[best_xa[2] + 1:]
#                 list_text_3.reverse()
#                 if len(list_text_3) != 0:
#                     text_correct = ' '.join(list_text_3) + ', ' + name_xa + ', ' + name_huyen + ', ' + name_tinh
#                     text_correct = text_correct.title()
#                     return text_correct
#                 else:
#                     text_correct = name_xa + ', ' + name_huyen + ', ' + name_tinh
#                     text_correct = text_correct.title()
#                     return text_correct
#             else:
#                 list_text_2.reverse()
#                 if len(list_text_2) != 0:
#                     text_correct = ' '.join(list_text_2) + ', ' + name_huyen + ', ' + name_tinh
#                     text_correct = text_correct.title()
#                     return text_correct
#                 else:
#                     text_correct = name_huyen + ', ' + name_tinh
#                     text_correct = text_correct.title()
#                     return text_correct
#         else:
#             list_text_1.reverse()
#             if len(list_text_1) != 0:
#                 text_correct = ' '.join(list_text_1) + ', ' + name_tinh
#                 text_correct = text_correct.title()
#                 return text_correct
#             else:
#                 text_correct = name_tinh
#                 text_correct = text_correct.title()
#                 return text_correct
#     else:
#         return text
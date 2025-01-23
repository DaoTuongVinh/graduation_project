import time

import streamlit as st
import cv2
import numpy as np
import uuid

import warnings
warnings.filterwarnings('ignore')

from merge import MERGE
model = MERGE()

convert_title = {
    'id': 'Số / No.:',
    'hoten': 'Họ và tên / Full name:',
    'ngaysinh': 'Ngày sinh / Date of birth:',
    'gioitinh': 'Giới tính / Sex:',
    'quoctich': 'Quốc tịch / Nationality:',
    'quequan': 'Quê quán / Place of origin:',
    'diachi': 'Nơi thường trú / Place of residence:',
    'giatriden': 'Có giá trị đến / Date of expiry:'
}

# Tiêu đề của ứng dụng
st.title("Demo")

uploaded_file = st.file_uploader("Tải ảnh lên...", type=["jpg", 'jpeg', 'png'])

# Kiểm tra nếu có file ảnh được tải lên
if uploaded_file is not None:
    # Đọc ảnh bằng OpenCV
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.title('Ảnh tải lên:')
    st.image(image_rgb, use_container_width=True)

    start = time.time()
    data = model.run(image)
    end = time.time()

    if data['code'] == 401:
        st.warning("Ảnh không chứa Căn cước công dân!")

    elif data['code'] == 402:
        st.warning("Không xác định được 4 góc Căn cước công dân!")

    elif data['code'] == 403:
        st.success("Trích xuất thành công!")
        st.success(f"Thời gian trích xuất {end-start}s")
        if len(data['data']) == 1:
            st.header('Thông tin:')
            for field, value in data['data'][0].items():
                st.markdown(convert_title[field])  # Hiển thị tên trường
                st.markdown(f"""
                    <div style="
                        border: 1px solid black; 
                        padding: 10px; 
                        border-radius: 5px; 
                        background-color: #f9f9f9;
                        margin-bottom: 15px;">
                        {value}
                    </div>
                    """, unsafe_allow_html=True)

        else:
            for i, data_person in enumerate(data['data']):
                st.subheader(f'Thông tin người {i+1}:')
                for field, value in data_person.items():
                    st.markdown(convert_title[field])  # Hiển thị tên trường
                    st.markdown(f"""
                        <div style="
                            border: 1px solid black; 
                            padding: 10px; 
                            border-radius: 5px; 
                            background-color: #f9f9f9;
                            margin-bottom: 15px;">
                            {value}
                        </div>
                        """, unsafe_allow_html=True)


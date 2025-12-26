import streamlit as st
from deepface import DeepFace
from PIL import Image
import numpy as np
import cv2

st.title("AI Face Scanner 📸")

img_file = st.camera_input("Quét khuôn mặt")

if img_file is not None:
    img = Image.open(img_file)
    img_array = np.array(img)
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    with st.spinner('Đang phân tích...'):
        try:
            results = DeepFace.analyze(img_cv, actions=['age', 'gender', 'emotion'], enforce_detection=False)
            res = results[0]
            st.success("Xong!")
            st.metric("Tuổi", f"~{int(res['age'])}")
            st.metric("Giới tính", res['dominant_gender'])
            st.metric("Tâm trạng", res['dominant_emotion'])
        except Exception as e:
            st.error(f"Lỗi: {e}")

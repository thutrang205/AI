import streamlit as st
import numpy as np
import pickle
import pandas as pd
import os
import requests

GOOGLE_SHEET_WEBAPP_URL = "https://script.google.com/macros/s/AKfycbwDrurTh5cDK7lDnB3ZbvB9emPOpwpZU2JVL63CpOjvgI4DY7vZVJwHbNhee7IOns-k/exec"  

# Load model
with open('SP_rf.pkl', 'rb') as file:
    model = pickle.load(file)
    
# Load scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
    
st.title("Dự đoán điểm học tập của học sinh")

# Thêm biến trạng thái cho việc gửi đánh giá
if 'feedback_submitted' not in st.session_state:
    st.session_state['feedback_submitted'] = False
# Trạng thái dự đoán
if 'predicted' not in st.session_state:
    st.session_state['predicted'] = False
if 'prediction' not in st.session_state:
    st.session_state['prediction'] = None
    
if not st.session_state['feedback_submitted']:
    user_name = st.text_input("Nhập tên của bạn")

    # Nhập dữ liệu người dùng
    age = st.slider("Tuổi (age)", 15, 22)
    studytime = st.selectbox("Thời gian học ngoài giờ lên lớp  (studytime)", [1, 2, 3, 4], format_func=lambda x: ["<2h", "2-5h", "5-10h", ">10h"][x - 1])
    freetime = st.slider("Thời gian rảnh sau giờ học (freetime)", 1, 5)
    goout = st.slider("Mức độ đi chơi với bạn (goout): 1 - rất thấp ; 5 - rất cao", 1, 5)
    failures = st.slider("Số lần trượt môn (failures)", 0, 4)
    Medu = st.selectbox("Trình độ học vấn của mẹ (Medu)", [0, 1, 2, 3, 4], format_func=lambda x: ["Không có", "Tiểu học", "THCS", "THPT", "Đại học"][x])
    Fedu = st.selectbox("Trình độ học vấn của cha (Fedu)", [0, 1, 2, 3, 4], format_func=lambda x: ["Không có", "Tiểu học", "THCS", "THPT", "Đại học"][x])
    famrel = st.slider("Mối quan hệ trong gia đình (famrel): 1 - rất tệ ; 5 - rất tốt", 1, 5)
    Dalc = st.slider("Mức độ uống rượu ngày thường (Dalc): 1 - rất thấp ; 5 - rất cao", 1, 5)
    Walc = st.slider("Mức độ uống rượu cuối tuần (Walc): 1 - rất thấp ; 5 - rất cao", 1, 5)
    health = st.slider("Tình trạng sức khỏe hiện tại (health): 1 - rất kém ; 5 - rất tốt", 1, 5)
    absences = st.number_input("Số buổi nghỉ học (absences)", min_value=0)


    overall_health = (0.5 * Dalc + 0.5 * Walc + 2 * health + famrel) / 4
    # Encoding one-hot
    def binary_input(label, options):
        return st.radio(label, options)

    sex = binary_input("Giới tính (F: Nữ; M: Nam)", ["F", "M"])
    address = binary_input("Địa chỉ (U: Thành phố; R: Nông thôn)", ["U", "R"])
    famsize = binary_input("Quy mô gia đình (LE3 <=3 người; GT3 >3 người)", ["LE3", "GT3"])
    Pstatus = binary_input("Tình trạng sống của cha mẹ (T: Sống chung; A: Không sống chung)", ["T", "A"])
    Mjob = st.selectbox("Nghề nghiệp của mẹ (Mjob)", ["at_home", "health", "other", "services", "teacher"])
    Fjob = st.selectbox("Nghề nghiệp của cha (Fjob)", ["at_home", "health", "other", "services", "teacher"])
    schoolsup = binary_input("Hỗ trợ học tập ở trường (schoolsup)", ["yes", "no"])
    famsup = binary_input("Hỗ trợ học tập từ gia đình (famsup)", ["yes", "no"])
    paid = binary_input("Học thêm (paid)", ["yes", "no"])
    activities = binary_input("Hoạt động ngoại khóa (activities)", ["yes", "no"])
    nursery = binary_input("Đi học mẫu giáo (nursery)", ["yes", "no"])
    higher = binary_input("Muốn học đại học (higher)", ["yes", "no"])
    internet = binary_input("Có internet ở nhà (internet)", ["yes", "no"])
    romantic = binary_input("Đang trong mối quan hệ yêu đương (romantic)", ["yes", "no"])


    # Tạo vector đầu vào
    def one_hot(value, categories):
        return [1 if value == cat else 0 for cat in categories]

    G1 = st.slider("Điểm kiểm tra miệng (G1)", 0, 20)
    G2 = st.slider("Điểm giữa kỳ (G2)", 0, 20)
    input_data = [
        age, Medu, Fedu, studytime, failures, famrel, freetime, goout,
        Dalc, Walc, health, absences, G1, G2, overall_health #15
    ] \
    + one_hot(sex, ["F", "M"]) \
    + one_hot(address, ["R", "U"]) \
    + one_hot(famsize, ["GT3", "LE3"]) \
    + one_hot(Pstatus, ["A", "T"]) \
    + one_hot(Mjob, ["at_home", "health", "other", "services", "teacher"]) \
    + one_hot(Fjob, ["at_home", "health", "other", "services", "teacher"]) \
    + one_hot(schoolsup, ["no", "yes"]) \
    + one_hot(famsup, ["no", "yes"]) \
    + one_hot(paid, ["no", "yes"]) \
    + one_hot(activities, ["no", "yes"]) \
    + one_hot(nursery, ["no", "yes"]) \
    + one_hot(higher, ["no", "yes"]) \
    + one_hot(internet, ["no", "yes"]) \
    + one_hot(romantic, ["no", "yes"])


    # Dự đoán
    if st.button("Dự đoán điểm học tập"):
        if not user_name.strip():
            st.warning("Vui lòng nhập tên của bạn trước khi dự đoán!")
        else:
            input_array = np.array(input_data).reshape(1, -1)
            input_scaled = scaler.transform(input_array)
            prediction = model.predict(input_scaled)
            st.session_state['predicted'] = True
            st.session_state['prediction'] = prediction[0]
            
    if st.session_state['predicted']:
        st.success(f"{user_name}, dự đoán điểm cuối kỳ: {st.session_state['prediction']:.2f}")
            
    # Giao diện đánh giá
    if st.session_state['predicted'] and not st.session_state['feedback_submitted']:
        st.markdown("---")
        st.header("Đánh giá mô hình")
        st.subheader("Mức độ hài lòng chung:")
        user_feedback = st.slider(" Bạn có hài lòng với kết quả dự đoán của mô hình không? (Thang điểm: 1 – Rất không hài lòng → 5 – Rất hài lòng)", 1, 5)
        
        st.subheader("Đánh giá tính chính xác:")
        user_feedback1 = binary_input("Kết quả dự đoán có phản ánh đúng thực tế không?", ["Có", "Không"])
        user_feedback2 = binary_input("Bạn có tin tưởng vào độ tin cậy của mô hình không?", ["Có", "Không"])
        
        st.subheader("Giao diện & Trải nghiệm sử dụng:")
        user_feedback3 = binary_input("Giao diện trình bày kết quả có dễ hiểu không?", ["Có", "Không"])
        user_feedback4 = binary_input("Tốc độ xử lý của mô hình có đáp ứng nhu cầu của bạn không?", ["Có", "Không"])
        if st.button("Gửi đánh giá"):
            # Lưu đánh giá vào GG Sheet 
            feedback_data = {
                "Ten": user_name,
                "DuDoan": st.session_state['prediction'],
                "HaiLong": user_feedback,
                "DungThucTe": user_feedback1,
                "TinCay": user_feedback2,
                "DeHieu": user_feedback3,
                "TocDo": user_feedback4
            }
            try:
                requests.post(GOOGLE_SHEET_WEBAPP_URL, json=feedback_data)
                
                st.session_state['feedback_submitted'] = True
            except Exception as e:
                st.error("Không gửi được đánh giá lên Google Sheets. Lỗi: " + str(e))

 # Giao diện sau khi gửi đánh giá
if st.session_state['feedback_submitted']:
    st.success("Cảm ơn bạn đã đánh giá mô hình!")
    if st.button("Tiếp tục dự đoán"):
        st.session_state['predicted'] = False
        st.session_state['prediction'] = None
        st.session_state['feedback_submitted'] = False

from decision_tree_processor import predict_fc as dt_pred
from random_forest_processor import predict_fc as rd_pred
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from load_data import get_training_data
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X_train, y_train, X_val, y_val = get_training_data()

st.write("""
# Dự đoán rủi ro về khả năng tử vong của bệnh nhân trong 10 năm
""")
st.write("""
Trong những thập kỷ gần đây, với sự phát triển nhanh chóng của việc số hóa dữ liệu, lượng dữ liệu điện tử trong lĩnh vực y tế đã tăng hơn 300 lần từ 130 exabytes (năm 2005) lên 40,000 exabytes (năm 2020) [1]. Cùng với sự phát triển vượt bậc của công nghệ và các nghiên cứu về khai phá, phân tích và xây dựng tri thức đã giúp những chuyên gia và các y bác sĩ có nhận định chính xác hơn trong việc chẩn đoán và chăm sóc bệnh nhân. Những ứng dụng trong đời sống hiện nay có thể kể đến như: sử dụng tri thức khai phá để đưa ra quyết định lâm sàng; chẩn đoán mối đe dọa nghiêm trọng đối với chất lượng và an toàn trong chăm sóc sức khỏe; phân tích hồ sơ và quản lý khám, chữa bệnh của bệnh nhân [2].

Bên cạnh đó, sự thay đổi về môi trường sống của con người cũng đang có nhiều ảnh hưởng lớn đến sức khỏe trong tương lai của chính mỗi người chúng ta. Chính vì vậy, việc đưa ra dự đoán về khả năng bệnh tật hay rủi ro trong tương lai của con người có thể giúp chúng ta cải thiện hơn về mặt sức khỏe và đời sống. Do đó, trong đồ án môn học này, đề tài được khai phá và xây dựng cơ sở tri thức về “Dự đoán rủi ro về khả năng tử vong của bệnh nhân trong 10 năm tới” bằng các phương pháp học máy như cây quyết định (Decision Tree) [3] và rừng ngẫu nhiên (Random Forest) [4].

""")

uploaded_file = st.file_uploader("Choose csv file")

X = pd.DataFrame()

if uploaded_file is not None:
    datasets = pd.read_csv(uploaded_file)
    X = datasets
    st.write(datasets)

classifier_name = st.sidebar.selectbox(
    'Select classifier',
    ('Decision Tree', 'Random Forest')
)

st.write('Shape of dataset:', X.shape)

def add_parameter(clf_name):
    params = dict()
    if clf_name == 'Decision Tree':
        max_depth = st.sidebar.slider('max_depth', 2, 10)
        params['max_depth'] = max_depth
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 10)
        params['max_depth'] = max_depth
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators
        min_samples_leaf = st.sidebar.slider('min_samples_leaf', 1, 20)
        params['min_samples_leaf'] = min_samples_leaf
    return params

params = add_parameter(classifier_name)

def get_pred(clf_name):
    pred = None
    if clf_name == 'Decision Tree':
        decision_tree_with_max_depth = DecisionTreeClassifier(**params, random_state=42)
        decision_tree_with_max_depth.fit(X_train, y_train)
        pred = decision_tree_with_max_depth.predict_proba(X)[:, 1]
    else:
        random_forest_hypertuning = RandomForestClassifier(**params, random_state=42)
        random_forest_hypertuning.fit(X_train, y_train)
        pred = random_forest_hypertuning.predict_proba(X)[:, 1]
    return pred

if X.empty == True:
    st.write("Insert csv file")
else:
    pred = get_pred(classifier_name)
    X_risk = X.copy(deep=True)
    X_risk.loc[:, 'Risk'] = pred
    st.write(f'Classifier : {classifier_name}')
    st.write(f'Predict :', X_risk)

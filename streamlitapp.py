import streamlit as st

# 载入包
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, roc_auc_score, roc_curve, auc, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score

import joblib # 如果没有安装请使用pip安装

# 解决画图中文显示问题
#plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = True

# 标题设置 "XX预测系统"改成你想设置的
#st.markdown('<h1 style="text-align: center; color: white; background: #ff4b4b; font-size: 18px; border-radius: .5rem; margin-bottom: 15px;">XX预测系统</h1>', unsafe_allow_html=True)


# 缓存数据, 使用缓存此部分代码只执行一次，可大幅度优化运行时长
@st.cache_resource
def load_model_fun():
    # 读取数据 训练
    dataset = pd.read_csv('d2_smotetrain.csv')  
    x_train = pd.DataFrame(dataset.iloc[:, 0 : dataset.shape[1] - 1])
    y_train = pd.DataFrame(dataset.iloc[:, dataset.shape[1] - 1])
    
    # 读取数据 验证
    dataset = pd.read_csv('test_data22.csv')  
    x_test = pd.DataFrame(dataset.iloc[:, 0 : dataset.shape[1] - 1])
    y_test = pd.DataFrame(dataset.iloc[:, dataset.shape[1] - 1])

    # 导入模型
    loaded_model = joblib.load('model.pkl')

    # 确保 SVM 模型输出概率
    Svm_model_prob = lambda x: loaded_model.predict_proba(x)[:, 1]

    # 创建解释器
    explainer2 = shap.KernelExplainer(Svm_model_prob, x_train)

    # 计算 SHAP 值
    shap_values3 = explainer2.shap_values(x_test)

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Arial'
    plt.rcParams['font.size'] = 8
    plt.rcParams['font.weight'] = 'bold'
    
    return loaded_model, shap_values3, explainer2

x = "Gender,Time to FTS,Neck circumference,Thigh circumference,Time to FTSS,Waist circumference,Age,BMI".split(",")

if "data" not in st.session_state.keys():
    st.session_state["data"] = {i:None for i in x}

# 输入表单
with st.form("predict"):
    st.session_state["data"][x[0]] = st.selectbox(x[0], [1,2], index=0)
    st.session_state["data"][x[1]] = st.selectbox(x[1], [0, 1], index=0)
    st.session_state["data"][x[2]] = st.number_input(x[1], step=0.1, placeholder=x[1], value=30.0, min_value=0.0, max_value=100.0)
    st.session_state["data"][x[3]] = st.number_input(x[2], step=0.01, placeholder=x[2], value=50.00, min_value=0.00, max_value=100.00)
    st.session_state["data"][x[4]] = st.selectbox(x[4], [1, 2, 3, 4], index=0)
    st.session_state["data"][x[5]] = st.number_input(x[5], step=0.1, placeholder=x[5], value=80.0, min_value=0.0, max_value=200.0)
    st.session_state["data"][x[6]] = st.number_input(x[6], step=1, placeholder=x[6], value=60, min_value=0, max_value=100)
    st.session_state["data"][x[7]] = st.number_input(x[7], step=0.01, placeholder=x[7], value=25.00, min_value=0.00, max_value=100.00)

    col = st.columns(5)
    submit = col[2].form_submit_button("Start Predict", use_container_width=True) # 预测按钮

model, shap_values3, explainer2 = load_model_fun() # 获取模型与shap图

with st.expander("Predict result", True):
    r = model.predict(np.array([list(st.session_state["data"].values())]))
    p_r = model.predict_proba(np.array([list(st.session_state["data"].values())])) # 预测结果
    p_r = (p_r[0]*100)[0] if r[0]==0 else (p_r[0]*100)[1] # 预测概率
    st.info(f"The predicted probability of sarcopenic obesity is {str(round(p_r, 2))}%.") # 展示预测结果
    
    # 单样本特征影响图
    shap_values = explainer2.shap_values(pd.DataFrame([st.session_state["data"]]))
    shap.force_plot(explainer2.expected_value, shap_values[0], pd.DataFrame([st.session_state["data"]]).iloc[0, :], matplotlib=True)
    st.pyplot(plt.gcf()) # 展示shap图

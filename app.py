
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
import datetime
import streamlit as st 
from ml import main 

st.set_option('deprecation.showfileUploaderEncoding', False)

st.markdown("# 이더리움 주식 분석기")

st.sidebar.subheader("파일 업로드")

uploaded_file = st.sidebar.file_uploader(label="Upload your csv or Excel File.",
type=['csv','xlsx'])

global df
if uploaded_file is not None:
    print(uploaded_file)
    try:
        df  = pd.read_csv(uploaded_file)
    except Exception as e:
        print(e)
        df = pd.read_excel(uploaded_file)

try:
    st.write(df)
except Exception as e:
    print(e)
    st.write("파일을 업로드 해 주세요")
data = main(df)
#print(data)
st.line_chart(data)
# st.line_chart(y_test)
 



import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
import datetime
import streamlit as st 
from ml import main 
import altair as alt
from fbprophet import Prophet
from fbprophet.plot import plot_plotly


st.set_option('deprecation.showfileUploaderEncoding', False)

st.markdown("# 이더리움 주식 분석기")

st.sidebar.subheader("파일 업로드")

uploaded_file = st.sidebar.file_uploader(label="Upload your csv or Excel File.",
type=['csv','xlsx'])


if uploaded_file is not None:
    print(uploaded_file)
    try:
        df  = pd.read_csv(uploaded_file)
    except Exception as e:
        print(e)
        df = pd.read_excel(uploaded_file)

try:
    st.subheader('분석할 데이터')
    st.write(df)
except Exception as e:
    print(e)
    st.write("파일을 업로드 해 주세요")


def do_lstm():
    data = main(df)
    st.line_chart(data)
    st.success("시세 예측 완료!")
    st.balloons()

def do_prophet():
    n_days = 7
    # Predict forecast with Prophet.
    df_train = df[['Date','Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=n_days)
    forecast = m.predict(future)
    # Show and plot forecast
    st.subheader('Forecast data')
    st.write(forecast.tail())
        
    st.write(f'시세 예측 그래프: {n_days} 일')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)

    # st.write("Forecast components")
    # fig2 = m.plot_components(forecast)
    # st.write(fig2)
    st.success("시세 예측 완료!")
    st.balloons()


with st.spinner("파일을 분석 중입니다... 잠시만 기다려주세요.."):
    try:
        do_prophet()
    except:
        pass
#print(data)



# st.altair_chart(line_chart)
# st.line_chart(y_test)
 


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
import datetime
def normalize_windows(data):
    normalized_data = []
    for window in data:
        normalized_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalized_data.append(normalized_window)
    return np.array(normalized_data)


def pre_processing(df):
    print('전처리 진행 중')
    high_prices = df['High'].values
    low_prices = df['Low'].values
    mid_prices = (high_prices + low_prices) / 2

    standard_value = mid_prices[0]


    seq_len = 50
    sequence_length = seq_len + 1

    result = []
    for index in range(len(mid_prices) - sequence_length):
        result.append(mid_prices[index: index + sequence_length])



    result = normalize_windows(result)

    # split train and test data
    row = int(round(result.shape[0] * 0.7))
    train = result[:row, :]
    np.random.shuffle(train)

    x_train = train[:, :-1]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    y_train = train[:, -1]

    x_test = result[row:, :-1]
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    y_test = result[row:, -1]
    

    return x_test, x_train, y_test, y_train, standard_value
 

def build_model(days=50):
    print('모델 생성 중')
    model = Sequential()

    model.add(LSTM(days, return_sequences=True, input_shape=(days, 1)))

    model.add(LSTM(64, return_sequences=False))

    model.add(Dense(1, activation='linear'))

    model.compile(loss='mse', optimizer='rmsprop')

    model.summary()

    return model 


def train_model(model, x_test, x_train, y_test, y_train):
    print('모델 학습 중')
    start_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

    model.fit(x_train, y_train,
        validation_data=(x_test, y_test),
        batch_size=10,
        epochs=2,
        callbacks=[
            TensorBoard(log_dir='logs/%s' % (start_time)),
            ModelCheckpoint('./models/%s_eth.h5' % (start_time), monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, mode='auto')
    ])

    return model 

# pred = model.predict(x_test)

# fig = plt.figure(facecolor='white', figsize=(20, 10))
# ax = fig.add_subplot(111)
# ax.plot(y_test, label='True')
# ax.plot(pred, label='Prediction')
# ax.legend()
# plt.show()


def main(df):
    x_test, x_train, y_test, y_train, standard_value = pre_processing(df)
    model = build_model()
    model = train_model(model, x_test, x_train, y_test, y_train)
    pred = model.predict(x_test)
    final_price = (pred + 1) * float(standard_value)
    true_price = (y_test + 1) * float(standard_value)

    data =pd.DataFrame(
        [[i[0][0],i[1]] for i in zip(final_price, true_price)],
        columns=['pred','actual']
    )


    return data